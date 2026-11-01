import hashlib
import json
import time
from collections import deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functions.etl import clean_quotes as fx
from functions.volatility import multiscale_volatility as utilities

NS = 1_000_000_000
STEP = 10 * NS
PIP = 0.0001

WINDOWS = {"3m": 18, "5m": 30, "15m": 90, "30m": 180, "60m": 360}
SPREAD_STEPS = 30
MAX_PRIOR = max(WINDOWS.values()) + SPREAD_STEPS - 1

MATCH_ATOL = 1e-8
ENTROPY_ATOL = 1e-10

SOURCE_RUN = "sampling_02_v1"
SOURCE_FP = (
    "ad5235dc14f854d9f26ef04dd2c1fdf846faf05cff43eb9f84afccc229628e54"
)

SOURCE_COLUMNS = [
    "sample_ns", "available_ns", "mid", "spread",
    "has_price", "is_interpolated", "in_long_gap",
    "project_excluded", "project_eligible",
]


def dependencies():
    return {
        "entropy_helper": utilities.sha(__file__),
        "utility_helper": utilities.sha(utilities.__file__),
        "etl_helper": utilities.sha(fx.__file__),
    }


def prepare(root, run_id, stop_after_date, memory="2GB", threads=2):
    root = Path(root).resolve()
    assert run_id and Path(run_id).name == run_id

    source = root / "data" / "processed" / SOURCE_RUN
    reports = root / "reports" / "tables" / "sampling" / SOURCE_RUN

    source_config = fx.load_json(source / "configuration.json")
    source_state = fx.load_json(source / "run_state.json")
    source_summary = fx.load_json(reports / "execution_summary.json")
    completion = fx.load_json(reports / "full_history_completion_checks.json")

    assert source_state["fingerprint"] == SOURCE_FP
    assert source_state["status"] == "full_history_processed"
    assert source_summary["fingerprint"] == SOURCE_FP
    assert source_summary["full_history_complete"]
    assert source_summary["nonzero_machine_checks"] == 0
    assert completion["fingerprint"] == SOURCE_FP
    assert completion["machine_checks_passed"]
    assert all(completion["checks"].values())

    first_real = int(source_config["inputs"][0]["first_ns"])
    last_real = int(source_config["inputs"][-1]["last_ns"])

    inputs = []
    for checkpoint_path in sorted((source / "_checkpoints").glob("*.json")):
        month = checkpoint_path.stem
        checkpoint = fx.load_json(checkpoint_path)

        assert checkpoint["fingerprint"] == SOURCE_FP
        assert all(value == 0 for value in checkpoint["checks"].values())

        path = source / "sampled_10s" / f"{month}.parquet"
        actual = utilities.file_info(path)
        expected = checkpoint["outputs"]["sampled_10s"]

        assert all(actual[key] == expected[key] for key in actual), path

        schema = pq.read_schema(path)
        assert set(SOURCE_COLUMNS).issubset(schema.names), path
        for name in ("sample_ns", "available_ns"):
            assert pa.types.is_int64(schema.field(name).type), (path, name)
        for name in ("mid", "spread"):
            assert pa.types.is_floating(schema.field(name).type), (path, name)

        lower, upper = fx.month_bounds(month)
        first = max(lower, ((first_real + STEP - 1) // STEP) * STEP)
        last = min(upper - STEP, (last_real // STEP) * STEP)

        assert actual["rows"] == (last - first) // STEP + 1

        inputs.append({
            "year_month": month,
            "path": path.relative_to(root).as_posix(),
            **actual,
            "first_ns": first,
            "last_ns": last,
            "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": utilities.sha(checkpoint_path),
        })

    months = [item["year_month"] for item in inputs]
    assert months == [
        str(period)
        for period in pd.period_range(months[0], months[-1], freq="M")
    ]
    assert len(inputs) == source_summary["processed_months"]
    assert sum(item["rows"] for item in inputs) == source_summary["sample_rows"]

    first = pd.Timestamp(inputs[0]["first_ns"], unit="ns", tz="UTC")
    last = pd.Timestamp(inputs[-1]["last_ns"], unit="ns", tz="UTC")
    first = first.tz_convert("Europe/Athens")
    last = last.tz_convert("Europe/Athens")

    jobs = []
    for day in pd.date_range(first.normalize(), last.normalize(), freq="D"):
        lower = max(int(day.value), inputs[0]["first_ns"])
        upper = min(
            int((day + pd.DateOffset(days=1)).value),
            inputs[-1]["last_ns"] + STEP,
        )
        if lower < upper:
            jobs.append({
                "date": day.strftime("%Y-%m-%d"),
                "lower_ns": lower,
                "upper_ns": upper,
                "rows": (upper - lower) // STEP,
            })

    if stop_after_date is not None:
        assert stop_after_date in [item["date"] for item in jobs]

    run = root / "outputs" / "features" / "entropy" / run_id
    tables = root / "reports" / "tables" / "entropy" / run_id
    figures = root / "reports" / "figures" / "entropy" / run_id

    configuration = {
        "definition": "entropy_v1",
        "run_id": run_id,
        "source_run": SOURCE_RUN,
        "source_fingerprint": SOURCE_FP,
        "source_configuration_sha256": utilities.sha(source / "configuration.json"),
        "source_contract_sha256": utilities.sha(source / "output_contract.json"),
        "dependencies": dependencies(),
        "versions": utilities.versions(),
        "inputs": inputs,
        "step_seconds": 10,
        "windows_changes": WINDOWS,
        "spread_change_seconds": 300,
        "price_symbols": "down, unchanged, up",
        "conditional_history_symbols": 1,
        "conditional_score": "H(current|previous)/log2(3)",
        "conditional_diagnostic": "H(current|previous)/H(current); zero if constant",
        "sample_pattern_length": 2,
        "sample_tolerance_fraction": 0.2,
        "sample_sd": "population, ddof=0, within current trailing window",
        "sample_common_template_starts": "0 through N-3",
        "sample_self_matches": False,
        "sample_overlapping_templates": True,
        "sample_score": "1-A/B; missing if B=0",
        "transfer_source": "previous five-minute spread-change sign",
        "transfer_target": "following ten-second price-change sign",
        "transfer_condition": "previous ten-second price-change sign",
        "transfer_score": "TE/H(current price sign|previous price sign)",
        "transfer_zero_denominator": "missing with flag",
        "full_fresh_window_after_missing": True,
        "excluded_hours_with_prices_continue": True,
        "interpolation_eligible": True,
        "comparison_tolerance_pips": MATCH_ATOL,
        "entropy_roundoff_tolerance": ENTROPY_ATOL,
        "label_boundaries": [1 / 3, 2 / 3],
        "learned_thresholds": False,
        "pip_size": PIP,
    }

    fingerprint = hashlib.sha256(
        json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    for directory in (run, tables, figures, run / "_checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)

    config_path = run / "configuration.json"
    if config_path.exists():
        existing = fx.load_json(config_path)
        changed = [
            key for key in sorted(set(existing) | set(configuration))
            if existing.get(key) != configuration.get(key)
        ]
        assert not changed, (
            f"Configuration changed in {changed}. Use a new RUN_ID."
        )
    else:
        fx.dump_json(config_path, configuration)

    pd.DataFrame(inputs).to_csv(tables / "input_inventory.csv", index=False)

    return {
        "root": root, "run": run, "tables": tables, "figures": figures,
        "run_id": run_id, "fingerprint": fingerprint,
        "configuration": configuration, "inputs": inputs, "jobs": jobs,
        "target_jobs": [
            job for job in jobs
            if stop_after_date is None or job["date"] <= stop_after_date
        ],
        "memory": memory, "threads": threads,
    }


def check_snapshot(ctx):
    source = ctx["root"] / "data" / "processed" / SOURCE_RUN
    config = ctx["configuration"]

    assert dependencies() == config["dependencies"]
    assert utilities.versions() == config["versions"]
    assert utilities.sha(source / "configuration.json") == (
        config["source_configuration_sha256"]
    )
    assert utilities.sha(source / "output_contract.json") == (
        config["source_contract_sha256"]
    )

    for item in ctx["inputs"]:
        actual = utilities.file_info(ctx["root"] / item["path"])
        assert all(actual[key] == item[key] for key in actual), item["path"]
        assert utilities.sha(ctx["root"] / item["checkpoint_path"]) == (
            item["checkpoint_sha256"]
        )


def values(table, name, fill=None):
    column = table[name]
    if fill is not None:
        column = column.fill_null(fill)
    return column.to_numpy(zero_copy_only=False)


def arrow(values_array, valid=None):
    return pa.array(
        values_array,
        mask=None if valid is None else ~valid,
        from_pandas=True,
    )


def symbols(changes):
    result = np.full(len(changes), -1, dtype=np.int8)
    valid = np.isfinite(changes)
    result[valid] = np.sign(changes[valid]).astype(np.int8) + 1
    result[valid & (np.abs(changes) <= MATCH_ATOL)] = 1
    return result


def labels(score):
    result = np.full(len(score), "unavailable", dtype=object)
    valid = np.isfinite(score)
    result[valid & (score < 1 / 3)] = "low"
    result[valid & (score >= 1 / 3) & (score < 2 / 3)] = "medium"
    result[valid & (score >= 2 / 3)] = "high"
    return result


def rolling_sum(data, width):
    cumulative = np.r_[0, np.cumsum(data, dtype=np.int64)]
    end = np.arange(1, len(data) + 1)
    return cumulative[end] - cumulative[np.maximum(0, end - width)]


def rolling_counts(ids, categories, width):
    data = np.zeros((len(ids), categories), dtype=np.int64)
    valid = ids >= 0
    data[np.flatnonzero(valid), ids[valid]] = 1
    cumulative = np.vstack([
        np.zeros((1, categories), dtype=np.int64),
        np.cumsum(data, axis=0),
    ])
    end = np.arange(1, len(ids) + 1)
    return cumulative[end] - cumulative[np.maximum(0, end - width)]


def rolling_max_integer(data, width):
    result = np.empty(len(data), dtype=np.int64)
    queue = deque()
    for index, value in enumerate(data):
        while queue and queue[0] <= index - width:
            queue.popleft()
        while queue and data[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        result[index] = data[queue[0]]
    return result


def entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1)
    log_counts = np.zeros_like(counts)
    np.log2(counts, out=log_counts, where=counts > 0)

    result = np.zeros_like(total)
    valid = total > 0
    result[valid] = (
        np.log2(total[valid])
        - (counts * log_counts).sum(axis=-1)[valid] / total[valid]
    )
    assert np.all(result >= -ENTROPY_ATOL)
    return np.maximum(result, 0)


def conditional_statistics(counts):
    joint = entropy(counts.reshape(len(counts), -1))
    previous = entropy(counts.sum(axis=2))
    target = entropy(counts.sum(axis=1))
    conditional = joint - previous

    assert np.all(conditional >= -ENTROPY_ATOL)
    assert np.all(conditional <= target + ENTROPY_ATOL)
    return np.maximum(conditional, 0), target


def transfer_statistics(cube):
    baseline, _ = conditional_statistics(cube.sum(axis=2))
    joint = entropy(cube.reshape(len(cube), -1))
    context = entropy(cube.sum(axis=3).reshape(len(cube), -1))
    with_spread = joint - context
    transfer = baseline - with_spread

    assert np.all(with_spread >= -ENTROPY_ATOL)
    assert np.all(transfer >= -ENTROPY_ATOL)
    assert np.all(transfer <= baseline + ENTROPY_ATOL)
    return baseline, np.maximum(transfer, 0)


def bounded(score):
    valid = np.isfinite(score)
    assert np.all(score[valid] >= -ENTROPY_ATOL)
    assert np.all(score[valid] <= 1 + ENTROPY_ATOL)
    result = score.copy()
    result[valid] = np.clip(result[valid], 0, 1)
    return result


def sample_counts(changes, tolerance):
    # Exact matching with first-coordinate candidate pruning.
    # Duplicate templates retain their full pair multiplicities.
    changes = np.asarray(changes, dtype=np.float64)
    assert changes.ndim == 1 and np.isfinite(changes).all()
    assert np.isfinite(tolerance) and tolerance >= 0

    templates = np.lib.stride_tricks.sliding_window_view(changes, 3)
    points, multiplicity = np.unique(
        templates, axis=0, return_counts=True
    )
    multiplicity = multiplicity.astype(np.int64, copy=False)

    order = np.argsort(points[:, 0], kind="stable")
    points = points[order]
    multiplicity = multiplicity[order]

    internal = int(np.sum(multiplicity * (multiplicity - 1) // 2))
    B = A = internal
    groups = len(points)

    if groups > 1:
        limit = float(tolerance) + MATCH_ATOL
        first = points[:, 0]
        scale = max(1.0, float(np.max(np.abs(first))), abs(limit))
        padding = 8 * np.finfo(np.float64).eps * scale

        upper = np.searchsorted(
            first,
            np.nextafter(first + limit + padding, np.inf),
            side="right",
        )
        index = np.arange(groups, dtype=np.int64)
        counts = np.maximum(upper - index - 1, 0)
        total = int(counts.sum())

        if total:
            left = np.repeat(index, counts)
            segment_start = np.cumsum(counts) - counts
            offset = np.arange(total) - np.repeat(segment_start, counts)
            right = left + 1 + offset
            weights = multiplicity[left] * multiplicity[right]

            match_two = (
                (np.abs(points[left, 0] - points[right, 0]) <= limit)
                & (np.abs(points[left, 1] - points[right, 1]) <= limit)
            )
            match_three = (
                match_two
                & (np.abs(points[left, 2] - points[right, 2]) <= limit)
            )

            B += int(weights[match_two].sum())
            A += int(weights[match_three].sum())

    maximum = len(templates) * (len(templates) - 1) // 2
    assert 0 <= A <= B <= maximum
    return B, A


def compute(source):
    ts = values(source, "sample_ns")
    available = values(source, "available_ns")
    mid = values(source, "mid")
    spread = values(source, "spread")

    has = values(source, "has_price", False).astype(bool)
    interpolated = values(source, "is_interpolated", False).astype(bool)
    excluded = values(source, "project_excluded", False).astype(bool)
    eligible = values(source, "project_eligible", False).astype(bool)
    long_gap = values(source, "in_long_gap", False).astype(bool)

    checks = {
        "duplicate_input_times": len(ts) - len(np.unique(ts)),
        "broken_input_grid": int(np.count_nonzero(np.diff(ts) != STEP)),
        "invalid_input_timing": int(np.count_nonzero(
            (available < ts) | (ts % STEP != 0)
        )),
        "missing_required_input_fields": sum(
            source[name].null_count
            for name in (
                "sample_ns", "available_ns", "has_price",
                "in_long_gap", "project_excluded", "project_eligible",
            )
        ),
        "invalid_present_prices": int(np.count_nonzero(
            has & (
                ~np.isfinite(mid) | (mid <= 0)
                | ~np.isfinite(spread) | (spread < -MATCH_ATOL * PIP)
            )
        )),
        "missing_interpolation_flags": int(np.count_nonzero(
            has & source["is_interpolated"].is_null().to_numpy()
        )),
        "prices_on_missing_rows": int(np.count_nonzero(
            ~has & source["mid"].is_valid().to_numpy()
        )),
        "prices_inside_long_gaps": int(np.count_nonzero(has & long_gap)),
        "inconsistent_project_scope": int(np.count_nonzero(
            eligible & (~has | excluded)
        )),
    }
    assert all(value == 0 for value in checks.values()), checks

    index = np.arange(len(ts))
    last_missing = np.maximum.accumulate(np.where(has, -1, index))
    fresh = np.where(has, index - last_missing - 1, 0)

    change = np.full(len(ts), np.nan)
    change[1:] = (mid[1:] - mid[:-1]) / PIP
    change[fresh == 0] = np.nan
    change[np.isfinite(change) & (np.abs(change) <= MATCH_ATOL)] = 0

    spread_change = np.full(len(ts), np.nan)
    spread_change[SPREAD_STEPS:] = (
        spread[SPREAD_STEPS:] - spread[:-SPREAD_STEPS]
    ) / PIP
    spread_change[fresh < SPREAD_STEPS] = np.nan
    spread_change[
        np.isfinite(spread_change)
        & (np.abs(spread_change) <= MATCH_ATOL)
    ] = 0

    price_symbol = symbols(change)
    spread_symbol = symbols(spread_change)
    previous_price = np.r_[-1, price_symbol[:-1]]
    previous_spread = np.r_[-1, spread_symbol[:-1]]

    ce_id = np.full(len(ts), -1, dtype=np.int64)
    valid = (price_symbol >= 0) & (previous_price >= 0)
    ce_id[valid] = previous_price[valid] * 3 + price_symbol[valid]

    te_id = np.full(len(ts), -1, dtype=np.int64)
    valid = valid & (previous_spread >= 0)
    te_id[valid] = (
        previous_price[valid] * 9
        + previous_spread[valid] * 3
        + price_symbol[valid]
    )

    interpolated_change = (
        np.isfinite(change)
        & (interpolated | np.r_[False, interpolated[:-1]])
    )

    output = {
        name: source[name]
        for name in SOURCE_COLUMNS
        if name != "available_ns"
    }
    output.update({
        "input_available_ns": source["available_ns"],
        "available_ns": arrow(
            rolling_max_integer(available, MAX_PRIOR + 1)
        ),
        "change_10s_pips": arrow(change),
        "price_symbol": arrow(price_symbol, price_symbol >= 0),
        "spread_change_5m_pips": arrow(spread_change),
        "spread_change_5m_symbol": arrow(
            spread_symbol, spread_symbol >= 0
        ),
        "restart": arrow(has & (fresh == 0)),
        "fresh_changes_capped": arrow(np.minimum(fresh, MAX_PRIOR)),
    })

    timings = {}

    for key, n in WINDOWS.items():
        started = time.perf_counter()
        price_ready = fresh >= n
        te_ready = fresh >= n + SPREAD_STEPS - 1

        C = rolling_counts(ce_id, 9, n - 1).reshape(-1, 3, 3)
        h_conditional, h_target = conditional_statistics(C)

        ce_score = bounded(h_conditional / np.log2(3))
        residual = np.divide(
            h_conditional, h_target,
            out=np.zeros(len(ts)),
            where=h_target > ENTROPY_ATOL,
        )
        residual = bounded(residual)

        cube = rolling_counts(te_id, 27, n - 1).reshape(-1, 3, 3, 3)
        te_baseline, te_bits = transfer_statistics(cube)

        assert np.all(
            np.abs(te_baseline[te_ready] - h_conditional[te_ready])
            <= ENTROPY_ATOL
        )

        te_zero = te_baseline <= ENTROPY_ATOL
        te_fraction = np.divide(
            te_bits, te_baseline,
            out=np.full(len(ts), np.nan),
            where=te_ready & ~te_zero,
        )
        te_fraction = bounded(te_fraction)

        B = np.full(len(ts), -1, dtype=np.int64)
        A = np.full(len(ts), -1, dtype=np.int64)
        sd = np.full(len(ts), np.nan)

        for end in np.flatnonzero(price_ready):
            window = change[end - n + 1:end + 1]
            sd[end] = np.std(window, ddof=0)
            B[end], A[end] = sample_counts(window, 0.2 * sd[end])

        sample_score = np.full(len(ts), np.nan)
        raw_sample = np.full(len(ts), np.nan)

        matched = B > 0
        extended = matched & (A > 0)
        infinite = matched & (A == 0)

        sample_score[matched] = 1 - A[matched] / B[matched]
        raw_sample[extended] = -np.log(A[extended] / B[extended])
        raw_sample[infinite] = np.inf

        sample_status = np.full(len(ts), "warmup", dtype=object)
        sample_status[price_ready & (B == 0)] = "no_matches"
        sample_status[extended] = "finite"
        sample_status[infinite] = "infinite_raw"

        def save_float(name, data, ready):
            saved = np.asarray(data, dtype=np.float64).copy()
            saved[~ready] = np.nan
            output[f"{name}_{key}"] = arrow(saved)
            return saved

        saved_ce = save_float("ce_score", ce_score, price_ready)
        save_float("ce_bits", h_conditional, price_ready)
        save_float("ce_residual_fraction", residual, price_ready)

        save_float("sampen_sd_pips", sd, price_ready)
        save_float("sampen_tolerance_pips", 0.2 * sd, price_ready)
        saved_sample = save_float("sampen_score", sample_score, price_ready)
        save_float("sampen_raw", raw_sample, price_ready)

        save_float("te_bits", te_bits, te_ready)
        saved_te = save_float("te_fraction", te_fraction, te_ready)

        save_float(
            "interp_change_fraction",
            rolling_sum(interpolated_change, n) / n,
            price_ready,
        )
        save_float(
            "zero_change_fraction",
            rolling_sum(price_symbol == 1, n) / n,
            price_ready,
        )
        save_float(
            "te_support_interp_fraction",
            rolling_sum(interpolated, n + SPREAD_STEPS)
            / (n + SPREAD_STEPS),
            te_ready,
        )

        price_available = rolling_max_integer(available, n + 1)
        te_available = rolling_max_integer(
            available, n + SPREAD_STEPS
        )

        output.update({
            f"ready_price_{key}": arrow(price_ready),
            f"ready_te_{key}": arrow(te_ready),
            f"ce_pairs_{key}": arrow(C.sum(axis=(1, 2))),
            f"ce_occupied_cells_{key}": arrow(
                np.count_nonzero(C, axis=(1, 2))
            ),
            f"ce_constant_target_{key}": arrow(
                h_target <= ENTROPY_ATOL, price_ready
            ),
            f"sampen_B_{key}": arrow(B, price_ready),
            f"sampen_A_{key}": arrow(A, price_ready),
            f"sampen_status_{key}": arrow(sample_status),
            f"te_pairs_{key}": arrow(cube.sum(axis=(1, 2, 3))),
            f"te_occupied_cells_{key}": arrow(
                np.count_nonzero(cube, axis=(1, 2, 3))
            ),
            f"te_zero_baseline_{key}": arrow(te_zero, te_ready),
            f"ce_available_ns_{key}": arrow(price_available),
            f"sampen_available_ns_{key}": arrow(price_available),
            f"te_available_ns_{key}": arrow(te_available),
            f"ce_label_{key}": arrow(labels(saved_ce)),
            f"sampen_label_{key}": arrow(labels(saved_sample)),
            f"te_information_label_{key}": arrow(labels(saved_te)),
        })

        timings[key] = time.perf_counter() - started

    return pa.table(output), checks, timings


def validate_output(table):
    issues = {}
    ts = values(table, "sample_ns")
    source_available = values(table, "input_available_ns")
    common_available = values(table, "available_ns")
    fresh = values(table, "fresh_changes_capped")

    issues["duplicate_output_times"] = len(ts) - len(np.unique(ts))
    issues["broken_output_grid"] = int(np.count_nonzero(
        np.diff(ts) != STEP
    ))
    issues["invalid_common_availability"] = int(np.count_nonzero(
        (common_available < ts) | (common_available < source_available)
    ))

    def count(name, condition):
        issues[name] = int(np.count_nonzero(condition))

    for key, n in WINDOWS.items():
        ready = values(table, f"ready_price_{key}")
        te_ready = values(table, f"ready_te_{key}")
        ce = values(table, f"ce_score_{key}")
        residual = values(table, f"ce_residual_fraction_{key}")
        sample = values(table, f"sampen_score_{key}")
        raw = values(table, f"sampen_raw_{key}")
        te = values(table, f"te_fraction_{key}")
        te_bits = values(table, f"te_bits_{key}")
        ce_bits = values(table, f"ce_bits_{key}")
        B = values(table, f"sampen_B_{key}", -1)
        A = values(table, f"sampen_A_{key}", -1)
        zero = values(table, f"te_zero_baseline_{key}", False)
        status = values(table, f"sampen_status_{key}")

        count(f"{key}_price_readiness", ready != (fresh >= n))
        count(
            f"{key}_te_readiness",
            te_ready != (fresh >= n + SPREAD_STEPS - 1),
        )
        count(
            f"{key}_ce_transition_count",
            ready & (values(table, f"ce_pairs_{key}") != n - 1),
        )
        count(
            f"{key}_te_transition_count",
            te_ready & (values(table, f"te_pairs_{key}") != n - 1),
        )

        for name, score in (
            ("ce", ce), ("ce_residual", residual),
            ("sampen", sample), ("te", te),
        ):
            count(
                f"{key}_{name}_range",
                np.isinf(score)
                | (np.isfinite(score) & ((score < 0) | (score > 1))),
            )

        count(f"{key}_ce_presence", np.isfinite(ce) != ready)
        count(
            f"{key}_sample_presence",
            np.isfinite(sample) != (ready & (B > 0)),
        )
        count(
            f"{key}_te_presence",
            np.isfinite(te) != (te_ready & ~zero),
        )
        count(
            f"{key}_sample_pair_order",
            ready & ((A < 0) | (B < A) | (B > (n - 2) * (n - 3) // 2)),
        )
        count(
            f"{key}_sample_score",
            (B > 0) & (
                np.abs(sample - (1 - A / np.maximum(B, 1)))
                > ENTROPY_ATOL
            ),
        )

        finite_raw = (B > 0) & (A > 0)
        expected_raw = np.zeros(len(ts))
        expected_raw[finite_raw] = -np.log(
            A[finite_raw] / B[finite_raw]
        )

        count(
            f"{key}_finite_raw_sample",
            finite_raw & (
                ~np.isfinite(raw)
                | (np.abs(raw - expected_raw) > ENTROPY_ATOL)
            ),
        )
        count(
            f"{key}_infinite_raw_sample",
            (B > 0) & (A == 0) & ~np.isposinf(raw),
        )
        count(
            f"{key}_undefined_raw_sample",
            (B <= 0) & ~np.isnan(raw),
        )

        expected_status = np.full(len(ts), "warmup", dtype=object)
        expected_status[ready & (B == 0)] = "no_matches"
        expected_status[finite_raw] = "finite"
        expected_status[(B > 0) & (A == 0)] = "infinite_raw"
        count(f"{key}_sample_status", status != expected_status)

        count(
            f"{key}_te_information_bound",
            te_ready & (
                (te_bits < -ENTROPY_ATOL)
                | (te_bits > ce_bits + ENTROPY_ATOL)
            ),
        )

        for name, score in (
            ("ce_label", ce),
            ("sampen_label", sample),
            ("te_information_label", te),
        ):
            count(
                f"{key}_{name}",
                values(table, f"{name}_{key}") != labels(score),
            )

        for family in ("ce", "sampen", "te"):
            available = values(table, f"{family}_available_ns_{key}")
            count(
                f"{key}_{family}_availability",
                (available < ts)
                | (available < source_available)
                | (available > common_available),
            )

        for name in (
            "interp_change_fraction",
            "zero_change_fraction",
            "te_support_interp_fraction",
        ):
            score = values(table, f"{name}_{key}")
            count(
                f"{key}_{name}_range",
                np.isinf(score)
                | (np.isfinite(score) & ((score < 0) | (score > 1))),
            )

    assert all(value == 0 for value in issues.values()), {
        key: value for key, value in issues.items() if value
    }
    return issues


def synthetic_checks(ctx):
    rng = np.random.default_rng(2026)
    checks = {}
    examples = []

    def brute_counts(changes, tolerance):
        templates = np.lib.stride_tricks.sliding_window_view(changes, 3)
        left, right = np.triu_indices(len(templates), k=1)
        distance = np.abs(templates[left] - templates[right])
        limit = tolerance + MATCH_ATOL
        B = int(np.count_nonzero(distance[:, :2].max(axis=1) <= limit))
        A = int(np.count_nonzero(distance.max(axis=1) <= limit))
        return B, A

    fixtures = [
        np.zeros(18),
        np.tile([0.0, 0.0, 1.0], 10),
        rng.normal(size=18),
        rng.normal(size=30),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 2.0]),
    ]

    for number, changes in enumerate(fixtures):
        for tolerance in (0.0, 0.2 * np.std(changes)):
            assert sample_counts(changes, tolerance) == brute_counts(
                changes, tolerance
            )
    checks["pruned_counts_match_direct_enumeration"] = True

    assert sample_counts(np.arange(6.0), 0.0) == (0, 0)
    assert sample_counts(fixtures[-1], 0.0) == (1, 0)
    checks["no_matches_and_no_extensions"] = True

    patterns = {
        "flat": np.zeros(180),
        "persistent_trend": np.ones(180),
        "alternating": np.tile([-1.0, 1.0], 90),
        "random_three_state": rng.choice([-1.0, 0.0, 1.0], size=180),
        "random_binary": rng.choice([-1.0, 1.0], size=180),
    }

    for name, changes in patterns.items():
        symbol = symbols(changes)
        ids = symbol[:-1] * 3 + symbol[1:]
        C = np.bincount(ids, minlength=9).reshape(1, 3, 3)
        conditional, target = conditional_statistics(C)
        B, A = sample_counts(changes, 0.2 * np.std(changes))

        score = conditional[0] / np.log2(3)
        sample_score = np.nan if B == 0 else 1 - A / B

        if name in ("flat", "persistent_trend", "alternating"):
            assert abs(score) <= ENTROPY_ATOL
            assert abs(sample_score) <= ENTROPY_ATOL

        examples.append({
            "pattern": name,
            "changes": len(changes),
            "ce_score": float(score),
            "marginal_entropy_bits": float(target[0]),
            "sampen_B": B,
            "sampen_A": A,
            "sampen_score": sample_score,
        })

    checks["flat_trend_alternation_behavior"] = True

    changes = rng.normal(size=180)
    assert sample_counts(changes, 0.2 * np.std(changes)) == (
        sample_counts(changes * 20, 0.2 * np.std(changes * 20))
    )
    checks["sample_entropy_amplitude_invariance"] = True

    independent = np.ones((1, 3, 3, 3), dtype=np.int64)
    baseline, transfer = transfer_statistics(independent)
    assert abs(transfer[0]) <= ENTROPY_ATOL

    driven = np.zeros((1, 3, 3, 3), dtype=np.int64)
    for previous in range(3):
        for spread in range(3):
            driven[0, previous, spread, (previous + spread) % 3] = 1

    baseline, transfer = transfer_statistics(driven)
    assert abs(transfer[0] / baseline[0] - 1) <= ENTROPY_ATOL
    checks["independent_and_informative_transfer_tables"] = True

    length = 500
    origin = int(pd.Timestamp("2020-01-01", tz="UTC").value)
    ts = origin + np.arange(length, dtype=np.int64) * STEP
    available = ts.copy()
    available[100] += 117 * NS + 123

    changes = rng.choice([-0.2, 0.0, 0.2], size=length)
    mid = 1.1 + np.cumsum(changes) * PIP
    spread = 0.0001 + 0.00003 * np.sin(np.arange(length) / 37)

    has = np.ones(length, dtype=bool)
    has[400:407] = False
    mid[~has] = np.nan
    spread[~has] = np.nan

    excluded = np.zeros(length, dtype=bool)
    excluded[50:75] = True
    interpolated = np.zeros(length, dtype=bool)
    interpolated[100:111] = True

    source = pa.table({
        "sample_ns": arrow(ts),
        "available_ns": arrow(available),
        "mid": arrow(mid),
        "spread": arrow(spread),
        "has_price": arrow(has),
        "is_interpolated": arrow(interpolated, has),
        "in_long_gap": arrow(~has),
        "project_excluded": arrow(excluded),
        "project_eligible": arrow(has & ~excluded),
    })

    whole, input_checks, _ = compute(source)
    validate_output(whole)
    assert all(value == 0 for value in input_checks.values())

    fresh = values(whole, "fresh_changes_capped")
    assert fresh[75] == 75
    assert fresh[407] == 0
    assert not values(whole, "ready_price_3m")[424]
    assert values(whole, "ready_price_3m")[425]
    assert not values(whole, "ready_te_3m")[453]
    assert values(whole, "ready_te_3m")[454]
    checks["restart_and_full_fresh_warmup"] = True
    checks["available_excluded_hours_do_not_reset"] = True

    assert values(whole, "ce_available_ns_3m")[105] == available[100]
    checks["delayed_availability_keeps_exact_nanoseconds"] = True

    cut = 440
    start = cut - MAX_PRIOR
    tail, _, _ = compute(source.slice(start))
    assert whole.slice(cut).equals(
        tail.slice(cut - start), check_metadata=False
    )
    checks["processing_boundary_equivalence"] = True

    evidence = {
        "fingerprint": ctx["fingerprint"],
        "checks": checks,
        "passed": all(checks.values()),
    }

    fx.dump_json(ctx["tables"] / "synthetic_checks.json", evidence)
    pd.DataFrame(examples).to_csv(
        ctx["tables"] / "synthetic_pattern_examples.csv", index=False
    )
    return evidence


def daily_statistics(table, date, timings):
    result = {
        "date": date,
        "year_month": date[:7],
        "sample_rows": table.num_rows,
        "first_ns": int(values(table, "sample_ns")[0]),
        "last_ns": int(values(table, "sample_ns")[-1]),
        "priced_rows": int(values(table, "has_price").sum()),
        "missing_price_rows": int((~values(table, "has_price")).sum()),
        "restart_rows": int(values(table, "restart").sum()),
    }

    for key in WINDOWS:
        ready = values(table, f"ready_price_{key}")
        te_ready = values(table, f"ready_te_{key}")
        eligible = values(table, "project_eligible")
        status = values(table, f"sampen_status_{key}")
        te = values(table, f"te_fraction_{key}")

        result.update({
            f"ce_ready_{key}_rows": int(ready.sum()),
            f"project_ce_ready_{key}_rows": int((ready & eligible).sum()),
            f"sampen_defined_{key}_rows": int(np.isfinite(
                values(table, f"sampen_score_{key}")
            ).sum()),
            f"sampen_no_matches_{key}_rows": int(
                np.count_nonzero(status == "no_matches")
            ),
            f"sampen_infinite_raw_{key}_rows": int(
                np.count_nonzero(status == "infinite_raw")
            ),
            f"te_ready_{key}_rows": int(te_ready.sum()),
            f"te_defined_{key}_rows": int(np.isfinite(te).sum()),
            f"project_te_defined_{key}_rows": int(
                (np.isfinite(te) & eligible).sum()
            ),
            f"calculation_seconds_{key}": float(timings[key]),
        })
    return result


def finish(ctx, completed):
    daily = pd.DataFrame([item["summary"] for item in completed])
    assert not daily.empty

    coverage = []
    for key, n in WINDOWS.items():
        coverage.append({
            "window": key,
            "price_changes_required": n,
            "transition_pairs_per_complete_window": n - 1,
            "ce_first_ready_seconds_after_fresh_open": n * 10,
            "te_first_ready_seconds_after_fresh_open": (n + 29) * 10,
            "ce_ready_rows": int(daily[f"ce_ready_{key}_rows"].sum()),
            "project_ce_ready_rows": int(
                daily[f"project_ce_ready_{key}_rows"].sum()
            ),
            "sampen_defined_rows": int(
                daily[f"sampen_defined_{key}_rows"].sum()
            ),
            "sampen_no_match_rows": int(
                daily[f"sampen_no_matches_{key}_rows"].sum()
            ),
            "sampen_infinite_raw_rows": int(
                daily[f"sampen_infinite_raw_{key}_rows"].sum()
            ),
            "te_ready_rows": int(daily[f"te_ready_{key}_rows"].sum()),
            "te_defined_rows": int(daily[f"te_defined_{key}_rows"].sum()),
            "project_te_defined_rows": int(
                daily[f"project_te_defined_{key}_rows"].sum()
            ),
            "calculation_seconds_including_context": float(
                daily[f"calculation_seconds_{key}"].sum()
            ),
        })

    coverage = pd.DataFrame(coverage)
    daily.to_csv(ctx["tables"] / "daily_summary.csv", index=False)
    coverage.to_csv(ctx["tables"] / "window_coverage.csv", index=False)

    check_rows = [
        {"date": item["summary"]["date"], **item["checks"]}
        for item in completed
    ]
    check_frame = pd.DataFrame(check_rows)
    check_frame.to_csv(ctx["tables"] / "daily_checks.csv", index=False)

    count_columns = [
        name for name in daily.columns if name.endswith("_rows")
    ]
    daily.groupby("year_month", as_index=False)[count_columns].sum().to_csv(
        ctx["tables"] / "monthly_summary.csv", index=False
    )

    expected_jobs = ctx["jobs"][:len(completed)]
    checks = {
        "processed_dates_are_contiguous": (
            daily["date"].tolist() == [job["date"] for job in expected_jobs]
        ),
        "row_totals_match_expected_grid": (
            int(daily["sample_rows"].sum())
            == sum(job["rows"] for job in expected_jobs)
        ),
        "all_daily_checks_zero": bool(
            (check_frame.drop(columns="date") == 0).all().all()
        ),
        "all_synthetic_checks_pass": bool(
            fx.load_json(ctx["tables"] / "synthetic_checks.json")["passed"]
        ),
    }
    assert all(checks.values()), checks

    full = len(completed) == len(ctx["jobs"])
    summary = {
        "notebook": "04_entropy.ipynb",
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "source_run_id": SOURCE_RUN,
        "source_fingerprint": SOURCE_FP,
        "mode": "full_history" if full else "prefix_review",
        "available_input_months": len(ctx["inputs"]),
        "available_input_rows": sum(item["rows"] for item in ctx["inputs"]),
        "processed_days": len(completed),
        "processed_through": daily["date"].iloc[-1],
        "full_history_complete": full,
        "sample_rows": int(daily["sample_rows"].sum()),
        "priced_rows": int(daily["priced_rows"].sum()),
        "missing_price_rows": int(daily["missing_price_rows"].sum()),
        "restart_rows": int(daily["restart_rows"].sum()),
        "windows_minutes": [3, 5, 15, 30, 60],
        "update_seconds": 10,
        "spread_measurement_seconds": 300,
        "sample_tolerance_fraction": 0.2,
        "synthetic_checks_passed": True,
        "nonzero_machine_checks": int(
            (check_frame.drop(columns="date") != 0).sum().sum()
        ),
        "processed_prefix_machine_checks_passed": all(checks.values()),
        "full_history_machine_checks_passed": full and all(checks.values()),
        "human_review": "pending",
        "source_stability_check": (
            "file sizes/times and checkpoint hashes; no bulk data content hashes"
        ),
    }

    fx.dump_json(ctx["tables"] / "execution_summary.json", summary)
    fx.dump_json(ctx["tables"] / "completion_checks.json", {
        "fingerprint": ctx["fingerprint"],
        "checks": checks,
        "full_history_complete": full,
    })
    fx.dump_json(ctx["run"] / "run_state.json", {
        "fingerprint": ctx["fingerprint"],
        "processed_through": summary["processed_through"],
        "status": "full_history_processed" if full else "prefix_processed",
        "machine_checks_passed": True,
        "human_review": "pending",
    })

    return {"summary": summary, "daily": daily, "coverage": coverage}


def process(ctx):
    evidence = fx.load_json(ctx["tables"] / "synthetic_checks.json")
    assert evidence["fingerprint"] == ctx["fingerprint"]
    assert evidence["passed"]
    check_snapshot(ctx)

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": ctx["threads"],
    }) as con:
        for job in ctx["target_jobs"]:
            date = job["date"]
            destination = ctx["run"] / f"{date}.parquet"
            checkpoint_path = ctx["run"] / "_checkpoints" / f"{date}.json"

            if checkpoint_path.exists():
                done = fx.load_json(checkpoint_path)
                assert done["fingerprint"] == ctx["fingerprint"]
                assert all(value == 0 for value in done["checks"].values())
                assert utilities.file_info(destination) == done["output"]
                print(date, "resumed")
                continue

            started = time.perf_counter()
            lower = max(
                job["lower_ns"] - MAX_PRIOR * STEP,
                ctx["inputs"][0]["first_ns"],
            )
            upper = job["upper_ns"]

            paths = [
                str(ctx["root"] / item["path"])
                for item in ctx["inputs"]
                if item["last_ns"] >= lower and item["first_ns"] < upper
            ]

            con.read_parquet(
                paths, hive_partitioning=False
            ).create_view("entropy_source", replace=True)

            source = con.sql(f"""
                SELECT {", ".join(SOURCE_COLUMNS)}
                FROM entropy_source
                WHERE sample_ns >= {lower} AND sample_ns < {upper}
                ORDER BY sample_ns
            """).to_arrow_table()

            source_ts = values(source, "sample_ns")
            assert int(source_ts[0]) == lower
            assert int(source_ts[-1]) == upper - STEP

            result, checks, timings = compute(source)
            offset = (job["lower_ns"] - lower) // STEP
            result = result.slice(offset)
            original = source.slice(offset)

            assert result.num_rows == job["rows"]
            for name in SOURCE_COLUMNS:
                output_name = (
                    "input_available_ns" if name == "available_ns" else name
                )
                assert result[output_name].equals(original[name]), name

            checks.update(validate_output(result))

            temporary = destination.with_suffix(".parquet.tmp")
            pq.write_table(result, temporary, compression="zstd")
            saved = pq.read_table(temporary)

            checks["serialized_output_difference"] = (
                0 if result.equals(saved, check_metadata=False) else 1
            )
            assert all(value == 0 for value in checks.values()), checks
            temporary.replace(destination)

            summary = daily_statistics(result, date, timings)
            summary["processing_seconds"] = time.perf_counter() - started

            fx.dump_json(checkpoint_path, {
                "fingerprint": ctx["fingerprint"],
                "summary": summary,
                "checks": checks,
                "output": utilities.file_info(destination),
            })

            print(
                date,
                f"rows={result.num_rows:,}",
                f"seconds={summary['processing_seconds']:.1f}",
            )

    check_snapshot(ctx)

    completed = []
    missing_seen = False

    for job in ctx["jobs"]:
        checkpoint_path = ctx["run"] / "_checkpoints" / f"{job['date']}.json"
        if not checkpoint_path.exists():
            missing_seen = True
            continue

        assert not missing_seen, "Non-contiguous checkpoints require investigation."
        done = fx.load_json(checkpoint_path)
        assert done["fingerprint"] == ctx["fingerprint"]
        assert all(value == 0 for value in done["checks"].values())
        assert utilities.file_info(
            ctx["run"] / f"{job['date']}.parquet"
        ) == done["output"]
        completed.append(done)

    contract = {
        "fingerprint": ctx["fingerprint"],
        "source_fingerprint": SOURCE_FP,
        "timestamp_units": "UTC Unix nanoseconds",
        "display_timezone": "Europe/Athens",
        "mid_and_spread_units": "USD per EUR",
        "change_and_sample_tolerance_units": "pips",
        "price_symbols": {"0": "down", "1": "unchanged", "2": "up"},
        "spread_symbols": {"0": "narrowing", "1": "unchanged", "2": "widening"},
        "raw_conditional_and_transfer_units": "bits",
        "raw_sample_entropy_units": "natural logarithm units",
        "raw_sample_infinity": "valid when B>0 and A=0",
        "score_units": "dimensionless, 0 through 1",
        "availability": (
            "Use family_available_ns_window for that feature. "
            "Common available_ns is conservative for the complete row."
        ),
        "warmup": "ready_price for CE/SampEn; ready_te for transfer entropy",
        "undefined": (
            "SampEn B=0 and TE zero baseline remain unavailable; "
            "no forward filling of feature values."
        ),
        "label_meaning": (
            "CE uncertainty, sample irregularity, TE additional information"
        ),
        "support_warning": (
            "Full lookback does not guarantee statistical reliability. "
            "Short-window estimates require support and later out-of-sample review."
        ),
        "schema": str(pq.read_schema(
            ctx["run"] / f"{completed[0]['summary']['date']}.parquet"
        )),
    }
    fx.dump_json(ctx["run"] / "output_contract.json", contract)

    return finish(ctx, completed)
