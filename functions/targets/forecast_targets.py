import gc
import hashlib
import json
import time
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
MAX_GAP = 300 * NS
HORIZONS = {"30s": 3, "60s": 6}
MAX_STEPS = 6
TZ = "Europe/Athens"

SOURCE_RUN = "sampling_02_v1"
SOURCE_FP = (
    "ad5235dc14f854d9f26ef04dd2c1fdf846faf05cff43eb9f84afccc229628e54"
)

SAMPLE_COLUMNS = [
    "sample_ns", "available_ns", "has_price", "in_long_gap",
    "project_excluded", "project_eligible",
    "mid", "is_interpolated",
    "source_ts_ns", "source_available_ns", "source_age_ns",
    "left_anchor_ns", "right_anchor_ns",
]

ROLES = [
    "unused_reference", "train", "validation",
    "development_pool", "final_evaluation",
]


def timestamp(value):
    return int(pd.Timestamp(value, tz=TZ).value)


TRAIN_START = timestamp("2018-01-01")
TRAIN_END = timestamp("2019-01-01")
VALIDATION_END = timestamp("2019-07-01")
FINAL_START = timestamp("2026-01-01")
FINAL_END = timestamp("2026-10-17")


def retry_io(operation):
    delays = [0.25, 0.5, 1, 2, 4]
    for attempt in range(len(delays) + 1):
        try:
            return operation()
        except PermissionError as exc:
            if (
                getattr(exc, "winerror", None) not in (32, 33)
                or attempt == len(delays)
            ):
                raise
        gc.collect()
        time.sleep(delays[attempt])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    def write():
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    retry_io(write)
    retry_io(lambda: temporary.replace(path))


def save_csv(data, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    retry_io(lambda: data.to_csv(temporary, index=False))
    retry_io(lambda: temporary.replace(path))


def save_parquet(table, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")

    retry_io(lambda: pq.write_table(
        table, temporary, compression="zstd"
    ))

    # Close the disk file before Arrow performs verification.
    payload = retry_io(temporary.read_bytes)
    with pa.BufferReader(payload) as buffer:
        restored = pq.read_table(
            buffer,
            memory_map=False,
            pre_buffer=False,
            use_threads=False,
        )

    assert table.equals(restored, check_metadata=False), (
        "Serialized output differs from the calculated table."
    )
    del restored, payload

    retry_io(lambda: temporary.replace(path))


def values(table, name, fill=None):
    column = table[name]
    if fill is not None:
        column = column.fill_null(fill)
    return column.to_numpy(zero_copy_only=False)


def array(data, valid=None):
    return pa.array(
        data,
        mask=None if valid is None else ~valid,
        from_pandas=True,
    )


def dependencies():
    return {
        "target_helper": utilities.sha(__file__),
        "utility_helper": utilities.sha(utilities.__file__),
        "etl_helper": utilities.sha(fx.__file__),
    }


def expected_mid(reference_mid, predicted_change_pips):
    return (
        np.asarray(reference_mid, dtype=float)
        + PIP * np.asarray(predicted_change_pips, dtype=float)
    )


def exclusion_overlap(starts, ends, windows):
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    # Closed forecast interval versus half-open exclusion intervals.
    if len(windows) == 0:
        return np.zeros(len(starts), dtype=bool)

    index = np.searchsorted(windows[:, 1], starts, side="right")
    safe = np.minimum(index, len(windows) - 1)

    return (
        (index < len(windows))
        & (windows[safe, 0] <= ends)
    )


def split_roles(times):
    conditions = [
        times < TRAIN_START,
        times < TRAIN_END,
        times < VALIDATION_END,
        times < FINAL_START,
    ]

    roles = np.select(
        conditions,
        ROLES[:4],
        default="final_evaluation",
    )
    cutoffs = np.select(
        conditions,
        [TRAIN_START, TRAIN_END, VALIDATION_END, FINAL_START],
        default=FINAL_END,
    ).astype(np.int64)

    return roles, cutoffs


def prepare(root, run_id, stop_after_date, memory="2GB", threads=2):
    root = Path(root).resolve()
    assert run_id and Path(run_id).name == run_id

    source = root / "data" / "processed" / SOURCE_RUN
    source_reports = root / "reports" / "tables" / "sampling" / SOURCE_RUN

    source_config = fx.load_json(source / "configuration.json")
    source_state = fx.load_json(source / "run_state.json")
    source_summary = fx.load_json(source_reports / "execution_summary.json")
    completion = fx.load_json(
        source_reports / "full_history_completion_checks.json"
    )

    assert source_state["fingerprint"] == SOURCE_FP
    assert source_state["status"] == "full_history_processed"
    assert source_summary["full_history_complete"]
    assert source_summary["nonzero_machine_checks"] == 0
    assert completion["fingerprint"] == SOURCE_FP
    assert all(completion["checks"].values())

    calendar_path = (
        root / "data" / "processed" / "etl_01_v3"
        / "merged_exclusion_windows.json"
    )
    assert utilities.sha(calendar_path) == source_config["calendar_sha256"]

    windows = np.asarray(fx.load_json(calendar_path), dtype=np.int64)
    assert windows.ndim == 2 and windows.shape[1] == 2
    assert np.all(windows[:, 1] > windows[:, 0])
    assert np.all(windows[1:, 0] > windows[:-1, 1])

    raw_inputs = []

    for recorded in source_config["inputs"]:
        item = dict(recorded)
        month = item["year_month"]

        checkpoint = (
            root / "data" / "processed" / "etl_01_v3"
            / "_checkpoints" / f"observed_{month}.json"
        )
        actual = utilities.file_info(root / item["path"])

        assert all(actual[key] == item[key] for key in actual)
        assert utilities.sha(checkpoint) == item["checkpoint_sha256"]

        schema = pq.read_schema(root / item["path"])
        assert {"ts_ns", "bid", "ask"}.issubset(schema.names)
        assert pa.types.is_int64(schema.field("ts_ns").type)

        item["checkpoint_path"] = checkpoint.relative_to(root).as_posix()
        raw_inputs.append(item)

    first_grid = (
        (raw_inputs[0]["first_ns"] + STEP - 1) // STEP
    ) * STEP
    last_grid = (raw_inputs[-1]["last_ns"] // STEP) * STEP

    assert first_grid < TRAIN_START
    assert FINAL_START <= last_grid < FINAL_END

    inputs = []

    for checkpoint in sorted((source / "_checkpoints").glob("*.json")):
        month = checkpoint.stem
        saved = fx.load_json(checkpoint)

        assert saved["fingerprint"] == SOURCE_FP
        assert all(value == 0 for value in saved["checks"].values())

        path = source / "sampled_10s" / f"{month}.parquet"
        actual = utilities.file_info(path)
        expected = saved["outputs"]["sampled_10s"]
        assert all(actual[key] == expected[key] for key in actual)

        schema = pq.read_schema(path)
        assert set(SAMPLE_COLUMNS).issubset(schema.names)
        for name in ("sample_ns", "available_ns"):
            assert pa.types.is_int64(schema.field(name).type)

        lower, upper = fx.month_bounds(month)
        first = max(lower, first_grid)
        last = min(upper - STEP, last_grid)

        assert actual["rows"] == (last - first) // STEP + 1

        inputs.append({
            "year_month": month,
            "path": path.relative_to(root).as_posix(),
            **actual,
            "first_ns": first,
            "last_ns": last,
            "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": utilities.sha(checkpoint),
        })

    months = [item["year_month"] for item in inputs]
    assert months == [
        str(period)
        for period in pd.period_range(months[0], months[-1], freq="M")
    ]
    assert months == [item["year_month"] for item in raw_inputs]
    assert len(inputs) == source_summary["processed_months"]
    assert sum(item["rows"] for item in inputs) == source_summary["sample_rows"]

    first_day = pd.Timestamp(first_grid, unit="ns", tz="UTC").tz_convert(TZ)
    last_day = pd.Timestamp(last_grid, unit="ns", tz="UTC").tz_convert(TZ)

    jobs = []
    for day in pd.date_range(
        first_day.normalize(), last_day.normalize(), freq="D"
    ):
        lower = max(int(day.value), first_grid)
        upper = min(
            int((day + pd.DateOffset(days=1)).value),
            last_grid + STEP,
        )
        jobs.append({
            "date": day.strftime("%Y-%m-%d"),
            "lower_ns": lower,
            "upper_ns": upper,
            "rows": (upper - lower) // STEP,
        })

    if stop_after_date is not None:
        assert stop_after_date in [job["date"] for job in jobs]

    run = root / "data" / "processed" / run_id
    tables = root / "reports" / "tables" / "targets" / run_id
    figures = root / "reports" / "figures" / "targets" / run_id

    configuration = {
        "definition": "forecast_targets_v1",
        "run_id": run_id,
        "source_run": SOURCE_RUN,
        "source_fingerprint": SOURCE_FP,
        "source_configuration_sha256": utilities.sha(
            source / "configuration.json"
        ),
        "source_contract_sha256": utilities.sha(
            source / "output_contract.json"
        ),
        "calendar_sha256": utilities.sha(calendar_path),
        "dependencies": dependencies(),
        "versions": utilities.versions(),
        "raw_inputs": raw_inputs,
        "sample_inputs": inputs,
        "step_seconds": 10,
        "horizons_seconds": {"30s": 30, "60s": 60},
        "reference": "latest real quoted mid known at issue time",
        "additional_reference_age_filter": False,
        "future_endpoint": "eligible reconstructed ten-second mid",
        "interval": "closed [issue,target_end]",
        "anchor_permissions": "inherit accepted upstream permissions",
        "label_availability": "integer maximum of all interval dependencies",
        "purge": "label_available_ns >= assigned period upper boundary",
        "boundaries_athens": {
            "train_start": "2018-01-01",
            "train_end_exclusive": "2019-01-01",
            "validation_end_exclusive": "2019-07-01",
            "final_start": "2026-01-01",
            "final_end_exclusive": "2026-10-17",
        },
        "pip_size": PIP,
        "io_policy": "buffer verification and bounded Windows lock retries",
    }

    fingerprint = hashlib.sha256(json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    for directory in (run, run / "_checkpoints", tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    config_path = run / "configuration.json"
    if config_path.exists():
        old = fx.load_json(config_path)
        changed = [
            key for key in sorted(set(old) | set(configuration))
            if old.get(key) != configuration.get(key)
        ]
        assert not changed, (
            f"Configuration changed in {changed}. Use a new RUN_ID."
        )
    else:
        save_json(config_path, configuration)

    save_csv(pd.DataFrame(inputs), tables / "sample_input_inventory.csv")
    save_csv(pd.DataFrame(raw_inputs), tables / "quote_input_inventory.csv")

    split_manifest = {
        "fingerprint": fingerprint,
        "timezone": TZ,
        "interval_convention": "lower inclusive, upper exclusive",
        "periods": [
            {"split": "unused_reference", "upper_ns": TRAIN_START},
            {"split": "train", "lower_ns": TRAIN_START, "upper_ns": TRAIN_END},
            {
                "split": "validation",
                "lower_ns": TRAIN_END,
                "upper_ns": VALIDATION_END,
            },
            {
                "split": "development_pool",
                "lower_ns": VALIDATION_END,
                "upper_ns": FINAL_START,
            },
            {
                "split": "final_evaluation",
                "lower_ns": FINAL_START,
                "upper_ns": FINAL_END,
            },
        ],
        "eligibility": (
            "Filter by the intended split AND usable_<horizon>. "
            "Development-pool rows do not enter the initial fit automatically."
        ),
        "maturity": "All label information must arrive strictly before period end.",
        "final_evaluation": "No model or trading-rule selection on this period.",
    }
    save_json(run / "split_manifest.json", split_manifest)

    forecast_contract = {
        "targets_fingerprint": fingerprint,
        "produced_in": "Notebook 06",
        "price_units": "USD per EUR",
        "change_units": "pips",
        "pip_size": PIP,
        "columns": [
            "issue_ns", "reference_mid",
            "predicted_change_30s_pips", "expected_eurusd_rate_30s",
            "predicted_change_60s_pips", "expected_eurusd_rate_60s",
            "forecast_available_ns_30s", "forecast_available_ns_60s",
            "model_identifier_30s", "model_identifier_60s",
            "training_cutoff_ns_30s", "training_cutoff_ns_60s",
        ],
        "conversion": (
            "expected_eurusd_rate_h = "
            "reference_mid + predicted_change_h_pips * 0.0001"
        ),
        "initial_benchmark_definition": (
            "Zero change: predicted change = 0; expected rate = reference_mid."
        ),
        "predictions_generated_by_notebook_05": False,
    }
    save_json(run / "forecast_output_contract.json", forecast_contract)

    return {
        "root": root, "run": run, "tables": tables, "figures": figures,
        "run_id": run_id, "fingerprint": fingerprint,
        "configuration": configuration,
        "inputs": inputs, "raw_inputs": raw_inputs,
        "windows": windows,
        "first_grid": first_grid, "last_grid": last_grid,
        "jobs": jobs,
        "target_jobs": [
            job for job in jobs
            if stop_after_date is None or job["date"] <= stop_after_date
        ],
        "memory": memory, "threads": threads,
    }


def check_snapshot(ctx):
    config = ctx["configuration"]
    source = ctx["root"] / "data" / "processed" / SOURCE_RUN

    assert dependencies() == config["dependencies"]
    assert utilities.versions() == config["versions"]
    assert utilities.sha(source / "configuration.json") == (
        config["source_configuration_sha256"]
    )
    assert utilities.sha(source / "output_contract.json") == (
        config["source_contract_sha256"]
    )

    calendar = (
        ctx["root"] / "data" / "processed" / "etl_01_v3"
        / "merged_exclusion_windows.json"
    )
    assert utilities.sha(calendar) == config["calendar_sha256"]

    for item in ctx["inputs"] + ctx["raw_inputs"]:
        actual = utilities.file_info(ctx["root"] / item["path"])
        assert all(actual[key] == item[key] for key in actual), item["path"]
        assert utilities.sha(
            ctx["root"] / item["checkpoint_path"]
        ) == item["checkpoint_sha256"]


def reference_lookup(con, issue_grid, quote_query):
    con.register("_target_issue_grid", issue_grid)

    return con.sql(f"""
        SELECT
            g.sample_ns AS issue_ns,
            q.ts_ns AS reference_quote_ns,
            q.bid AS reference_bid,
            q.ask AS reference_ask,
            (q.bid + q.ask) / 2 AS reference_mid
        FROM _target_issue_grid g
        ASOF LEFT JOIN ({quote_query}) q
            ON g.sample_ns >= q.ts_ns
        ORDER BY g.sample_ns
    """).to_arrow_table()


def load_day(ctx, job, con):
    lower, upper = job["lower_ns"], job["upper_ns"]

    sample_paths = [
        str(ctx["root"] / item["path"])
        for item in ctx["inputs"]
        if item["last_ns"] >= lower
        and item["first_ns"] < upper + MAX_STEPS * STEP
    ]

    con.read_parquet(
        sample_paths, hive_partitioning=False
    ).create_view("_target_samples", replace=True)

    source = con.sql(f"""
        SELECT {", ".join(SAMPLE_COLUMNS)}
        FROM _target_samples
        WHERE sample_ns >= {lower}
          AND sample_ns < {upper + MAX_STEPS * STEP}
        ORDER BY sample_ns
    """).to_arrow_table()

    ts = values(source, "sample_ns")
    expected_last = min(
        upper - STEP + MAX_STEPS * STEP,
        ctx["last_grid"],
    )
    assert int(ts[0]) == lower
    assert int(ts[-1]) == expected_last
    assert len(ts) == (expected_last - lower) // STEP + 1

    month = job["date"][:7]
    index = next(
        index for index, item in enumerate(ctx["raw_inputs"])
        if item["year_month"] == month
    )
    raw = ctx["raw_inputs"][index]

    con.read_parquet(
        str(ctx["root"] / raw["path"]), hive_partitioning=False
    ).create_view("_target_real_month", replace=True)

    prior = con.sql(f"""
        SELECT ts_ns, bid, ask
        FROM _target_real_month
        WHERE ts_ns < {lower}
        ORDER BY ts_ns DESC
        LIMIT 1
    """).to_arrow_table()
    con.register("_target_prior_quote", prior)

    previous = None
    if index:
        previous = fx.load_json(
            ctx["root"] / ctx["raw_inputs"][index - 1]["checkpoint_path"]
        )["last"]

    seed = pa.table({
        "ts_ns": pa.array(
            [] if previous is None else [previous["ts_ns"]],
            type=pa.int64(),
        ),
        "bid": pa.array(
            [] if previous is None else [previous["bid"]],
            type=pa.float64(),
        ),
        "ask": pa.array(
            [] if previous is None else [previous["ask"]],
            type=pa.float64(),
        ),
    })
    con.register("_target_seed_quote", seed)

    quote_query = f"""
        SELECT ts_ns, bid, ask
        FROM _target_real_month
        WHERE ts_ns >= {lower} AND ts_ns < {upper}
        UNION ALL
        SELECT ts_ns, bid, ask FROM _target_prior_quote
        UNION ALL
        SELECT ts_ns, bid, ask FROM _target_seed_quote
    """

    references = reference_lookup(
        con,
        source.select(["sample_ns"]).slice(0, job["rows"]),
        quote_query,
    )
    return source, references


def build_labels(source, references, windows):
    rows = references.num_rows
    required = [
        "sample_ns", "available_ns", "has_price", "in_long_gap",
        "project_excluded", "project_eligible",
    ]
    assert all(source[name].null_count == 0 for name in required)

    ts = values(source, "sample_ns")
    av = values(source, "available_ns")
    has = values(source, "has_price").astype(bool)
    mid = values(source, "mid")
    issue = ts[:rows]

    found = references["reference_quote_ns"].is_valid().to_numpy()
    reference_ns = values(references, "reference_quote_ns", 0)
    reference_mid = values(references, "reference_mid")
    reference_bid = values(references, "reference_bid")
    reference_ask = values(references, "reference_ask")

    assert references.num_rows == rows
    assert np.array_equal(values(references, "issue_ns"), issue)

    source_age = values(source, "source_age_ns", 0)
    source_ts = values(source, "source_ts_ns", 0)
    source_av = values(source, "source_available_ns", 0)

    checks = {
        "duplicate_source_times": len(ts) - len(np.unique(ts)),
        "broken_source_grid": int(np.count_nonzero(np.diff(ts) != STEP)),
        "invalid_source_availability": int(np.count_nonzero(
            (av < ts) | (ts % STEP != 0)
        )),
        "invalid_present_mid": int(np.count_nonzero(
            has & (~np.isfinite(mid) | (mid <= 0))
        )),
        "prices_on_missing_rows": int(np.count_nonzero(
            ~has & source["mid"].is_valid().to_numpy()
        )),
        "prices_inside_long_gaps": int(np.count_nonzero(
            has & values(source, "in_long_gap")
        )),
        "invalid_source_age": int(np.count_nonzero(
            has & (
                (source_age < 0) | (source_age >= STEP)
                | (source_ts + source_age != ts)
                | (av < source_av)
            )
        )),
        "future_reference_quote": int(np.count_nonzero(
            found & (reference_ns > issue)
        )),
        "invalid_reference_values": int(np.count_nonzero(
            found & (
                ~np.isfinite(reference_mid)
                | ~np.isfinite(reference_bid)
                | ~np.isfinite(reference_ask)
                | (reference_bid <= 0)
                | (reference_ask < reference_bid)
            )
        )),
        "start_calendar_disagreement": int(np.count_nonzero(
            values(source, "project_excluded")[:rows]
            != exclusion_overlap(issue, issue, windows)
        )),
    }
    assert all(value == 0 for value in checks.values()), checks

    roles, cutoffs = split_roles(issue)
    reference_age = issue - reference_ns

    output = {
        "issue_ns": array(issue),
        "split": array(roles),
        "split_information_cutoff_ns": array(cutoffs),
        "reference_quote_ns": references["reference_quote_ns"],
        "reference_bid": references["reference_bid"],
        "reference_ask": references["reference_ask"],
        "reference_mid": references["reference_mid"],
        "reference_age_ns": array(reference_age, found),
        "reference_age_seconds": array(reference_age / NS, found),
        "label_start_has_price": source["has_price"].slice(0, rows),
        "label_start_project_eligible": source["project_eligible"].slice(0, rows),
        "label_start_is_interpolated": source["is_interpolated"].slice(0, rows),
        "label_start_available_ns": source["available_ns"].slice(0, rows),
    }

    pad = max(0, rows + MAX_STEPS - len(ts))
    padded_has = np.r_[has, np.zeros(pad, dtype=bool)]
    padded_av = np.r_[av, np.zeros(pad, dtype=np.int64)]

    for key, steps in HORIZONS.items():
        endpoint_index = np.arange(rows) + steps
        covered = endpoint_index < len(ts)
        safe = np.minimum(endpoint_index, len(ts) - 1)
        target_end = issue + steps * STEP

        complete_prices = np.lib.stride_tricks.sliding_window_view(
            padded_has, steps + 1
        )[:rows].all(axis=1)

        interval_available = np.lib.stride_tricks.sliding_window_view(
            padded_av, steps + 1
        )[:rows].max(axis=1)

        information_end = np.maximum(interval_available, reference_ns)
        excluded = exclusion_overlap(issue, target_end, windows)

        valid = covered & found & complete_prices & ~excluded
        purged = valid & (information_end >= cutoffs)
        usable = valid & ~purged

        future_mid = mid[safe]
        target_mid = np.full(rows, np.nan)
        change = np.full(rows, np.nan)

        target_mid[valid] = future_mid[valid]
        change[valid] = (
            future_mid[valid] - reference_mid[valid]
        ) / PIP

        reason = np.full(rows, "valid", dtype=object)
        reason[~found] = "no_known_reference"
        reason[covered & ~complete_prices] = "disruption"
        reason[excluded] = "excluded_interval"
        reason[~covered] = "future_not_covered"

        endpoint_present = covered & has[safe]

        output.update({
            f"label_end_ns_{key}": array(target_end),
            f"future_covered_{key}": array(covered),
            f"interval_has_missing_prices_{key}": array(
                ~complete_prices, covered
            ),
            f"interval_overlaps_exclusion_{key}": array(excluded),
            f"label_valid_{key}": array(valid),
            f"label_reason_{key}": array(reason),
            f"label_available_ns_{key}": array(information_end, covered),
            f"purged_at_split_{key}": array(purged),
            f"usable_{key}": array(usable),
            f"target_change_{key}_pips": array(change),
            f"target_future_mid_{key}": array(target_mid),
            f"endpoint_has_price_{key}": array(has[safe], covered),
            f"endpoint_available_ns_{key}": array(av[safe], covered),
            f"endpoint_is_interpolated_{key}": array(
                values(source, "is_interpolated", False)[safe],
                endpoint_present,
            ),
        })

        for output_name, source_name in (
            ("endpoint_source_ns", "source_ts_ns"),
            ("endpoint_source_available_ns", "source_available_ns"),
            ("endpoint_source_age_ns", "source_age_ns"),
            ("endpoint_left_anchor_ns", "left_anchor_ns"),
            ("endpoint_right_anchor_ns", "right_anchor_ns"),
        ):
            output[f"{output_name}_{key}"] = array(
                values(source, source_name, 0)[safe],
                endpoint_present,
            )

    return pa.table(output), checks


def validate_labels(table, global_last_grid):
    issue = values(table, "issue_ns")
    reference_present = table["reference_quote_ns"].is_valid().to_numpy()
    reference_ns = values(table, "reference_quote_ns", 0)
    reference = values(table, "reference_mid")
    cutoff = values(table, "split_information_cutoff_ns")

    checks = {}

    def count(name, condition):
        checks[name] = int(np.count_nonzero(condition))

    count("duplicate_issue_times", np.r_[False, np.diff(issue) == 0])
    count("broken_candidate_grid", np.diff(issue) != STEP)
    count(
        "reference_after_issue",
        reference_present & (reference_ns > issue),
    )
    count(
        "reference_age_difference",
        reference_present & (
            values(table, "reference_age_ns", 0)
            != issue - reference_ns
        ),
    )

    expected_split, expected_cutoff = role_and_cutoff(issue)
    count("incorrect_split", values(table, "split") != expected_split)
    count("incorrect_split_cutoff", cutoff != expected_cutoff)

    for key, steps in HORIZONS.items():
        end = values(table, f"label_end_ns_{key}")
        covered = values(table, f"future_covered_{key}")
        overlap = values(table, f"interval_overlaps_exclusion_{key}")
        missing = values(table, f"interval_has_missing_prices_{key}", False)
        valid = values(table, f"label_valid_{key}")
        usable = values(table, f"usable_{key}")
        purged = values(table, f"purged_at_split_{key}")
        information = values(table, f"label_available_ns_{key}", 0)
        future = values(table, f"target_future_mid_{key}")
        change = values(table, f"target_change_{key}_pips")

        count(f"{key}_horizon", end != issue + steps * STEP)
        count(f"{key}_coverage", covered != (end <= global_last_grid))
        count(
            f"{key}_validity",
            valid != (
                covered & reference_present & ~missing & ~overlap
            ),
        )
        count(
            f"{key}_target_presence",
            (np.isfinite(future) != valid)
            | (np.isfinite(change) != valid),
        )
        count(
            f"{key}_pip_conversion",
            valid & (
                np.abs(change - (future - reference) / PIP)
                > VALUE_ATOL
            ),
        )
        count(
            f"{key}_price_reconstruction",
            valid & (
                np.abs(expected_mid(reference, change) - future)
                > VALUE_ATOL * PIP
            ),
        )
        count(
            f"{key}_label_availability_presence",
            table[f"label_available_ns_{key}"].is_valid().to_numpy()
            != covered,
        )
        count(
            f"{key}_early_label_availability",
            covered & (
                (information < end)
                | (reference_present & (information < reference_ns))
            ),
        )
        count(
            f"{key}_purging",
            purged != (valid & (information >= cutoff)),
        )
        count(f"{key}_usability", usable != (valid & ~purged))
        count(
            f"{key}_invalid_reason",
            (values(table, f"label_reason_{key}") == "valid") != valid,
        )
        count(
            f"{key}_uncovered_reason",
            ~covered & (
                values(table, f"label_reason_{key}") != "future_not_covered"
            ),
        )

    valid30 = values(table, "label_valid_30s")
    valid60 = values(table, "label_valid_60s")
    usable30 = values(table, "usable_30s")
    usable60 = values(table, "usable_60s")

    count("long_valid_without_short_valid", valid60 & ~valid30)
    count("long_usable_without_short_usable", usable60 & ~usable30)

    covered60 = values(table, "future_covered_60s")
    count(
        "information_order_between_horizons",
        covered60 & (
            values(table, "label_available_ns_60s", 0)
            < values(table, "label_available_ns_30s", 0)
        ),
    )

    assert all(value == 0 for value in checks.values()), {
        key: value for key, value in checks.items() if value
    }
    return checks


def synthetic_checks(ctx):
    checks = {}
    origin = timestamp_ns("2018-06-15 10:00:00")

    def fixture(base, quote_offsets, quote_mids, last_step, windows=()):
        quote_times = base + np.asarray(
            quote_offsets, dtype=np.int64
        ) * NS
        quote_mids = np.asarray(quote_mids, dtype=float)
        grid = base + np.arange(last_step + 1, dtype=np.int64) * STEP

        has = np.zeros(len(grid), dtype=bool)
        mid = np.full(len(grid), np.nan)
        interpolated = np.zeros(len(grid), dtype=bool)
        left = np.zeros(len(grid), dtype=np.int64)
        right = np.zeros(len(grid), dtype=np.int64)
        available = grid.copy()
        gap = np.zeros(len(grid), dtype=bool)

        for i, point in enumerate(grid):
            a = np.searchsorted(quote_times, point, side="right") - 1
            assert a >= 0

            if quote_times[a] == point:
                has[i] = True
                mid[i] = quote_mids[a]
                left[i] = right[i] = point
                continue

            assert a + 1 < len(quote_times)
            A, B = quote_times[a], quote_times[a + 1]

            if B - A > 300 * NS:
                gap[i] = True
                available[i] = max(point, A + 300 * NS + 1)
            else:
                has[i] = True
                interpolated[i] = True
                left[i], right[i] = A, B
                mid[i] = (
                    quote_mids[a]
                    + (point - A) / (B - A)
                    * (quote_mids[a + 1] - quote_mids[a])
                )
                available[i] = B

        excluded = exclusion_overlap(grid, grid, windows)

        source = pa.table({
            "sample_ns": arrow(grid),
            "available_ns": arrow(available),
            "has_price": arrow(has),
            "in_long_gap": arrow(gap),
            "mid": arrow(mid),
            "is_interpolated": arrow(interpolated, has),
            "source_ts_ns": arrow(grid, has),
            "source_available_ns": arrow(
                np.where(interpolated, right, grid), has
            ),
            "source_age_ns": arrow(np.zeros(len(grid), dtype=np.int64), has),
            "left_anchor_ns": arrow(left, has),
            "right_anchor_ns": arrow(right, has),
            "project_excluded": arrow(excluded),
            "project_eligible": arrow(has & ~excluded),
        })

        quotes = pa.table({
            "ts_ns": arrow(quote_times),
            "bid": arrow(quote_mids - 0.00001),
            "ask": arrow(quote_mids + 0.00001),
        })
        return source, quotes

    with duckdb.connect() as con:
        source, quotes = fixture(
            origin,
            [-10, 30, 60, 100],
            [1.10000, 1.10020, 1.09990, 1.10010],
            10,
        )
        con.register("toy_quotes", quotes)
        refs = reference_lookup(
            con, source.select(["sample_ns"]),
            "SELECT * FROM toy_quotes",
        )

        table, input_checks = build_labels(source, refs, [])
        validate_labels(table, origin + 100 * NS)
        assert all(value == 0 for value in input_checks.values())

        assert abs(values(table, "target_change_30s_pips")[0] - 2) < VALUE_ATOL
        assert abs(values(table, "target_change_60s_pips")[0] + 1) < VALUE_ATOL
        checks["hand_checked_pip_targets"] = True

        assert values(table, "reference_quote_ns")[0] == origin - 10 * NS
        assert abs(values(table, "reference_mid")[0] - 1.1) < 1e-12
        assert values(source, "available_ns")[0] > origin
        assert abs(
            values(source, "mid")[0] - values(table, "reference_mid")[0]
        ) > 1e-6
        checks["known_reference_uses_asof_quote"] = True

        assert values(table, "reference_age_ns")[1] == 20 * NS
        assert values(table, "label_valid_30s")[1]
        checks["no_extra_ten_second_reference_age_filter"] = True

        assert (
            values(table, "label_reason_30s")[-1]
            == "future_not_covered"
        )
        checks["end_of_source_coverage"] = True

        tiny_window = [[origin + 15 * NS, origin + 16 * NS]]
        tiny, _ = build_labels(source, refs, tiny_window)
        assert not values(tiny, "label_valid_30s")[0]
        checks["sub_grid_exclusion_detected"] = True

        dead_source, dead_quotes = fixture(
            origin,
            [-10, 40, 400],
            [1.10000, 1.10020, 1.10030],
            40,
        )
        con.register("toy_quotes", dead_quotes)
        dead_refs = reference_lookup(
            con, dead_source.select(["sample_ns"]),
            "SELECT * FROM toy_quotes",
        )
        dead, _ = build_labels(dead_source, dead_refs, [])
        validate_labels(dead, origin + 400 * NS)

        assert values(dead, "label_valid_30s")[0]
        assert values(dead, "endpoint_is_interpolated_30s", False)[0]
        assert not values(dead, "label_valid_60s")[0]
        checks["synthetic_before_disruption_and_separate_horizons"] = True

        boundary = timestamp_ns("2018-12-31 23:59:20")
        closure = [[boundary + 35 * NS, boundary + 45 * NS]]
        b_source, b_quotes = fixture(
            boundary,
            [-10, 40, 120],
            [1.10000, 1.10020, 1.10030],
            12,
            closure,
        )

        availability = values(b_source, "available_ns").copy()
        availability[3] += 123
        b_source = b_source.set_column(
            b_source.schema.get_field_index("available_ns"),
            "available_ns",
            arrow(availability),
        )

        con.register("toy_quotes", b_quotes)
        b_refs = reference_lookup(
            con, b_source.select(["sample_ns"]),
            "SELECT * FROM toy_quotes",
        )
        b_table, _ = build_labels(b_source, b_refs, closure)
        validate_labels(b_table, boundary + 120 * NS)

        assert values(b_table, "label_valid_30s")[0]
        assert values(b_table, "endpoint_is_interpolated_30s", False)[0]
        assert values(b_table, "purged_at_split_30s")[0]
        assert not values(b_table, "usable_30s")[0]
        assert values(b_table, "label_available_ns_30s", 0)[0] == (
            boundary + 40 * NS + 123
        )
        checks["permitted_anchor_and_information_cutoff_purge"] = True
        checks["exact_nanosecond_availability"] = True

        long_origin = timestamp_ns("2018-12-31 23:00:00")
        offsets = np.arange(-10, 7440, 20, dtype=np.int64)
        quote_mids = (
            1.1 + 0.0001 * np.sin(np.arange(len(offsets)) / 11)
        )
        long_source, long_quotes = fixture(
            long_origin, offsets, quote_mids, 725
        )
        con.register("toy_quotes", long_quotes)

        full_refs = reference_lookup(
            con, long_source.select(["sample_ns"]).slice(0, 720),
            "SELECT * FROM toy_quotes",
        )
        whole, _ = build_labels(long_source, full_refs, [])
        validate_labels(whole, int(values(long_quotes, "ts_ns")[-1]))

        cut = 360
        cut_ns = long_origin + cut * STEP
        quote_times = values(long_quotes, "ts_ns")
        prior_index = np.flatnonzero(quote_times < cut_ns)[-1]
        keep = np.r_[
            prior_index,
            np.flatnonzero(quote_times >= cut_ns),
        ]
        tail_quotes = long_quotes.take(pa.array(keep, type=pa.int64()))
        con.register("toy_quotes", tail_quotes)

        tail_source = long_source.slice(cut)
        tail_refs = reference_lookup(
            con, tail_source.select(["sample_ns"]).slice(0, 360),
            "SELECT * FROM toy_quotes",
        )
        tail, _ = build_labels(tail_source, tail_refs, [])

        assert whole.slice(cut).equals(tail, check_metadata=False)
        checks["day_month_and_split_boundary_equivalence"] = True

    assert abs(expected_mid(1.10000, 2) - 1.10020) < 1e-12
    assert abs(expected_mid(1.10000, -1) - 1.09990) < 1e-12
    checks["prediction_price_conversion"] = True

    evidence = {
        "fingerprint": ctx["fingerprint"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    dump_json(ctx["tables"] / "synthetic_checks.json", evidence)
    return evidence


def daily_statistics(table, date):
    issue = values(table, "issue_ns")
    split = values(table, "split")
    reference_present = table["reference_quote_ns"].is_valid().to_numpy()

    summary = {
        "date": date,
        "year_month": date[:7],
        "candidate_rows": table.num_rows,
        "first_ns": int(issue[0]),
        "last_ns": int(issue[-1]),
        "known_reference_rows": int(reference_present.sum()),
    }

    coverage = []

    for role in ROLES:
        selected = split == role

        for key in HORIZONS:
            valid = values(table, f"label_valid_{key}")
            usable = values(table, f"usable_{key}")
            purged = values(table, f"purged_at_split_{key}")
            interpolated = values(
                table, f"endpoint_is_interpolated_{key}", False
            )

            coverage.append({
                "split": role,
                "horizon": key,
                "candidate_rows": int(selected.sum()),
                "data_valid_rows": int((selected & valid).sum()),
                "usable_rows": int((selected & usable).sum()),
                "purged_rows": int((selected & purged).sum()),
                "valid_synthetic_endpoint_rows": int(
                    (selected & valid & interpolated).sum()
                ),
            })

    for key in HORIZONS:
        summary[f"valid_{key}_rows"] = int(
            values(table, f"label_valid_{key}").sum()
        )
        summary[f"usable_{key}_rows"] = int(
            values(table, f"usable_{key}").sum()
        )
        summary[f"purged_{key}_rows"] = int(
            values(table, f"purged_at_split_{key}").sum()
        )

    return summary, coverage


def finish(ctx, completed):
    daily = pd.DataFrame([item["summary"] for item in completed])
    check_frame = pd.DataFrame([
        {"date": item["summary"]["date"], **item["checks"]}
        for item in completed
    ])

    coverage = pd.DataFrame([
        row for item in completed for row in item["coverage"]
    ]).groupby(
        ["split", "horizon"], as_index=False, sort=False
    ).sum(numeric_only=True)

    count_columns = [
        name for name in daily.columns if name.endswith("_rows")
    ]
    monthly = daily.groupby(
        "year_month", as_index=False
    )[count_columns].sum()

    save_csv(daily, ctx["tables"] / "daily_summary.csv")
    save_csv(check_frame, ctx["tables"] / "daily_checks.csv")
    save_csv(coverage, ctx["tables"] / "period_coverage.csv")
    save_csv(monthly, ctx["tables"] / "monthly_summary.csv")

    jobs = ctx["jobs"][:len(completed)]

    checks = {
        "processed_dates_contiguous": (
            daily["date"].tolist() == [job["date"] for job in jobs]
        ),
        "candidate_rows_match_grid": (
            int(daily["candidate_rows"].sum())
            == sum(job["rows"] for job in jobs)
        ),
        "all_daily_checks_zero": bool(
            (check_frame.drop(columns="date") == 0).all().all()
        ),
        "synthetic_checks_passed": bool(
            fx.load_json(ctx["tables"] / "synthetic_checks.json")["passed"]
        ),
    }
    assert all(checks.values()), checks

    full = len(completed) == len(ctx["jobs"])

    summary = {
        "notebook": "05_targets_and_validation_design.ipynb",
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "source_run_id": SOURCE_RUN,
        "source_fingerprint": SOURCE_FP,
        "mode": "full_history" if full else "prefix_review",
        "available_input_months": len(ctx["inputs"]),
        "available_candidate_rows": sum(
            item["rows"] for item in ctx["inputs"]
        ),
        "processed_days": len(completed),
        "processed_through": daily["date"].iloc[-1],
        "candidate_rows": int(daily["candidate_rows"].sum()),
        "known_reference_rows": int(daily["known_reference_rows"].sum()),
        "full_history_complete": full,
        "candidate_spacing_seconds": 10,
        "horizons_seconds": [30, 60],
        "training_period": "2018",
        "validation_period": "2019-01-01 through 2019-06-30",
        "final_evaluation_period": "2026-01-01 through 2026-10-16",
        "nonzero_machine_checks": int(
            (check_frame.drop(columns="date") != 0).sum().sum()
        ),
        "synthetic_checks_passed": True,
        "processed_prefix_machine_checks_passed": all(checks.values()),
        "full_history_machine_checks_passed": full and all(checks.values()),
        "predictions_generated": False,
        "recorded_processing_seconds": float(
            daily["processing_seconds"].sum()
        ),
        "human_review": "pending",
        "source_stability_check": (
            "file sizes/times and checkpoint hashes; "
            "no bulk data content hashing"
        ),
    }

    dump_json(ctx["tables"] / "execution_summary.json", summary)
    dump_json(ctx["tables"] / "completion_checks.json", {
        "fingerprint": ctx["fingerprint"],
        "checks": checks,
        "full_history_complete": full,
    })
    dump_json(ctx["run"] / "run_state.json", {
        "fingerprint": ctx["fingerprint"],
        "status": "full_history_processed" if full else "prefix_processed",
        "processed_through": summary["processed_through"],
        "machine_checks_passed": True,
        "human_review": "pending",
    })

    return {
        "summary": summary, "daily": daily,
        "coverage": coverage, "checks": checks,
    }


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
            source, references = load_day(ctx, job, con)
            table, checks = build_labels(
                source, references, ctx["windows"]
            )
            checks.update(validate_labels(table, ctx["last_grid"]))

            assert table.num_rows == job["rows"]
            assert int(values(table, "issue_ns")[0]) == job["lower_ns"]
            assert int(values(table, "issue_ns")[-1]) == (
                job["upper_ns"] - STEP
            )

            save_parquet(table, destination)
            checks["serialized_output_difference"] = 0

            summary, coverage = daily_statistics(table, date)
            summary["processing_seconds"] = time.perf_counter() - started

            dump_json(checkpoint_path, {
                "fingerprint": ctx["fingerprint"],
                "summary": summary,
                "coverage": coverage,
                "checks": checks,
                "output": utilities.file_info(destination),
            })

            print(
                date,
                f"candidates={table.num_rows:,}",
                f"valid30={summary['valid_30s_rows']:,}",
                f"valid60={summary['valid_60s_rows']:,}",
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

        assert not missing_seen, (
            "Non-contiguous checkpoints require investigation."
        )
        done = fx.load_json(checkpoint_path)
        assert done["fingerprint"] == ctx["fingerprint"]
        assert all(value == 0 for value in done["checks"].values())
        assert utilities.file_info(
            ctx["run"] / f"{job['date']}.parquet"
        ) == done["output"]
        completed.append(done)

    assert completed

    contract = {
        "fingerprint": ctx["fingerprint"],
        "source_fingerprint": SOURCE_FP,
        "timestamp_units": "UTC Unix nanoseconds",
        "display_timezone": TZ,
        "price_units": "USD per EUR",
        "target_change_units": "pips",
        "reference": "latest real quoted mid available at issue time",
        "future_prices": "continuity-eligible real and reconstructed series",
        "initial_training_filter": "split == 'train' AND usable_h",
        "validation_filter": "split == 'validation' AND usable_h",
        "final_filter": "split == 'final_evaluation' AND usable_h",
        "future_dependent_metadata": (
            "All target_*, endpoint_*, label_*, future_*, "
            "interval_*, purged_* and usable_* fields."
        ),
        "known_reference_columns": [
            "reference_quote_ns", "reference_bid", "reference_ask",
            "reference_mid", "reference_age_ns", "reference_age_seconds",
        ],
        "feature_join_requirement": (
            "Notebook 06 must select only indicator values available "
            "by issue_ns and explicitly whitelist predictor columns."
        ),
        "feature_missingness": (
            "Missing entropy or other features does not invalidate these labels. "
            "Feature handling is designed in Notebook 06."
        ),
        "predictions": "Defined by forecast_output_contract.json; not generated here",
        "schema": str(pq.read_schema(
            ctx["run"] / f"{completed[0]['summary']['date']}.parquet"
        )),
    }
    dump_json(ctx["run"] / "output_contract.json", contract)

    return finish(ctx, completed)

# Notebook 05 Cell 9 compatibility repair
arrow = array
timestamp_ns = timestamp
role_and_cutoff = split_roles
dump_json = save_json
VALUE_ATOL = 1e-8
