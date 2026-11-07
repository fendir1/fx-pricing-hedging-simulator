
import gc
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from functions.etl import clean_quotes as fx
from functions.volatility import multiscale_volatility as util

NS = 1_000_000_000
STEP = 10 * NS
PIP = 0.0001
TOL = 1e-8
TZ = "Europe/Athens"
HORIZONS = {"30s": 3, "60s": 6}
SAMPLE_RUN = "sampling_02_v1"
SOURCE_FP = "ad5235dc14f854d9f26ef04dd2c1fdf846faf05cff43eb9f84afccc229628e54"
SAMPLE_COLS = [
    "sample_ns", "available_ns", "has_price", "in_long_gap",
    "project_excluded", "project_eligible", "mid", "is_interpolated",
    "source_ts_ns", "source_available_ns", "source_age_ns",
    "left_anchor_ns", "right_anchor_ns",
]
ROLES = [
    "unused_reference", "train", "validation",
    "development_pool", "final_evaluation",
]

def ns(date):
    return int(pd.Timestamp(date, tz=TZ).value)

CUTS = np.array([
    ns("2018-01-01"), ns("2019-01-01"), ns("2019-07-01"),
    ns("2026-01-01"), ns("2026-10-17"),
], dtype=np.int64)

def get(table, name, fill=None):
    col = table[name]
    if fill is not None:
        col = col.fill_null(fill)
    return col.to_numpy(zero_copy_only=False)

def arr(data, valid=None):
    return pa.array(
        data,
        mask=None if valid is None else ~valid,
        from_pandas=True,
    )

def retry(operation):
    delays = (0.25, 0.5, 1, 2, 4)
    for attempt in range(len(delays) + 1):
        try:
            return operation()
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (32, 33):
                raise
            if attempt == len(delays):
                raise
        gc.collect()
        time.sleep(delays[attempt])

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    def save():
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    retry(save)
    retry(lambda: temporary.replace(path))

def write_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    retry(lambda: frame.to_csv(temporary, index=False))
    retry(lambda: temporary.replace(path))

def write_parquet(table, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    retry(lambda: pq.write_table(table, temporary, compression="zstd"))

    payload = retry(temporary.read_bytes)
    with pa.BufferReader(payload) as buffer:
        restored = pq.read_table(
            buffer,
            memory_map=False,
            pre_buffer=False,
            use_threads=False,
        )
    assert table.equals(restored, check_metadata=False), "Readback mismatch."
    del restored, payload

    retry(lambda: temporary.replace(path))

def code_hashes():
    return {
        "target_helper": util.sha(__file__),
        "shared_utilities": util.sha(util.__file__),
        "etl_utilities": util.sha(fx.__file__),
    }

def predicted_mid(reference, predicted_pips):
    return np.asarray(reference) + PIP * np.asarray(predicted_pips)

def overlap(starts, ends, windows):
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    if len(windows) == 0:
        return np.zeros(len(starts), dtype=bool)

    index = np.searchsorted(windows[:, 1], starts, side="right")
    safe = np.minimum(index, len(windows) - 1)
    return (index < len(windows)) & (windows[safe, 0] <= ends)

def roles_at(times):
    index = np.searchsorted(CUTS[:4], times, side="right")
    return np.asarray(ROLES)[index], CUTS[index]

def split_manifest(fingerprint):
    starts = [
        None, int(CUTS[0]), int(CUTS[1]),
        int(CUTS[2]), int(CUTS[3]),
    ]
    periods = []

    for role, lower, upper in zip(ROLES, starts, CUTS):
        periods.append({
            "split": role,
            "lower_ns": lower,
            "upper_ns": int(upper),
            "Athens_start_inclusive": (
                "source start" if lower is None else
                pd.Timestamp(
                    lower, unit="ns", tz="UTC"
                ).tz_convert(TZ).isoformat()
            ),
            "Athens_end_exclusive": (
                pd.Timestamp(
                    int(upper), unit="ns", tz="UTC"
                ).tz_convert(TZ).isoformat()
            ),
        })

    return {
        "fingerprint": fingerprint,
        "timezone": TZ,
        "periods": periods,
        "maturity_rule": "usable requires label_available_ns < split upper_ns",
        "fit_filter": "split == 'train' AND usable_<horizon>",
        "final_evaluation": "reserved from model and policy selection",
    }

def forecast_contract(fingerprint):
    columns = [
        "issue_ns", "reference_quote_ns",
        "reference_mid", "reference_age_ns",
    ]
    for horizon in HORIZONS:
        columns += [
            f"predicted_change_{horizon}_pips",
            f"expected_eurusd_rate_{horizon}",
            f"forecast_available_ns_{horizon}",
            f"model_identifier_{horizon}",
            f"training_cutoff_ns_{horizon}",
        ]

    return {
        "targets_fingerprint": fingerprint,
        "produced_in": "Notebook 06",
        "columns": columns,
        "price_units": "USD per EUR",
        "change_units": "pips",
        "conversion": "expected rate = reference_mid + predicted pips * 0.0001",
        "predictions_generated_here": False,
        "horizons_seconds": {"30s": 30, "60s": 60},
        "timestamp_units": "integer UTC nanoseconds",
        "benchmark": "zero change; expected rate equals reference_mid",
    }



def prepare(root, run_id, stop_date, memory="2GB", threads=2):
    root = Path(root).resolve()
    assert run_id and Path(run_id).name == run_id

    source = root / "data" / "processed" / SAMPLE_RUN
    reports = root / "reports" / "tables" / "sampling" / SAMPLE_RUN

    source_config = fx.load_json(source / "configuration.json")
    source_summary = fx.load_json(reports / "execution_summary.json")
    completion = fx.load_json(reports / "full_history_completion_checks.json")

    assert source_summary["fingerprint"] == SOURCE_FP
    assert source_summary["full_history_complete"]
    assert source_summary["nonzero_machine_checks"] == 0
    assert completion["fingerprint"] == SOURCE_FP
    assert all(completion["checks"].values())

    calendar = (
        root / "data" / "processed" / "etl_01_v3"
        / "merged_exclusion_windows.json"
    )
    assert util.sha(calendar) == source_config["calendar_sha256"]
    windows = np.asarray(
        fx.load_json(calendar), dtype=np.int64
    ).reshape(-1, 2)

    assert np.all(windows[:, 1] > windows[:, 0])
    assert np.all(windows[1:, 0] >= windows[:-1, 1])

    raw_inputs = []
    for recorded in source_config["inputs"]:
        item = dict(recorded)
        checkpoint = (
            root / "data" / "processed" / "etl_01_v3"
            / "_checkpoints" / f"observed_{item['year_month']}.json"
        )
        item["checkpoint_path"] = checkpoint.relative_to(root).as_posix()

        actual = util.file_info(root / item["path"])
        assert all(actual[k] == item[k] for k in actual), item["path"]
        assert util.sha(checkpoint) == item["checkpoint_sha256"]
        assert {"ts_ns", "bid", "ask"}.issubset(
            pq.read_schema(root / item["path"]).names
        )
        raw_inputs.append(item)

    first_grid = ((raw_inputs[0]["first_ns"] + STEP - 1) // STEP) * STEP
    last_grid = (raw_inputs[-1]["last_ns"] // STEP) * STEP
    assert first_grid < CUTS[0] and CUTS[3] <= last_grid < CUTS[4]

    inputs = []
    for checkpoint in sorted((source / "_checkpoints").glob("*.json")):
        saved = fx.load_json(checkpoint)
        assert saved["fingerprint"] == SOURCE_FP
        assert all(value == 0 for value in saved["checks"].values())

        month = checkpoint.stem
        path = source / "sampled_10s" / f"{month}.parquet"
        info = util.file_info(path)
        assert all(
            info[k] == saved["outputs"]["sampled_10s"][k]
            for k in info
        )

        schema = pq.read_schema(path)
        assert set(SAMPLE_COLS).issubset(schema.names)
        for name in ("sample_ns", "available_ns"):
            assert pa.types.is_int64(schema.field(name).type)

        lower, upper = fx.month_bounds(month)
        first = max(lower, first_grid)
        last = min(upper - STEP, last_grid)
        assert info["rows"] == (last - first) // STEP + 1

        inputs.append({
            "year_month": month,
            "path": path.relative_to(root).as_posix(),
            **info,
            "first_ns": first,
            "last_ns": last,
            "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": util.sha(checkpoint),
        })

    months = [item["year_month"] for item in inputs]
    assert months == [item["year_month"] for item in raw_inputs]
    assert months == [
        str(p) for p in pd.period_range(months[0], months[-1], freq="M")
    ]
    assert len(inputs) == source_summary["processed_months"]
    assert sum(item["rows"] for item in inputs) == source_summary["sample_rows"]

    first_day = pd.Timestamp(
        first_grid, unit="ns", tz="UTC"
    ).tz_convert(TZ).normalize()
    last_day = pd.Timestamp(
        last_grid, unit="ns", tz="UTC"
    ).tz_convert(TZ).normalize()

    jobs = []
    for day in pd.date_range(first_day, last_day, freq="D"):
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

    if stop_date is not None:
        assert stop_date in [job["date"] for job in jobs]

    config = {
        "definition": "target_examples_v2",
        "run_id": run_id,
        "source_fingerprint": SOURCE_FP,
        "source_config_sha": util.sha(source / "configuration.json"),
        "source_contract_sha": util.sha(source / "output_contract.json"),
        "calendar_sha": util.sha(calendar),
        "code_hashes": code_hashes(),
        "versions": util.versions(),
        "raw_inputs": raw_inputs,
        "inputs": inputs,
        "step_seconds": 10,
        "horizons_seconds": [30, 60],
        "pip_size": PIP,
        "reference": "last real quote known at issue; no additional age cap",
        "future": "eligible reconstructed sample at issue plus horizon",
        "interval": "closed [issue,end]; exact exclusion intersection",
        "label_information": "maximum availability over the whole interval",
        "maturity": "information strictly before assigned period upper boundary",
        "periods": split_manifest("")["periods"],
        "numerical_validation_tolerance_pips": TOL,
        "io": "memory-buffer readback; bounded sharing-error retries",
    }

    fingerprint = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    run = root / "data" / "processed" / run_id
    tables = root / "reports" / "tables" / "targets" / run_id
    figures = root / "reports" / "figures" / "targets" / run_id

    for folder in (run, run / "_checkpoints", tables, figures):
        folder.mkdir(parents=True, exist_ok=True)

    path = run / "configuration.json"
    if path.exists():
        old = fx.load_json(path)
        changed = [
            k for k in sorted(set(old) | set(config))
            if old.get(k) != config.get(k)
        ]
        assert not changed, (
            f"Configuration changed in {changed}; use a new RUN_ID."
        )
    else:
        write_json(path, config)

    write_json(run / "split_manifest.json", split_manifest(fingerprint))
    write_json(
        run / "forecast_output_contract.json",
        forecast_contract(fingerprint),
    )
    write_csv(pd.DataFrame(inputs), tables / "sample_input_inventory.csv")
    write_csv(pd.DataFrame(raw_inputs), tables / "quote_input_inventory.csv")

    return {
        "root": root,
        "run": run,
        "tables": tables,
        "figures": figures,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "config": config,
        "inputs": inputs,
        "raw_inputs": raw_inputs,
        "windows": windows,
        "first_grid": first_grid,
        "last_grid": last_grid,
        "jobs": jobs,
        "target_jobs": [
            j for j in jobs
            if stop_date is None or j["date"] <= stop_date
        ],
        "memory": memory,
        "threads": threads,
    }

def check_snapshot(ctx):
    source = ctx["root"] / "data" / "processed" / SAMPLE_RUN
    config = ctx["config"]

    assert code_hashes() == config["code_hashes"]
    assert util.versions() == config["versions"]
    assert util.sha(source / "configuration.json") == config["source_config_sha"]
    assert util.sha(source / "output_contract.json") == config["source_contract_sha"]

    calendar = (
        ctx["root"] / "data" / "processed" / "etl_01_v3"
        / "merged_exclusion_windows.json"
    )
    assert util.sha(calendar) == config["calendar_sha"]

    for item in ctx["inputs"] + ctx["raw_inputs"]:
        actual = util.file_info(ctx["root"] / item["path"])
        assert all(actual[k] == item[k] for k in actual), item["path"]
        assert util.sha(
            ctx["root"] / item["checkpoint_path"]
        ) == item["checkpoint_sha256"]

def reference_lookup(con, grid, quote_query):
    con.register("_issue_grid", grid)
    return con.sql(f"""
        SELECT
            g.sample_ns AS issue_ns,
            q.ts_ns AS reference_quote_ns,
            q.bid AS reference_bid,
            q.ask AS reference_ask,
            (q.bid + q.ask) / 2 AS reference_mid
        FROM _issue_grid g
        ASOF LEFT JOIN ({quote_query}) q
        ON g.sample_ns >= q.ts_ns
        ORDER BY g.sample_ns
    """).to_arrow_table()

def load_day(ctx, job, con):
    lower, upper = job["lower_ns"], job["upper_ns"]

    paths = [
        str(ctx["root"] / p["path"])
        for p in ctx["inputs"]
        if p["last_ns"] >= lower and p["first_ns"] < upper + 6 * STEP
    ]
    con.read_parquet(
        paths, hive_partitioning=False
    ).create_view("_samples", replace=True)

    source = con.sql(f"""
        SELECT {", ".join(SAMPLE_COLS)}
        FROM _samples
        WHERE sample_ns >= {lower}
          AND sample_ns < {upper + 6 * STEP}
        ORDER BY sample_ns
    """).to_arrow_table()

    times = get(source, "sample_ns")
    assert int(times[0]) == lower
    assert int(times[-1]) == min(upper + 5 * STEP, ctx["last_grid"])
    assert len(times) == (int(times[-1]) - lower) // STEP + 1

    index = next(
        i for i, p in enumerate(ctx["raw_inputs"])
        if p["year_month"] == job["date"][:7]
    )
    current = ctx["raw_inputs"][index]

    con.read_parquet(
        str(ctx["root"] / current["path"]),
        hive_partitioning=False,
    ).create_view("_real_month", replace=True)

    prior = con.sql(f"""
        SELECT ts_ns, bid, ask
        FROM _real_month
        WHERE ts_ns < {lower}
        ORDER BY ts_ns DESC
        LIMIT 1
    """).to_arrow_table()
    con.register("_prior_quote", prior)

    previous = None if index == 0 else fx.load_json(
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
    con.register("_seed_quote", seed)

    query = f"""
        SELECT ts_ns, bid, ask
        FROM _real_month
        WHERE ts_ns >= {lower} AND ts_ns < {upper}
        UNION ALL
        SELECT * FROM _prior_quote
        UNION ALL
        SELECT * FROM _seed_quote
    """

    references = reference_lookup(
        con,
        source.select(["sample_ns"]).slice(0, job["rows"]),
        query,
    )
    return source, references



def build(source, references, windows):
    n, m = references.num_rows, source.num_rows
    assert 0 < n <= m

    t = get(source, "sample_ns").astype(np.int64, copy=False)
    assert np.all(np.diff(t) == STEP)
    assert np.array_equal(t[:n], get(references, "issue_ns"))

    for name in (
        "sample_ns", "available_ns", "has_price", "in_long_gap",
        "project_excluded", "project_eligible",
    ):
        assert source[name].null_count == 0, name

    has = get(source, "has_price")
    mid = get(source, "mid", np.nan)
    av = get(source, "available_ns")

    assert np.all(av >= t)
    assert np.all(np.isfinite(mid[has]) & (mid[has] > 0))
    assert np.all(~np.isfinite(mid[~has]))
    assert not np.any(has & get(source, "in_long_gap"))

    for name in ("source_ts_ns", "source_available_ns", "source_age_ns"):
        assert not np.any(
            has & source[name].is_null().to_numpy(zero_copy_only=False)
        )

    age = get(source, "source_age_ns", 0)
    source_t = get(source, "source_ts_ns", 0)
    assert np.all((age[has] >= 0) & (age[has] < STEP))
    assert np.all(source_t[has] + age[has] == t[has])
    assert np.all(
        av[has] >= get(source, "source_available_ns", 0)[has]
    )

    issue = t[:n]
    ref_ok = ~references["reference_quote_ns"].is_null().to_numpy(
        zero_copy_only=False
    )
    ref_t = get(references, "reference_quote_ns", 0)
    ref_mid = get(references, "reference_mid", np.nan)
    bid = get(references, "reference_bid", np.nan)
    ask = get(references, "reference_ask", np.nan)

    assert np.all(ref_t[ref_ok] <= issue[ref_ok])
    assert np.all(
        np.isfinite(ref_mid[ref_ok])
        & (bid[ref_ok] > 0)
        & (ask[ref_ok] >= bid[ref_ok])
    )
    assert np.all((issue - ref_t)[has[:n] & ref_ok] <= 300 * NS)

    exact_start_excluded = overlap(issue, issue, windows)
    assert np.array_equal(
        exact_start_excluded,
        get(source, "project_excluded")[:n],
    )

    role, cutoff = roles_at(issue)
    columns = {
        name: references[name]
        for name in references.column_names
    }
    columns.update({
        "split": pa.array(role),
        "split_information_cutoff_ns": arr(cutoff),
        "reference_age_ns": arr(issue - ref_t, ref_ok),
        "label_start_has_price": source["has_price"].slice(0, n),
        "label_start_project_eligible": source["project_eligible"].slice(0, n),
        "label_start_is_interpolated": source["is_interpolated"].slice(0, n),
        "label_start_available_ns": source["available_ns"].slice(0, n),
    })

    for horizon, steps in HORIZONS.items():
        index = np.arange(n) + steps
        covered = index < m
        safe = np.minimum(index, m - 1)

        all_prices = covered.copy()
        information = np.maximum(av[:n], ref_t)

        for offset in range(steps + 1):
            j = np.minimum(np.arange(n) + offset, m - 1)
            all_prices &= (np.arange(n) + offset < m) & has[j]
            information = np.maximum(information, av[j])

        excluded = overlap(issue, issue + steps * STEP, windows)
        valid = covered & all_prices & ~excluded & ref_ok
        purged = valid & (information >= cutoff)

        reason = np.full(n, "valid", dtype="<U24")
        reason[~ref_ok] = "no_known_reference"
        reason[covered & ~all_prices] = "disruption"
        reason[excluded] = "excluded_interval"
        reason[~covered] = "future_not_covered"

        columns.update({
            f"label_end_ns_{horizon}": arr(issue + steps * STEP),
            f"future_covered_{horizon}": arr(covered),
            f"interval_has_missing_prices_{horizon}": arr(
                ~all_prices, covered
            ),
            f"interval_overlaps_exclusion_{horizon}": arr(excluded),
            f"label_valid_{horizon}": arr(valid),
            f"label_reason_{horizon}": pa.array(reason),
            f"label_available_ns_{horizon}": arr(information, covered),
            f"purged_at_split_{horizon}": arr(purged),
            f"usable_{horizon}": arr(valid & ~purged),
            f"target_future_mid_{horizon}": arr(mid[safe], valid),
            f"target_change_{horizon}_pips": arr(
                (mid[safe] - ref_mid) / PIP, valid
            ),
        })

        for name in (
            "has_price", "available_ns", "is_interpolated",
            "source_ts_ns", "source_available_ns", "source_age_ns",
            "left_anchor_ns", "right_anchor_ns",
        ):
            selected = source[name].take(pa.array(safe))
            columns[f"endpoint_{name}_{horizon}"] = pc.if_else(
                pa.array(covered),
                selected,
                pa.scalar(None, type=selected.type),
            )

    return pa.table(columns)

def validate(table, last_grid):
    checks = {}
    t = get(table, "issue_ns")
    role, cutoff = roles_at(t)

    ref = get(table, "reference_mid", np.nan)
    ref_ok = ~table["reference_quote_ns"].is_null().to_numpy(
        zero_copy_only=False
    )
    ref_t = get(table, "reference_quote_ns", 0)

    checks["nonpositive_spacing"] = int(np.count_nonzero(np.diff(t) != STEP))
    checks["incorrect_split"] = int(
        np.count_nonzero(get(table, "split") != role)
    )
    checks["incorrect_cutoff"] = int(np.count_nonzero(
        get(table, "split_information_cutoff_ns") != cutoff
    ))
    checks["reference_from_future"] = int(
        np.count_nonzero(ref_ok & (ref_t > t))
    )

    for horizon, steps in HORIZONS.items():
        end = get(table, f"label_end_ns_{horizon}")
        covered = get(table, f"future_covered_{horizon}")
        missing = get(table, f"interval_has_missing_prices_{horizon}", True)
        excluded = get(table, f"interval_overlaps_exclusion_{horizon}")
        valid = get(table, f"label_valid_{horizon}")
        info = get(table, f"label_available_ns_{horizon}", 0)
        purged = get(table, f"purged_at_split_{horizon}")
        usable = get(table, f"usable_{horizon}")
        target = get(table, f"target_future_mid_{horizon}", np.nan)
        change = get(table, f"target_change_{horizon}_pips", np.nan)

        expected_valid = covered & ~missing & ~excluded & ref_ok

        tests = {
            "wrong_endpoint": end != t + steps * STEP,
            "wrong_coverage": covered != (end <= last_grid),
            "wrong_validity": valid != expected_valid,
            "wrong_purge": purged != (valid & (info >= cutoff)),
            "wrong_usability": usable != (valid & ~purged),
            "missing_targets": valid & (
                ~np.isfinite(target) | ~np.isfinite(change)
            ),
            "invalid_targets_present": ~valid & (
                np.isfinite(target) | np.isfinite(change)
            ),
            "wrong_pip_conversion": valid & (
                np.abs((target - ref) / PIP - change) > TOL
            ),
            "wrong_price_conversion": valid & (
                np.abs(predicted_mid(ref, change) - target) > PIP * TOL
            ),
            "early_information": covered & (info < end),
            "endpoint_information_lost": covered & (
                info < get(table, f"endpoint_available_ns_{horizon}", 0)
            ),
            "start_information_lost": covered & (
                info < get(table, "label_start_available_ns")
            ),
            "valid_reason_mismatch": (
                get(table, f"label_reason_{horizon}") == "valid"
            ) != valid,
        }

        for name, failures in tests.items():
            checks[f"{horizon}_{name}"] = int(np.count_nonzero(failures))

    checks["valid_60_without_valid_30"] = int(np.count_nonzero(
        get(table, "label_valid_60s") & ~get(table, "label_valid_30s")
    ))
    checks["usable_60_without_usable_30"] = int(np.count_nonzero(
        get(table, "usable_60s") & ~get(table, "usable_30s")
    ))

    both_covered = get(table, "future_covered_60s")
    checks["information_decreases_with_horizon"] = int(np.count_nonzero(
        both_covered & (
            get(table, "label_available_ns_60s", 0)
            < get(table, "label_available_ns_30s", 0)
        )
    ))

    assert all(value == 0 for value in checks.values()), checks
    return checks



def fixture(con, origin, quote_seconds, mids, grid_seconds, windows=()):
    qt = origin + np.asarray(quote_seconds, dtype=np.int64) * NS
    mids = np.asarray(mids, dtype=float)
    times = origin + np.asarray(grid_seconds, dtype=np.int64) * NS

    quotes = pa.table({
        "ts_ns": qt,
        "bid": mids - PIP / 2,
        "ask": mids + PIP / 2,
    })
    con.register("_toy_quotes", quotes)

    refs = reference_lookup(
        con,
        pa.table({"sample_ns": times}),
        "SELECT * FROM _toy_quotes",
    )

    values, availability, interpolation, lefts, rights = [], [], [], [], []

    for t in times:
        left = int(np.searchsorted(qt, t, side="right") - 1)
        right = int(np.searchsorted(qt, t, side="left"))
        assert left >= 0 and right < len(qt)

        if qt[left] == t:
            values.append(mids[left])
            availability.append(t)
            interpolation.append(False)
            lefts.append(None)
            rights.append(None)

        elif qt[right] - qt[left] <= 300 * NS:
            weight = (t - qt[left]) / (qt[right] - qt[left])
            values.append(
                mids[left] + weight * (mids[right] - mids[left])
            )
            availability.append(qt[right])
            interpolation.append(True)
            lefts.append(int(qt[left]))
            rights.append(int(qt[right]))

        else:
            values.append(None)
            availability.append(max(t, qt[left] + 300 * NS + 1))
            interpolation.append(None)
            lefts.append(None)
            rights.append(None)

    has = np.array([v is not None for v in values])
    excluded = overlap(times, times, windows)

    source = pa.table({
        "sample_ns": times,
        "available_ns": np.asarray(availability, dtype=np.int64),
        "has_price": has,
        "in_long_gap": ~has,
        "project_excluded": excluded,
        "project_eligible": has & ~excluded,
        "mid": pa.array(values, type=pa.float64()),
        "is_interpolated": pa.array(interpolation, type=pa.bool_()),
        "source_ts_ns": arr(times, has),
        "source_available_ns": arr(
            np.asarray(availability, dtype=np.int64), has
        ),
        "source_age_ns": arr(
            np.zeros(len(times), dtype=np.int64), has
        ),
        "left_anchor_ns": pa.array(lefts, type=pa.int64()),
        "right_anchor_ns": pa.array(rights, type=pa.int64()),
    })
    return source, refs

def synthetic_checks():
    passed = {}

    with duckdb.connect() as con:
        origin = ns("2018-06-15 10:00:00")

        source, refs = fixture(
            con, origin,
            [-10, 30, 60, 100],
            [1.10000, 1.10020, 1.09990, 1.10010],
            np.arange(0, 101, 10),
        )
        table = build(source, refs, [])
        validate(table, origin + 100 * NS)

        assert abs(get(table, "reference_mid")[0] - 1.10000) < 1e-12
        assert abs(get(source, "mid")[0] - 1.10005) < 1e-12
        assert abs(
            get(table, "target_change_30s_pips", np.nan)[0] - 2
        ) < TOL
        assert abs(
            get(table, "target_change_60s_pips", np.nan)[0] + 1
        ) < TOL

        assert get(table, "reference_age_ns", 0)[1] == 20 * NS
        assert get(table, "label_valid_30s")[1]
        passed["known_quote_reference_and_pip_targets"] = True

        narrow = [[origin + 15 * NS, origin + 16 * NS]]
        source2, refs2 = fixture(
            con, origin,
            [-10, 30, 60, 100],
            [1.10000, 1.10020, 1.09990, 1.10010],
            np.arange(0, 101, 10),
            narrow,
        )
        excluded = build(source2, refs2, narrow)

        assert not get(excluded, "label_valid_30s")[0]
        assert get(excluded, "interval_overlaps_exclusion_30s")[0]
        passed["exclusion_between_grid_points"] = True

        source, refs = fixture(
            con, origin,
            [-10, 40, 400],
            [1.1, 1.1002, 1.1003],
            np.arange(0, 401, 10),
        )
        table = build(source, refs, [])
        validate(table, origin + 400 * NS)

        assert get(table, "label_valid_30s")[0]
        assert get(table, "endpoint_is_interpolated_30s", False)[0]
        assert not get(table, "label_valid_60s")[0]
        assert get(table, "label_reason_60s")[0] == "disruption"
        passed["independent_horizons_and_pre_outage_synthetic_prices"] = True

        assert get(table, "label_reason_60s")[-1] == "future_not_covered"
        passed["end_of_source_coverage"] = True

        origin = ns("2018-12-31 23:59:20")
        windows = [[origin + 35 * NS, origin + 45 * NS]]

        source, refs = fixture(
            con, origin,
            [-10, 40, 120],
            [1.1, 1.1002, 1.1003],
            np.arange(0, 121, 10),
            windows,
        )

        av = get(source, "available_ns").copy()
        av[2] = origin + 40 * NS + 123
        source = source.set_column(
            source.schema.get_field_index("available_ns"),
            "available_ns",
            pa.array(av),
        )

        table = build(source, refs, windows)
        validate(table, origin + 120 * NS)

        assert get(table, "label_valid_30s")[0]
        assert get(
            table, "label_available_ns_30s", 0
        )[0] == origin + 40 * NS + 123
        assert get(table, "purged_at_split_30s")[0]
        assert not get(table, "usable_30s")[0]
        assert not get(table, "label_valid_60s")[0]

        passed["interior_information_maximum_integer_precision_and_purging"] = True
        passed["anchor_in_exclusion_outside_nominal_interval_allowed"] = True

        origin = ns("2018-12-31 23:00:00")
        source, refs = fixture(
            con, origin,
            np.arange(-10, 7261, 20),
            1.1 + 0.0001 * np.sin(np.arange(364) / 11),
            np.arange(726) * 10,
        )
        whole = build(source, refs.slice(0, 720), [])

        cut = origin + 3600 * NS
        seeded_refs = reference_lookup(
            con,
            source.select(["sample_ns"]).slice(360, 360),
            f"""
                SELECT * FROM _toy_quotes WHERE ts_ns >= {cut}
                UNION ALL
                (
                    SELECT * FROM _toy_quotes
                    WHERE ts_ns < {cut}
                    ORDER BY ts_ns DESC LIMIT 1
                )
            """,
        )

        assert refs.slice(360, 360).equals(
            seeded_refs, check_metadata=False
        )

        tail = build(source.slice(360), seeded_refs, [])
        validate(whole, origin + 7250 * NS)

        assert whole.slice(360).equals(tail, check_metadata=False)
        passed["daily_boundary_halo_and_year_split"] = True

        assert np.allclose(
            predicted_mid([1.1, 1.1], [2, -1]),
            [1.10020, 1.09990],
            rtol=0,
            atol=1e-12,
        )
        passed["future_prediction_price_contract"] = True

    return {"passed": all(passed.values()), "checks": passed}



def statistics(table, date, seconds):
    summary = {
        "date": date,
        "candidate_rows": table.num_rows,
        "processing_seconds": seconds,
    }
    coverage = []
    role = get(table, "split")

    for horizon in HORIZONS:
        valid = get(table, f"label_valid_{horizon}")
        purged = get(table, f"purged_at_split_{horizon}")
        usable = get(table, f"usable_{horizon}")
        reason = get(table, f"label_reason_{horizon}")

        for split in ROLES:
            use = role == split
            item = {
                "date": date,
                "split": split,
                "horizon": horizon,
                "candidate_rows": int(use.sum()),
                "valid_labels": int((use & valid).sum()),
                "purged_labels": int((use & purged).sum()),
                "usable_labels": int((use & usable).sum()),
            }

            for value in (
                "no_known_reference", "disruption",
                "excluded_interval", "future_not_covered",
            ):
                item[value] = int((use & (reason == value)).sum())

            assert item["candidate_rows"] == item["valid_labels"] + sum(
                item[key] for key in (
                    "no_known_reference", "disruption",
                    "excluded_interval", "future_not_covered",
                )
            )
            coverage.append(item)

    return summary, coverage

def run(ctx):
    started = time.perf_counter()
    check_snapshot(ctx)

    evidence = synthetic_checks()
    assert evidence["passed"]
    write_json(ctx["tables"] / "synthetic_checks.json", evidence)

    checkpoints = []

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": str(ctx["threads"]),
    }) as con:
        for job in ctx["target_jobs"]:
            path = ctx["run"] / f"{job['date']}.parquet"
            checkpoint = (
                ctx["run"] / "_checkpoints" / f"{job['date']}.json"
            )

            if checkpoint.exists():
                saved = fx.load_json(checkpoint)
                assert saved["fingerprint"] == ctx["fingerprint"]
                assert saved["summary"]["date"] == job["date"]
                assert saved["summary"]["candidate_rows"] == job["rows"]
                assert all(
                    value == 0 for value in saved["checks"].values()
                )
                assert util.file_info(path) == saved["output"]
                checkpoints.append(saved)
                continue

            tick = time.perf_counter()
            source, refs = load_day(ctx, job, con)
            table = build(source, refs, ctx["windows"])
            checks = validate(table, ctx["last_grid"])
            assert table.num_rows == job["rows"]

            write_parquet(table, path)
            daily, coverage = statistics(
                table, job["date"], time.perf_counter() - tick
            )

            saved = {
                "fingerprint": ctx["fingerprint"],
                "summary": daily,
                "coverage": coverage,
                "checks": checks,
                "output": util.file_info(path),
            }
            write_json(checkpoint, saved)
            checkpoints.append(saved)

            print(
                f"{job['date']} rows={table.num_rows:,} "
                f"seconds={daily['processing_seconds']:.1f}"
            )
            del table, source, refs

    check_snapshot(ctx)

    daily = pd.DataFrame([item["summary"] for item in checkpoints])
    coverage = pd.DataFrame([
        row for item in checkpoints for row in item["coverage"]
    ])
    checks = pd.DataFrame([
        {"date": item["summary"]["date"], **item["checks"]}
        for item in checkpoints
    ])

    assert daily["date"].tolist() == [
        j["date"] for j in ctx["target_jobs"]
    ]
    assert int(daily["candidate_rows"].sum()) == sum(
        j["rows"] for j in ctx["target_jobs"]
    )
    assert int(checks.drop(columns="date").to_numpy().sum()) == 0

    columns = [
        c for c in coverage.columns
        if c not in ("date", "split", "horizon")
    ]
    totals = (
        coverage.groupby(["split", "horizon"], sort=False)[columns]
        .sum()
        .reset_index()
    )

    complete = len(checkpoints) == len(ctx["jobs"])
    if complete:
        assert int(daily["candidate_rows"].sum()) == sum(
            p["rows"] for p in ctx["inputs"]
        )

    summary = {
        "notebook": "05_targets_and_validation_design.ipynb",
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "source_run_id": SAMPLE_RUN,
        "source_fingerprint": SOURCE_FP,
        "mode": "full_history" if complete else "prefix_review",
        "available_input_months": len(ctx["inputs"]),
        "processed_days": len(checkpoints),
        "processed_through": daily["date"].iloc[-1],
        "full_history_complete": complete,
        "candidate_rows": int(daily["candidate_rows"].sum()),
        "horizons_seconds": [30, 60],
        "example_spacing_seconds": 10,
        "training_period_athens": "2018-01-01 through 2018-12-31",
        "validation_period_athens": "2019-01-01 through 2019-06-30",
        "final_evaluation_period_athens": "2026-01-01 through 2026-10-16",
        "synthetic_checks_passed": evidence["passed"],
        "nonzero_machine_checks": 0,
        "processed_prefix_machine_checks_passed": True,
        "predictions_generated": False,
        "models_fitted": False,
        "human_review": "pending",
        "source_stability_check": (
            "file sizes/times and checkpoint hashes; no bulk data hashes"
        ),
        "io_policy": (
            "memory-buffer readback; bounded Windows sharing-error retries"
        ),
        "elapsed_minutes": round(
            (time.perf_counter() - started) / 60, 3
        ),
    }

    for name, frame in (
        ("daily_summary", daily),
        ("daily_checks", checks),
        ("daily_coverage", coverage),
        ("coverage_by_split", totals),
    ):
        write_csv(frame, ctx["tables"] / f"{name}.csv")

    write_json(ctx["tables"] / "execution_summary.json", summary)
    write_json(ctx["run"] / "output_contract.json", {
        "fingerprint": ctx["fingerprint"],
        "files": "one Parquet per Athens calendar day",
        "timestamp_units": "integer UTC nanoseconds",
        "price_units": "USD per EUR",
        "pip_size": PIP,
        "known_at_issue": [
            "issue_ns", "reference_quote_ns", "reference_bid",
            "reference_ask", "reference_mid", "reference_age_ns",
        ],
        "label_metadata_only": (
            "all target_*, endpoint_*, label_*, interval_*, future_*, "
            "purged_*, usable_*, split and split_information_cutoff_ns"
        ),
        "fit_filter": "split == 'train' and usable_<horizon>",
        "feature_join": "Notebook 06 must use feature availability timestamps",
        "forecast_schema": "forecast_output_contract.json; forecasts absent here",
    })

    return summary, totals

def design_tables(ctx):
    manifest = fx.load_json(ctx["run"] / "split_manifest.json")
    contract = fx.load_json(ctx["run"] / "forecast_output_contract.json")
    return (
        pd.DataFrame(manifest["periods"]),
        pd.DataFrame({"Notebook 06 output": contract["columns"]}),
    )

def examples(ctx):
    paths = [
        ctx["run"] / f"{j['date']}.parquet"
        for j in ctx["target_jobs"]
        if j["date"] < "2026-01-01"
    ]
    selected = []

    for path in paths:
        with pa.BufferReader(path.read_bytes()) as buffer:
            table = pq.read_table(
                buffer, use_threads=False, pre_buffer=False
            )
        frame = table.to_pandas(types_mapper=pd.ArrowDtype)

        base = [
            "issue_ns", "split", "reference_quote_ns",
            "reference_mid", "reference_age_ns",
        ]
        detail = []
        for horizon in HORIZONS:
            detail += [
                f"label_end_ns_{horizon}",
                f"label_available_ns_{horizon}",
                f"target_future_mid_{horizon}",
                f"target_change_{horizon}_pips",
                f"label_valid_{horizon}",
                f"purged_at_split_{horizon}",
                f"usable_{horizon}",
                f"label_reason_{horizon}",
                f"endpoint_is_interpolated_{horizon}",
            ]

        masks = {
            "valid": frame["label_valid_30s"],
            "invalid": ~frame["label_valid_30s"],
            "synthetic_endpoint": (
                frame["label_valid_30s"]
                & frame["endpoint_is_interpolated_30s"].fillna(False)
            ),
            "purged": (
                frame["purged_at_split_30s"]
                | frame["purged_at_split_60s"]
            ),
            "delayed": (
                frame["label_valid_30s"]
                & (
                    frame["label_available_ns_30s"]
                    > frame["label_end_ns_30s"]
                )
            ),
        }

        for name, mask in masks.items():
            part = frame.loc[mask, base + detail].head(8).copy()
            part.insert(0, "example_type", name)
            selected.append(part)
        break

    result = (
        pd.concat(selected, ignore_index=True)
        if selected else pd.DataFrame()
    )

    if not result.empty:
        for name in [
            c for c in result.columns
            if c.endswith("_ns")
            or c.startswith(("label_end_ns_", "label_available_ns_"))
        ]:
            if name != "reference_age_ns":
                result[name + "_athens"] = pd.to_datetime(
                    result[name], unit="ns", utc=True
                ).dt.tz_convert(TZ)
        result["reference_age_seconds"] = result["reference_age_ns"] / NS

    write_csv(result, ctx["tables"] / "hand_check_examples.csv")
    return result

def record(ctx, summary):
    scope = (
        f"{summary['processed_days']} days "
        f"through {summary['processed_through']}"
    )
    status = (
        "full history"
        if summary["full_history_complete"]
        else "technical prefix"
    )

    decision = f"""
## Notebook 05 — target design
- Agreed: examples every 10 seconds; future mid changes over 30 and 60 seconds.
- Reference: latest real quote known at issue time; retain its timestamp and age.
- Future prices: eligible real and reconstructed observations.
- Reject intervals crossing exclusions or disruptions. Keep each horizon independent.
- Preserve valid synthetic outcomes before a dead zone.
- Label availability: latest information time across the whole interval.
- Purge labels not fully available before their assigned split ends.
- Train: 2018. Validate: January–June 2019. Final evaluation: 2026 through October 16.
- Notebook 06 will output both predicted pips and expected EURUSD rates.
- No model or trading rule is fitted here.
"""

    contribution = f"""
## Notebook 05 — Raymond Fendi's agreed design
- Defined the simulator's purpose: use trend, MA distances, volatility and entropy
  to decide how quickly to dispose of a simulated position.
- Requested side-dependent good/medium/bad interpretation and intended exits at
  1, 30 or 60 seconds, potentially in several clips.
- Requested a dedicated interpretation discussion before trading rules are built.
- Chose 10-second examples, 30/60-second forecasts, eligible synthetic outcomes,
  2018 training, first-half 2019 validation and the 2026 final evaluation period.
- Requested both predicted change in pips and the corresponding expected rate.
- OpenAI Codex supplied the replacement notebook, helper code and validation code.
- Execution evidence for this run: {status}, {scope}; machine checks passed.
- Human acceptance of these replacement outputs remains pending.
- Good/medium/bad thresholds and their exit mapping remain provisional.
- Results pending execution and validation.
"""

    state = f"""
## Notebook 05 — current state
- Run: {ctx['run_id']}; fingerprint: {ctx['fingerprint']}.
- Completed processing: {status}, {scope}.
- Targets: data/processed/{ctx['run_id']}.
- Reports: reports/tables/targets/{ctx['run_id']}.
- No model was trained; no forecasts or trading results were produced.
- Earlier targets_05_v1 files are superseded and must not be used with this run.
- Upstream outputs were not changed.
- Next: review returned evidence; expand processing only after that review.
- Human acceptance: pending. Trading interpretation criteria: deferred to 09/10.
"""

    for filename, marker, body in (
        ("docs/design_decisions.md", "notebook_05_targets_v2", decision),
        ("CONTRIBUTIONS.md", "notebook_05_contributions_v2", contribution),
        ("docs/project_state.md", "notebook_05_state_v2", state),
    ):
        path = ctx["root"] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        util.write_section(path, marker, body.strip())
