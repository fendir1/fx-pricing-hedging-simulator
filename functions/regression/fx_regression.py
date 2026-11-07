
import gc
import hashlib
import importlib.metadata
import json
import pickle
import time
from collections import deque
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functions.targets import forecast_targets_v2 as tg
from functions.volatility import multiscale_volatility as util

NS, STEP, PIP = tg.NS, tg.STEP, tg.PIP
TZ = "Europe/Athens"
TARGET_RUN = "targets_05_v2"
SAMPLE_RUN = "sampling_02_v1"
ENTROPY_RUN = "entropy_04_v1"

EXPECTED = {
    "targets": "a3ca7b4d579202995b60e4fb7ad1c2e9b0f146aa2317a7bafa9508f510e8072c",
    "sampling": "ad5235dc14f854d9f26ef04dd2c1fdf846faf05cff43eb9f84afccc229628e54",
    "entropy": "355d1378dab75041dc3a5af4a02aea55040be758a3a7eefa1a482c0d3ec6e841",
}

LAGS = {
    "10s": 1,
    "30s": 3,
    "60s": 6,
    "3m": 18,
    "5m": 30,
    "15m": 90,
}
WINDOWS = (15, 30, 60, 100, 200)
ENTROPY_WINDOWS = ("3m", "5m", "15m", "30m", "60m")

GROUPS = {
    "Price changes": [f"past_change_{k}_pips" for k in LAGS],
    "Displayed sizes": ["bid_size_million_eur", "ask_size_million_eur"],
    "Spread": ["spread_pips"],
    "MA distances": [
        f"close_minus_{kind}_{n}_pips"
        for kind in ("sma", "ema")
        for n in WINDOWS
    ],
    "Sample entropy": [
        f"sampen_score_{w}" for w in ENTROPY_WINDOWS
    ],
    "Conditional entropy": [
        f"ce_score_{w}" for w in ENTROPY_WINDOWS
    ],
}
FEATURES = [
    column
    for columns in GROUPS.values()
    for column in columns
]

PERIODS = {
    "train": (tg.ns("2018-01-01"), tg.ns("2019-01-01")),
    "early_stopping": (tg.ns("2019-01-01"), tg.ns("2019-04-01")),
    "comparison": (tg.ns("2019-04-01"), tg.ns("2019-07-01")),
}

BASE = [
    "issue_ns",
    "reference_quote_ns",
    "reference_mid",
    "reference_age_ns",
]
for _h in ("30s", "60s"):
    BASE += [
        f"usable_{_h}",
        f"label_available_ns_{_h}",
        f"target_change_{_h}_pips",
        f"target_future_mid_{_h}",
        f"endpoint_is_interpolated_{_h}",
    ]

def versions():
    return {
        **util.versions(),
        **{
            n: importlib.metadata.version(n)
            for n in ("lightgbm", "scikit-learn")
        },
    }

def dependencies():
    return {
        "regression_helper": util.sha(__file__),
        "target_helper": util.sha(tg.__file__),
        "shared_utilities": util.sha(util.__file__),
        "etl_utilities": util.sha(tg.fx.__file__),
    }

def read(con, paths, columns, order, where="TRUE"):
    con.read_parquet(
        [str(p) for p in paths],
        hive_partitioning=False,
    ).create_view("_input", replace=True)

    return con.sql(
        f"SELECT {', '.join(columns)} "
        f"FROM _input WHERE {where} ORDER BY {order}"
    ).to_arrow_table()

def matrix(table):
    return np.column_stack([
        tg.get(table, c, np.nan) for c in FEATURES
    ]).astype(np.float32, copy=False)

def role_mask(table, role, horizon):
    lower, upper = PERIODS[role]
    t = tg.get(table, "issue_ns")

    return (
        (t >= lower)
        & (t < upper)
        & tg.get(table, f"usable_{horizon}")
        & (
            tg.get(table, f"label_available_ns_{horizon}", 0)
            < upper
        )
    )

def publish(events, times, clock, availability, columns):
    valid = ~events[availability].is_null().to_numpy(
        zero_copy_only=False
    )
    nominal = tg.get(events, clock)[valid]
    av = tg.get(events, availability, 0)[valid]

    assert np.all(av >= nominal)
    assert np.all(np.diff(nominal) > 0)

    order = np.lexsort((nominal, av))

    # Among published rows, prefer the most recent nominal observation.
    latest = np.maximum.accumulate(order)
    index = np.searchsorted(av[order], times, side="right") - 1
    found = index >= 0
    safe = np.maximum(index, 0)

    info = np.zeros(len(times), dtype=np.int64)
    values = {
        c: np.full(len(times), np.nan)
        for c in columns
    }

    if len(order):
        chosen = latest[safe[found]]
        info[found] = av[chosen]

        for c in columns:
            data = tg.get(events, c, np.nan)[valid]
            values[c][found] = data[chosen]

    assert np.all(info <= times)
    return values, info

def rolling_max(values, width):
    out = np.zeros(len(values), dtype=np.int64)
    queue = deque()

    for i, value in enumerate(values):
        while queue and queue[0] <= i - width:
            queue.popleft()

        while queue and values[queue[-1]] <= value:
            queue.pop()

        queue.append(i)

        if i >= width - 1:
            out[i] = values[queue[0]]

    return out

def past_changes(grid, times, reference, reference_time):
    gt = tg.get(grid, "sample_ns")
    av = tg.get(grid, "available_ns")
    has = tg.get(grid, "has_price")
    price = tg.get(grid, "mid", np.nan)

    assert np.all(np.diff(gt) == STEP)
    position = np.searchsorted(gt, times)
    assert np.array_equal(gt[position], times)

    bad = np.r_[0, np.cumsum(~has)]
    values, information = {}, {}

    for key, width in LAGS.items():
        start = position - width
        safe = np.maximum(start, 0)
        enough = start >= 0
        clean = enough & ((bad[position] - bad[safe]) == 0)

        maximum = rolling_max(av, width)
        info = np.maximum(
            reference_time,
            maximum[np.maximum(position - 1, 0)],
        )
        ready = clean & (info <= times)
        column = f"past_change_{key}_pips"

        values[column] = np.where(
            ready,
            (reference - price[safe]) / PIP,
            np.nan,
        )
        information[column] = np.where(
            ready, info, 0
        ).astype(np.int64)

    return values, information

def save_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    tg.retry(lambda: temporary.write_text(text, encoding="utf-8"))
    tg.retry(lambda: temporary.replace(path))

def save_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    tg.retry(lambda: temporary.write_bytes(payload))
    tg.retry(lambda: temporary.replace(path))

def ridge_model():
    return make_pipeline(
        SimpleImputer(strategy="mean", copy=True),
        StandardScaler(copy=False),
        Ridge(alpha=1.0, solver="cholesky"),
    )

def check_evidence(ctx):
    path = ctx["reports"] / "synthetic_checks.json"
    assert path.exists(), (
        "Synthetic-check evidence is required before historical processing."
    )
    evidence = tg.fx.load_json(path)
    assert evidence["fingerprint"] == ctx["fingerprint"]
    assert evidence["passed"]



def prepare(
    root, run_id, memory="2GB", threads=2, working_gb=3.0,
    maximum_trees=1000, early_stopping_rounds=50,
):
    root = Path(root).resolve()
    assert run_id and Path(run_id).name == run_id
    assert maximum_trees > 0 and early_stopping_rounds > 0

    sample = root / "data" / "processed" / SAMPLE_RUN
    target = root / "data" / "processed" / TARGET_RUN
    entropy = root / "outputs" / "features" / "entropy" / ENTROPY_RUN

    locations = {
        "sampling": root / "reports" / "tables" / "sampling" / SAMPLE_RUN,
        "targets": root / "reports" / "tables" / "targets" / TARGET_RUN,
        "entropy": root / "reports" / "tables" / "entropy" / ENTROPY_RUN,
    }

    for name, folder in locations.items():
        summary = tg.fx.load_json(folder / "execution_summary.json")
        assert summary["fingerprint"] == EXPECTED[name]
        assert summary["full_history_complete"]
        assert summary["nonzero_machine_checks"] == 0

        if name != "targets":
            completed = tg.fx.load_json(
                folder / "full_history_completion_checks.json"
            )
            assert all(completed["checks"].values())

    source_config = tg.fx.load_json(sample / "configuration.json")
    calendar = (
        root / "data" / "processed" / "etl_01_v3"
        / "merged_exclusion_windows.json"
    )
    assert util.sha(calendar) == source_config["calendar_sha256"]

    inputs = []

    def capture(kind, path, expected, checkpoint):
        info = util.file_info(path)
        assert all(info[k] == expected[k] for k in info), str(path)

        inputs.append({
            "kind": kind,
            "path": path.relative_to(root).as_posix(),
            **info,
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha": util.sha(checkpoint),
        })

    months = [
        str(m)
        for m in pd.period_range("2018-01", "2019-06", freq="M")
    ]
    source_months = ["2017-12", *months]
    raw = {
        p["year_month"]: p
        for p in source_config["inputs"]
    }

    for month in source_months:
        checkpoint = sample / "_checkpoints" / f"{month}.json"
        saved = tg.fx.load_json(checkpoint)

        assert saved["fingerprint"] == EXPECTED["sampling"]
        assert all(v == 0 for v in saved["checks"].values())

        for kind in ("sampled_10s", "candles_3min"):
            capture(
                kind,
                sample / kind / f"{month}.parquet",
                saved["outputs"][kind],
                checkpoint,
            )

        original = raw[month]
        cp = (
            root / "data" / "processed" / "etl_01_v3"
            / "_checkpoints" / f"observed_{month}.json"
        )
        assert util.sha(cp) == original["checkpoint_sha256"]
        capture("observed", root / original["path"], original, cp)

    for day in pd.date_range(
        "2017-12-31", "2019-06-30", freq="D"
    ):
        date = day.strftime("%Y-%m-%d")
        checkpoint = entropy / "_checkpoints" / f"{date}.json"
        saved = tg.fx.load_json(checkpoint)

        assert saved["fingerprint"] == EXPECTED["entropy"]
        assert all(v == 0 for v in saved["checks"].values())

        capture(
            "entropy",
            entropy / f"{date}.parquet",
            saved["output"],
            checkpoint,
        )

        if date >= "2018-01-01":
            checkpoint = target / "_checkpoints" / f"{date}.json"
            saved = tg.fx.load_json(checkpoint)

            assert saved["fingerprint"] == EXPECTED["targets"]
            assert all(v == 0 for v in saved["checks"].values())

            capture(
                "targets",
                target / f"{date}.parquet",
                saved["output"],
                checkpoint,
            )

    contracts = {
        path.relative_to(root).as_posix(): util.sha(path)
        for folder in (sample, target, entropy)
        for path in (
            folder / "configuration.json",
            folder / "output_contract.json",
        )
    }
    contracts[
        "data/processed/etl_01_v3/merged_exclusion_windows.json"
    ] = util.sha(calendar)
    contracts[
        "data/processed/targets_05_v2/forecast_output_contract.json"
    ] = util.sha(target / "forecast_output_contract.json")

    params = {
        "objective": "regression_l1",
        "metric": ["l1", "rmse"],
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 1000,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "num_threads": threads,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 42,
        "use_missing": True,
        "zero_as_missing": False,
    }

    config = {
        "definition": "regression_06_v1",
        "run_id": run_id,
        "upstream": EXPECTED,
        "inputs": inputs,
        "contracts": contracts,
        "dependencies": dependencies(),
        "versions": versions(),
        "features": GROUPS,
        "periods_ns": PERIODS,
        "params": params,
        "maximum_trees": maximum_trees,
        "early_stopping_rounds": early_stopping_rounds,
        "inference_latency_ns": 0,
        "price_change_continuity": (
            "all intervening historical samples known; "
            "reset after actual gaps"
        ),
        "missing_features": "native NaN; no entropy zero imputation",
        "forecast_estimate": "conditional median point forecast",
        "model_available_ns": PERIODS["comparison"][0],
        "comparison_scope": "April-June 2019; final 2026 unopened",
    }

    # Normalize tuples and lists before restart comparisons.
    config = json.loads(json.dumps(config))
    fingerprint = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()

    feature_dir = root / "outputs" / "features" / run_id
    model_dir = root / "outputs" / "models" / run_id
    forecast_dir = root / "outputs" / "forecasts" / run_id
    reports = root / "reports" / "tables" / "regression" / run_id

    for path in (
        feature_dir,
        feature_dir / "_checkpoints",
        model_dir,
        forecast_dir,
        reports,
    ):
        path.mkdir(parents=True, exist_ok=True)

    path = feature_dir / "configuration.json"
    if path.exists():
        old = tg.fx.load_json(path)
        changed = [
            k for k in config
            if old.get(k) != config[k]
        ]
        assert not changed, (
            f"Changed configuration {changed}; use a new RUN_ID."
        )
    else:
        tg.write_json(path, config)

    tg.write_csv(
        pd.DataFrame(inputs),
        reports / "input_inventory.csv",
    )
    tg.write_json(reports / "feature_groups.json", GROUPS)

    return {
        "root": root,
        "feature_dir": feature_dir,
        "model_dir": model_dir,
        "forecast_dir": forecast_dir,
        "reports": reports,
        "months": months,
        "raw": raw,
        "sample": sample,
        "entropy": entropy,
        "target": target,
        "config": config,
        "fingerprint": fingerprint,
        "run_id": run_id,
        "memory": memory,
        "threads": threads,
        "working_gb": working_gb,
    }

def check_snapshot(ctx):
    config = ctx["config"]

    assert dependencies() == config["dependencies"]
    assert versions() == config["versions"]

    for relative, expected in config["contracts"].items():
        assert util.sha(ctx["root"] / relative) == expected, relative

    for item in config["inputs"]:
        actual = util.file_info(ctx["root"] / item["path"])
        assert all(actual[k] == item[k] for k in actual), item["path"]
        assert util.sha(
            ctx["root"] / item["checkpoint"]
        ) == item["checkpoint_sha"]

def paths_for(ctx, kind, lower, upper):
    selected = []

    for item in ctx["config"]["inputs"]:
        if item["kind"] != kind:
            continue

        stem = Path(item["path"]).stem

        if kind in ("entropy", "targets"):
            a = tg.ns(stem)
            b = int((
                pd.Timestamp(stem, tz=TZ)
                + pd.DateOffset(days=1)
            ).value)
        else:
            a, b = tg.fx.month_bounds(stem)

        if a < upper and b > lower:
            selected.append(ctx["root"] / item["path"])

    assert selected, (kind, lower, upper)
    return selected



def make_month(ctx, month, con):
    lower, upper = tg.fx.month_bounds(month)

    base = read(
        con,
        paths_for(ctx, "targets", lower, upper),
        BASE,
        "issue_ns",
    )
    times = tg.get(base, "issue_ns")

    assert len(times) == (upper - lower) // STEP
    assert times[0] == lower
    assert times[-1] == upper - STEP
    assert np.all(np.diff(times) == STEP)

    reference = tg.get(base, "reference_mid")
    reference_time = tg.get(base, "reference_quote_ns")

    real = read(
        con,
        paths_for(ctx, "observed", lower, upper),
        ["ts_ns", "bid", "ask", "bid_size", "ask_size"],
        "ts_ns",
    )

    previous_month = str(pd.Period(month, freq="M") - 1)
    checkpoint = (
        ctx["root"] / "data" / "processed" / "etl_01_v3"
        / "_checkpoints" / f"observed_{previous_month}.json"
    )
    seed = tg.fx.load_json(checkpoint)["last"]

    quote_times = np.r_[
        np.int64(seed["ts_ns"]),
        tg.get(real, "ts_ns"),
    ]
    assert np.all(np.diff(quote_times) > 0)

    index = np.searchsorted(
        quote_times, times, side="right"
    ) - 1

    assert np.all(index >= 0)
    assert np.array_equal(quote_times[index], reference_time)

    quotes = {
        c: np.r_[float(seed[c]), tg.get(real, c)][index]
        for c in ("bid", "ask", "bid_size", "ask_size")
    }
    assert np.allclose(
        (quotes["bid"] + quotes["ask"]) / 2,
        reference,
        rtol=0,
        atol=1e-12,
    )

    values = {
        "bid_size_million_eur": quotes["bid_size"],
        "ask_size_million_eur": quotes["ask_size"],
        "spread_pips": (quotes["ask"] - quotes["bid"]) / PIP,
    }
    information = {
        "market_information_ns": reference_time,
    }
    del real, quote_times, quotes

    grid = read(
        con,
        paths_for(ctx, "sampled_10s", lower - 90 * STEP, upper),
        ["sample_ns", "available_ns", "has_price", "mid"],
        "sample_ns",
        f"sample_ns >= {lower - 90 * STEP} "
        f"AND sample_ns < {upper}",
    )
    changes, change_times = past_changes(
        grid, times, reference, reference_time
    )
    values.update(changes)
    information.update({
        c + "_information_ns": v
        for c, v in change_times.items()
    })
    del grid

    ma_columns = GROUPS["MA distances"]
    candles = read(
        con,
        paths_for(
            ctx, "candles_3min",
            lower - 32 * 86400 * NS, upper,
        ),
        ["candle_end_ns", "ma_available_ns", *ma_columns],
        "candle_end_ns",
        "ma_eligible",
    )
    selected, info = publish(
        candles,
        times,
        "candle_end_ns",
        "ma_available_ns",
        ma_columns,
    )
    values.update(selected)
    information["ma_information_ns"] = info
    del candles

    entropy_columns = ["sample_ns"]
    for family in ("sampen", "ce"):
        for w in ENTROPY_WINDOWS:
            entropy_columns += [
                f"{family}_score_{w}",
                f"{family}_available_ns_{w}",
            ]

    events = read(
        con,
        paths_for(ctx, "entropy", lower - 86400 * NS, upper),
        entropy_columns,
        "sample_ns",
        f"sample_ns >= {lower - 86400 * NS} "
        f"AND sample_ns < {upper}",
    )

    # Keep null/unready events: they replace earlier defined values.
    for family in ("sampen", "ce"):
        for w in ENTROPY_WINDOWS:
            column = f"{family}_score_{w}"
            selected, info = publish(
                events,
                times,
                "sample_ns",
                f"{family}_available_ns_{w}",
                [column],
            )
            values.update(selected)
            information[column + "_information_ns"] = info

    assert set(values) == set(FEATURES)

    output = {
        c: base[c]
        for c in base.column_names
    }

    for c in FEATURES:
        output[c] = pa.array(
            values[c],
            type=pa.float32(),
            from_pandas=True,
        )

    for c, value in information.items():
        assert np.all(value <= times), c
        output[c] = tg.arr(value, value > 0)

    output["input_information_max_ns"] = pa.array(
        np.maximum.reduce(list(information.values())),
        type=pa.int64(),
    )

    table = pa.table(output)
    x = matrix(table)

    checks = {
        "infinite_features": int(np.isinf(x).sum()),
        "information_after_issue": int(np.count_nonzero(
            tg.get(table, "input_information_max_ns") > times
        )),
        "reference_after_issue": int(np.count_nonzero(
            reference_time > times
        )),
        "wrong_feature_count": int(x.shape[1] != 29),
    }

    for c in GROUPS["Sample entropy"] + GROUPS["Conditional entropy"]:
        v = values[c]
        checks[c + "_outside_0_1"] = int(np.count_nonzero(
            np.isfinite(v) & ((v < 0) | (v > 1))
        ))

    assert all(v == 0 for v in checks.values()), checks

    counts = {
        "month": month,
        "rows": table.num_rows,
    }
    for role in PERIODS:
        for h in ("30s", "60s"):
            counts[f"{role}_{h}"] = int(
                role_mask(table, role, h).sum()
            )

    missing = [
        {
            "month": month,
            "feature": c,
            "missing": int(table[c].null_count),
            "rows": table.num_rows,
        }
        for c in FEATURES
    ]

    return table, checks, counts, missing

def assemble(ctx):
    check_evidence(ctx)
    check_snapshot(ctx)
    records = []

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": str(ctx["threads"]),
    }) as con:
        for month in ctx["months"]:
            path = ctx["feature_dir"] / f"{month}.parquet"
            checkpoint = (
                ctx["feature_dir"] / "_checkpoints"
                / f"{month}.json"
            )

            if checkpoint.exists():
                saved = tg.fx.load_json(checkpoint)
                assert saved["fingerprint"] == ctx["fingerprint"]
                assert util.file_info(path) == saved["output"]
                assert all(
                    v == 0 for v in saved["checks"].values()
                )
            else:
                started = time.perf_counter()
                table, checks, counts, missing = make_month(
                    ctx, month, con
                )
                tg.write_parquet(table, path)

                saved = {
                    "fingerprint": ctx["fingerprint"],
                    "checks": checks,
                    "counts": counts,
                    "missing": missing,
                    "output": util.file_info(path),
                }
                tg.write_json(checkpoint, saved)

                print(
                    f"{month}: {table.num_rows:,} rows, "
                    f"{time.perf_counter() - started:.1f}s"
                )
                del table

            records.append(saved)

    check_snapshot(ctx)

    counts = pd.DataFrame([
        r["counts"] for r in records
    ])
    missing = pd.DataFrame([
        v for r in records for v in r["missing"]
    ])
    checks = pd.DataFrame([
        {"month": r["counts"]["month"], **r["checks"]}
        for r in records
    ])

    assert int(
        checks.drop(columns="month").to_numpy().sum()
    ) == 0

    tg.write_csv(
        counts, ctx["reports"] / "feature_month_counts.csv"
    )
    tg.write_csv(
        missing, ctx["reports"] / "feature_missing_counts.csv"
    )
    tg.write_csv(
        checks, ctx["reports"] / "feature_checks.csv"
    )

    rows = []
    for role in PERIODS:
        for h in ("30s", "60s"):
            n = int(counts[f"{role}_{h}"].sum())
            rows.append({
                "Period": role,
                "Horizon": h,
                "Usable labels": n,
            })

    return pd.DataFrame(rows)

def memory_report(ctx, counts):
    rows = []

    for h in ("30s", "60s"):
        n = int(counts.loc[
            counts["Horizon"].eq(h)
            & counts["Period"].isin(["train", "early_stopping"]),
            "Usable labels",
        ].sum())

        raw_gb = n * (4 * len(FEATURES) + 8) / 1024**3
        working_gb = 6 * raw_gb

        rows.append({
            "Horizon": h,
            "Input arrays GiB": raw_gb,
            "Estimated working GiB": working_gb,
            "Budget GiB": ctx["working_gb"],
        })

    report = pd.DataFrame(rows)
    tg.write_csv(
        report, ctx["reports"] / "memory_estimate.csv"
    )

    assert report["Estimated working GiB"].le(
        ctx["working_gb"]
    ).all(), (
        "Memory estimate exceeds the settings budget. "
        "Review before fitting."
    )

    return report



def check_features(ctx):
    for month in ctx["months"]:
        checkpoint = (
            ctx["feature_dir"] / "_checkpoints"
            / f"{month}.json"
        )
        saved = tg.fx.load_json(checkpoint)

        assert saved["fingerprint"] == ctx["fingerprint"]
        assert all(v == 0 for v in saved["checks"].values())
        assert util.file_info(
            ctx["feature_dir"] / f"{month}.parquet"
        ) == saved["output"]

def load_role(ctx, role, horizon):
    lower, upper = PERIODS[role]

    paths = [
        ctx["feature_dir"] / f"{m}.parquet"
        for m in ctx["months"]
        if tg.fx.month_bounds(m)[0] < upper
        and tg.fx.month_bounds(m)[1] > lower
    ]

    where = (
        f"issue_ns >= {lower} AND issue_ns < {upper} "
        f"AND usable_{horizon} "
        f"AND label_available_ns_{horizon} < {upper}"
    )

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": str(ctx["threads"]),
    }) as con:
        table = read(
            con,
            paths,
            [
                *FEATURES,
                f"target_change_{horizon}_pips",
                f"label_available_ns_{horizon}",
            ],
            f"label_available_ns_{horizon}",
            where,
        )

    assert table.num_rows > 0, (
        role, horizon, "No usable labels"
    )

    x = matrix(table)
    y = tg.get(
        table, f"target_change_{horizon}_pips"
    ).astype(float, copy=True)
    information = int(
        tg.get(table, f"label_available_ns_{horizon}").max()
    )

    assert len(y) and np.isfinite(y).all()
    assert not np.isinf(x).any()
    return x, y, information

def losses(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    assert len(error) and np.isfinite(error).all()

    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }

def fit_models(ctx):
    check_evidence(ctx)
    check_snapshot(ctx)
    check_features(ctx)

    saved_counts = [
        tg.fx.load_json(
            ctx["feature_dir"] / "_checkpoints" / f"{m}.json"
        )["counts"]
        for m in ctx["months"]
    ]
    counts = pd.DataFrame([
        {
            "Period": role,
            "Horizon": h,
            "Usable labels": sum(
                c[f"{role}_{h}"] for c in saved_counts
            ),
        }
        for role in PERIODS
        for h in ("30s", "60s")
    ])
    memory_report(ctx, counts)

    bundles, fit_rows, gains = {}, [], []

    for h in ("30s", "60s"):
        meta_path = ctx["model_dir"] / f"{h}_metadata.json"
        model_path = ctx["model_dir"] / f"{h}_lightgbm.txt"
        ridge_path = ctx["model_dir"] / f"{h}_ridge.pkl"

        if meta_path.exists():
            meta = tg.fx.load_json(meta_path)

            assert meta["fingerprint"] == ctx["fingerprint"]
            assert util.sha(model_path) == meta["lightgbm_sha"]
            assert util.sha(ridge_path) == meta["ridge_sha"]

            booster = lgb.Booster(
                model_str=model_path.read_text(encoding="utf-8")
            )
            ridge = pickle.loads(ridge_path.read_bytes())

        else:
            started = time.perf_counter()

            x, y, fit_info = load_role(ctx, "train", h)
            stop_x, stop_y, stop_info = load_role(
                ctx, "early_stopping", h
            )

            supported = np.isfinite(x).any(axis=0)
            assert supported.all(), (
                "Features unavailable throughout training: "
                + str([
                    c for c, ok in zip(FEATURES, supported)
                    if not ok
                ])
            )

            train_data = lgb.Dataset(
                x,
                label=y,
                feature_name=FEATURES,
            )
            stop_data = lgb.Dataset(
                stop_x,
                label=stop_y,
                reference=train_data,
                feature_name=FEATURES,
            )

            booster = lgb.train(
                ctx["config"]["params"],
                train_data,
                num_boost_round=ctx["config"]["maximum_trees"],
                valid_sets=[stop_data],
                valid_names=["early_stopping"],
                callbacks=[
                    lgb.early_stopping(
                        ctx["config"]["early_stopping_rounds"],
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(0),
                ],
            )

            assert booster.best_iteration >= 1
            del train_data, stop_data
            gc.collect()

            ridge = ridge_model()
            ridge.fit(x, y)

            predicted = booster.predict(
                stop_x,
                num_iteration=booster.best_iteration,
            )
            assert np.isfinite(predicted).all()

            cuts = np.unique(np.quantile(
                predicted, [0.2, 0.4, 0.6, 0.8]
            ))
            cuts = cuts[
                (cuts > predicted.min())
                & (cuts < predicted.max())
            ]

            latest = max(fit_info, stop_info)
            ready = PERIODS["comparison"][0]
            assert latest < ready

            text = booster.model_to_string(
                num_iteration=booster.best_iteration
            )
            restored = lgb.Booster(model_str=text)

            assert np.allclose(
                predicted[:1000],
                restored.predict(stop_x[:1000]),
                rtol=0,
                atol=1e-10,
            )

            payload = pickle.dumps(
                ridge, protocol=pickle.HIGHEST_PROTOCOL
            )
            restored_ridge = pickle.loads(payload)

            assert np.allclose(
                ridge.predict(stop_x[:1000]),
                restored_ridge.predict(stop_x[:1000]),
                rtol=0,
                atol=1e-10,
            )

            save_text(model_path, text)
            save_bytes(ridge_path, payload)

            meta = {
                "fingerprint": ctx["fingerprint"],
                "model_identifier": f"{ctx['run_id']}_{h}",
                "horizon": h,
                "features": FEATURES,
                "objective": "regression_l1",
                "estimate": "conditional median point forecast",
                "training_rows": len(y),
                "early_stopping_rows": len(stop_y),
                "best_iteration": int(booster.best_iteration),
                "fit_information_cutoff_ns": fit_info,
                "selection_information_cutoff_ns": stop_info,
                "training_cutoff_ns": latest,
                "model_available_ns": ready,
                "cutoff_convention": "latest included information time",
                "early_stopping_end_exclusive_ns": ready,
                "inference_latency_ns": ctx["config"]["inference_latency_ns"],
                "bucket_edges": cuts.tolist(),
                "bucket_basis": (
                    "January-March forecasts; "
                    "fixed before April comparison"
                ),
                "early_stopping_scores": losses(stop_y, predicted),
                "lightgbm_sha": util.sha(model_path),
                "ridge_sha": util.sha(ridge_path),
                "fit_minutes": (
                    time.perf_counter() - started
                ) / 60,
            }
            tg.write_json(meta_path, meta)

            del (
                x, y, stop_x, stop_y, predicted,
                restored, restored_ridge, payload
            )
            gc.collect()

        assert (
            meta["training_cutoff_ns"]
            < meta["model_available_ns"]
        )

        bundles[h] = {
            "booster": booster,
            "ridge": ridge,
            "meta": meta,
        }
        fit_rows.append({
            "Horizon": h,
            "Training examples": meta["training_rows"],
            "Early-stopping examples": meta["early_stopping_rows"],
            "Trees retained": meta["best_iteration"],
        })

        importance = booster.feature_importance(
            importance_type="gain"
        )
        for c, value in zip(FEATURES, importance):
            group = next(
                k for k, cols in GROUPS.items()
                if c in cols
            )
            gains.append({
                "horizon": h,
                "group": group,
                "feature": c,
                "gain": float(value),
            })

    check_snapshot(ctx)
    check_features(ctx)

    tg.write_csv(
        pd.DataFrame(gains),
        ctx["reports"] / "feature_gain_diagnostic.csv",
    )
    result = pd.DataFrame(fit_rows)
    tg.write_csv(
        result, ctx["reports"] / "model_fit_summary.csv"
    )

    return bundles, result



def evaluate(
    ctx, bundles,
    example_start="2019-04-01 10:00:00",
):
    check_evidence(ctx)
    check_snapshot(ctx)
    check_features(ctx)

    scores, bucket_rows, all_checks = [], [], []
    examples, example_date = None, None
    demo_start = tg.ns(example_start)

    contract = tg.fx.load_json(
        ctx["target"] / "forecast_output_contract.json"
    )

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": str(ctx["threads"]),
    }) as con:
        for month in ("2019-04", "2019-05", "2019-06"):
            table = read(
                con,
                [ctx["feature_dir"] / f"{month}.parquet"],
                [*BASE, *FEATURES, "input_information_max_ns"],
                "issue_ns",
            )
            times = tg.get(table, "issue_ns")
            reference = tg.get(table, "reference_mid")
            x = matrix(table)

            forecast = {
                c: table[c]
                for c in (
                    "issue_ns",
                    "reference_quote_ns",
                    "reference_mid",
                    "reference_age_ns",
                )
            }
            benchmark = {
                "issue_ns": table["issue_ns"],
            }

            masks, predicted_prices, outcome_prices = {}, {}, {}
            checks = {"month": month}

            for h, bundle in bundles.items():
                meta = bundle["meta"]
                assert np.all(
                    times >= meta["model_available_ns"]
                )

                predicted = bundle["booster"].predict(x)
                ridge = bundle["ridge"].predict(x)

                trailing = tg.get(
                    table, f"past_change_{h}_pips", np.nan
                )
                trailing = np.where(
                    np.isfinite(trailing), trailing, 0.0
                )

                forecasts = {
                    "LightGBM": predicted,
                    "Zero change": np.zeros(len(times)),
                    "Trailing change": trailing,
                    "Ridge": ridge,
                }
                assert all(
                    np.isfinite(v).all()
                    for v in forecasts.values()
                )

                availability = (
                    times + meta["inference_latency_ns"]
                )
                price = tg.predicted_mid(reference, predicted)

                forecast.update({
                    f"predicted_change_{h}_pips": pa.array(predicted),
                    f"expected_eurusd_rate_{h}": pa.array(price),
                    f"forecast_available_ns_{h}": pa.array(availability),
                    f"model_identifier_{h}": pa.array(
                        [meta["model_identifier"]] * len(times)
                    ),
                    f"training_cutoff_ns_{h}": pa.array(
                        np.full(
                            len(times),
                            meta["training_cutoff_ns"],
                            dtype=np.int64,
                        )
                    ),
                })

                for name, value in forecasts.items():
                    if name != "LightGBM":
                        column = (
                            f"{name.lower().replace(' ', '_')}"
                            f"_{h}_pips"
                        )
                        benchmark[column] = pa.array(value)

                mask = role_mask(table, "comparison", h)
                actual = tg.get(
                    table, f"target_change_{h}_pips", np.nan
                )
                actual_price = tg.get(
                    table, f"target_future_mid_{h}", np.nan
                )

                n = int(mask.sum())
                assert n > 0
                assert np.isfinite(actual[mask]).all()
                assert np.isfinite(actual_price[mask]).all()

                for name, value in forecasts.items():
                    error = value[mask] - actual[mask]
                    scores.append({
                        "month": month,
                        "horizon": h,
                        "model": name,
                        "count": n,
                        "eligible": n,
                        "absolute_error_sum": float(
                            np.abs(error).sum()
                        ),
                        "squared_error_sum": float(
                            np.square(error).sum()
                        ),
                    })

                bucket = np.searchsorted(
                    meta["bucket_edges"],
                    predicted,
                    side="right",
                ) + 1

                for b in np.unique(bucket[mask]):
                    use = mask & (bucket == b)
                    bucket_rows.append({
                        "horizon": h,
                        "bucket": int(b),
                        "count": int(use.sum()),
                        "predicted_sum": float(
                            predicted[use].sum()
                        ),
                        "outcome_sum": float(
                            actual[use].sum()
                        ),
                        "absolute_error_sum": float(
                            np.abs(
                                predicted[use] - actual[use]
                            ).sum()
                        ),
                    })

                checks[h + "_nonpositive_forecast_price"] = int(
                    np.count_nonzero(price <= 0)
                )
                checks[h + "_late_inputs"] = int(
                    np.count_nonzero(
                        tg.get(table, "input_information_max_ns")
                        > times
                    )
                )
                checks[h + "_model_from_future"] = int(
                    np.count_nonzero(
                        times <= meta["training_cutoff_ns"]
                    )
                )
                checks[h + "_early_forecast"] = int(
                    np.count_nonzero(availability < times)
                )
                checks[h + "_wrong_outcome_conversion"] = int(
                    np.count_nonzero(
                        mask
                        & (
                            np.abs(
                                tg.predicted_mid(reference, actual)
                                - actual_price
                            ) > 1e-12
                        )
                    )
                )

                masks[h] = mask
                predicted_prices[h] = price
                outcome_prices[h] = actual_price

            assert set(contract["columns"]).issubset(forecast)
            assert all(
                v == 0
                for k, v in checks.items()
                if k != "month"
            ), checks

            all_checks.append(checks)

            # No future outcomes or eligibility flags enter forecast files.
            tg.write_parquet(
                pa.table(forecast),
                ctx["forecast_dir"] / f"{month}.parquet",
            )
            tg.write_parquet(
                pa.table(benchmark),
                ctx["forecast_dir"] / f"{month}_benchmarks.parquet",
            )

            if examples is None:
                eligible = (
                    masks["30s"]
                    & masks["60s"]
                    & (times >= demo_start)
                )
                runs = np.flatnonzero(
                    np.convolve(
                        eligible.astype(int),
                        np.ones(4, dtype=int),
                        mode="valid",
                    ) == 4
                )

                if len(runs):
                    index = np.arange(
                        int(runs[0]), int(runs[0]) + 4
                    )
                    display_time = pd.to_datetime(
                        times[index], unit="ns", utc=True
                    ).tz_convert(TZ)

                    data = {
                        "Forecast time": display_time.strftime("%H:%M:%S"),
                        "Current mid": [
                            f"{v:.6f}"
                            for v in reference[index]
                        ],
                    }

                    for h in ("30s", "60s"):
                        synthetic = tg.get(
                            table,
                            f"endpoint_is_interpolated_{h}",
                            False,
                        )
                        data[f"Predicted +{h}"] = [
                            f"{v:.6f}"
                            for v in predicted_prices[h][index]
                        ]
                        data[f"Outcome +{h}"] = [
                            f"{outcome_prices[h][i]:.6f}"
                            + ("*" if synthetic[i] else "")
                            for i in index
                        ]

                    examples = pd.DataFrame(data)
                    example_date = display_time[0].strftime(
                        "%A, %d %B %Y"
                    )

            print(
                f"{month}: comparison forecasts and checks saved."
            )
            del table, x

    monthly = pd.DataFrame(scores)
    sums = [
        "count",
        "eligible",
        "absolute_error_sum",
        "squared_error_sum",
    ]
    overall = (
        monthly.groupby(
            ["horizon", "model"], sort=False
        )[sums]
        .sum()
        .reset_index()
    )

    for frame in (monthly, overall):
        frame["mae_pips"] = (
            frame["absolute_error_sum"] / frame["count"]
        )
        frame["rmse_pips"] = np.sqrt(
            frame["squared_error_sum"] / frame["count"]
        )
        frame["coverage"] = (
            frame["count"] / frame["eligible"]
        )

    assert overall["mae_pips"].le(
        overall["rmse_pips"] + 1e-10
    ).all()
    assert overall.groupby(
        "horizon"
    )["count"].nunique().eq(1).all()

    buckets = (
        pd.DataFrame(bucket_rows)
        .groupby(["horizon", "bucket"], sort=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    buckets["mean_predicted_pips"] = (
        buckets["predicted_sum"] / buckets["count"]
    )
    buckets["mean_outcome_pips"] = (
        buckets["outcome_sum"] / buckets["count"]
    )
    buckets["mae_pips"] = (
        buckets["absolute_error_sum"] / buckets["count"]
    )

    check_frame = pd.DataFrame(all_checks)
    assert int(
        check_frame.drop(columns="month").to_numpy().sum()
    ) == 0
    assert examples is not None, (
        "No four-row example window; review coverage."
    )

    tg.write_csv(
        overall, ctx["reports"] / "performance_summary.csv"
    )
    tg.write_csv(
        monthly, ctx["reports"] / "monthly_performance.csv"
    )
    tg.write_csv(
        buckets, ctx["reports"] / "forecast_buckets.csv"
    )
    tg.write_csv(
        check_frame, ctx["reports"] / "forecast_checks.csv"
    )
    tg.write_csv(
        examples, ctx["reports"] / "forecast_price_examples.csv"
    )

    friendly = pd.DataFrame({
        "Horizon": overall["horizon"],
        "Model / benchmark": overall["model"],
        "MAE (pips)": overall["mae_pips"].round(4),
        "RMSE (pips)": overall["rmse_pips"].round(4),
        "Coverage": [
            f"{n:,} ({rate:.0%})"
            for n, rate in zip(
                overall["count"], overall["coverage"]
            )
        ],
    })

    evidence = tg.fx.load_json(
        ctx["reports"] / "synthetic_checks.json"
    )

    summary = {
        "notebook": "06_gradient_boosted_regression.ipynb",
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "upstream": EXPECTED,
        "feature_groups": 6,
        "numeric_features": 29,
        "feature_months": len(ctx["months"]),
        "lightgbm_models": 2,
        "ridge_benchmarks": 2,
        "objective": "regression_l1",
        "primary_metric": "MAE in pips",
        "secondary_metric": "RMSE in pips",
        "train": "2018",
        "early_stopping": "January-March 2019",
        "comparison": "April-June 2019",
        "final_evaluation_accessed_for_modeling": False,
        "forecast_rows_include_unscorable_times": True,
        "inference_latency_ns": ctx["config"]["inference_latency_ns"],
        "latency_status": (
            "zero modeled inference latency; "
            "revisit before market replay"
        ),
        "synthetic_checks_passed": evidence["passed"],
        "nonzero_machine_checks": 0,
        "human_review": "pending",
        "example_date": example_date,
        "forecast_interpretation": (
            "conditional median price-change forecast; "
            "no P&L forecast"
        ),
        "group_significance": (
            "deferred until full-model performance review"
        ),
        "source_stability": (
            "file metadata and checkpoint hashes; "
            "no bulk content hashes"
        ),
    }

    check_snapshot(ctx)
    check_features(ctx)
    tg.write_json(
        ctx["reports"] / "execution_summary.json",
        summary,
    )

    return friendly, examples, summary



def synthetic_checks():
    origin = tg.ns("2018-06-15 10:00:00")

    events = pa.table({
        "sample_ns": np.array([
            origin,
            origin + 10 * NS,
            origin + 20 * NS,
            origin + 30 * NS,
        ]),
        "published_ns": np.array([
            origin,
            origin + 20 * NS,
            origin + 20 * NS + 123,
            origin + 50 * NS,
        ]),
        "value": pa.array(
            [1.0, 2.0, None, 9.0],
            type=pa.float64(),
        ),
    })
    times = np.array([
        origin + 19 * NS,
        origin + 20 * NS + 122,
        origin + 20 * NS + 123,
        origin + 50 * NS,
    ], dtype=np.int64)

    values, information = publish(
        events, times, "sample_ns", "published_ns", ["value"]
    )

    assert values["value"][0] == 1
    assert values["value"][1] == 2
    assert np.isnan(values["value"][2])
    assert values["value"][3] == 9
    assert information[2] == origin + 20 * NS + 123

    checks = {
        "publication_time_and_integer_precision": True,
        "published_null_replaces_old_value": True,
    }

    delayed = pa.table({
        "sample_ns": [
            origin, origin + 10 * NS, origin + 20 * NS
        ],
        "published_ns": [
            origin, origin + 40 * NS, origin + 30 * NS
        ],
        "value": [1.0, 2.0, 3.0],
    })
    chosen, _ = publish(
        delayed,
        np.array([origin + 50 * NS]),
        "sample_ns",
        "published_ns",
        ["value"],
    )
    assert chosen["value"][0] == 3
    checks[
        "late_old_publication_does_not_replace_newer_observation"
    ] = True

    gt = origin + np.arange(40, dtype=np.int64) * STEP
    available = gt.copy()
    available[5] = gt[10] + 123

    has = np.ones(40, dtype=bool)
    has[20] = False
    mid = 1.1 + np.arange(40) * PIP

    grid = pa.table({
        "sample_ns": gt,
        "available_ns": available,
        "has_price": has,
        "mid": tg.arr(mid, has),
    })
    selected = np.array([8, 11, 23, 24])

    changes, info = past_changes(
        grid,
        gt[selected],
        mid[selected],
        gt[selected],
    )

    assert np.isnan(changes["past_change_30s_pips"][0])
    assert abs(
        changes["past_change_30s_pips"][1] - 3
    ) < 1e-8
    assert np.isnan(changes["past_change_30s_pips"][2])
    assert abs(
        changes["past_change_30s_pips"][3] - 3
    ) < 1e-8
    assert np.all(
        info["past_change_30s_pips"] <= gt[selected]
    )
    checks[
        "price_changes_wait_for_known_anchors_and_fresh_history"
    ] = True

    cut = PERIODS["early_stopping"][1]
    toy = pa.table({
        "issue_ns": [cut - STEP, cut - STEP],
        "usable_30s": [True, True],
        "label_available_ns_30s": [cut, cut - 1],
    })
    assert np.array_equal(
        role_mask(toy, "early_stopping", "30s"),
        [False, True],
    )
    checks["additional_april_information_cutoff"] = True

    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 3)).astype(np.float32)
    y = x[:, 0].copy()
    x[::7, 1] = np.nan

    training = lgb.Dataset(x[:100], label=y[:100])
    stopping = lgb.Dataset(
        x[100:], label=y[100:], reference=training
    )
    model = lgb.train(
        {
            "objective": "regression_l1",
            "metric": ["l1", "rmse"],
            "min_data_in_leaf": 5,
            "num_leaves": 4,
            "verbosity": -1,
            "num_threads": 1,
        },
        training,
        num_boost_round=5,
        valid_sets=[stopping],
        callbacks=[
            lgb.early_stopping(
                2, first_metric_only=True, verbose=False
            ),
            lgb.log_evaluation(0),
        ],
    )

    restored = lgb.Booster(
        model_str=model.model_to_string()
    )
    assert np.allclose(
        model.predict(x[100:]),
        restored.predict(x[100:]),
        rtol=0,
        atol=1e-10,
    )

    ridge = ridge_model().fit(x[:100], y[:100])
    assert np.isfinite(
        ridge.predict(x[100:])
    ).all()

    checks[
        "lightgbm_native_missing_early_stopping_and_serialization"
    ] = True
    checks[
        "ridge_training_only_imputation_interface"
    ] = True

    assert np.allclose(
        tg.predicted_mid([1.1, 1.1], [2, -1]),
        [1.10020, 1.09990],
        rtol=0,
        atol=1e-12,
    )
    checks["pip_to_price_conversion"] = True

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }

def record(ctx, summary):
    decisions = """
## Notebook 06 — initial regression experiment
- Agreed: two LightGBM models, predicting 30- and 60-second price changes.
- MAE training loss; conditional median point forecasts. No P&L prediction.
- MAE in pips is primary; RMSE in pips is secondary.
- Six groups: past changes, displayed sizes, spread, MA distances, SampEn and CE.
- Past changes: 10/30/60 seconds and 3/5/15 minutes.
- All ten completed-candle SMA/EMA distance columns: 15/30/60/100/200.
- Both entropy families use 3/5/15/30/60-minute scores.
- Retain valid target examples with missing features; native missing handling.
- Current mid/spread/sizes use the last known real quote.
- Historical reconstructed prices enter only after their information is available.
- Past changes need fresh continuous history after actual data gaps.
- Quotes in excluded clock hours can provide history; target exclusions still apply.
- Preserve MA history and the established entropy restart rules.
- Fit on 2018; choose tree count using January-March 2019.
- Early-stopping label information must be available before April 1.
- Compare the fixed models on April-June 2019. Keep 2026 untouched.
- Conservative single parameter configuration; no tuning search or blending.
- Benchmarks: zero change, trailing horizon change and ridge.
- Trailing baseline falls back to zero change when its history is unavailable.
- Ridge uses training-only mean imputation and scaling; source features stay missing.
- Forecast timestamps cover both fitting and early-stopping information.
- April 1 model readiness is a research clock assumption, not measured fit latency.
- Zero modeled inference latency is provisional until replay design.
- Export forecasts without future outcomes or future eligibility flags.
- Later group-removal tests require a separate agreed diagnostic experiment.
- After April-June informs changes, assess the revised specification on a later development block.
- Feature gain is descriptive; it is not a statistical-significance test.
"""

    contributions = """
## Notebook 06 — agreed contributions
- Raymond Fendi chose LightGBM and two joint models using all six feature groups.
- Requested every agreed SMA and EMA lookback, plus sample and conditional entropy.
- Chose MAE primary, RMSE secondary and MAE training loss.
- Agreed the six price-change lookbacks and the two 2019 validation roles.
- Chose to retain valid examples with missing inputs.
- Requested concise performance tables and a 30-second forecast-price example.
- Clarified that P&L must come from simulated trades and accounting.
- Requested later evaluation of each group's additional forecasting value.
- OpenAI Codex supplied the notebook, implementation and inline validation code.
- This executed run produced two LightGBM models and benchmark comparisons.
- Output interpretation and human review remain pending.
"""

    state = f"""
## Notebook 06 — current state
- Run: {ctx['run_id']}; fingerprint: {ctx['fingerprint']}.
- Features: outputs/features/{ctx['run_id']}.
- Models: outputs/models/{ctx['run_id']}.
- Forecasts: outputs/forecasts/{ctx['run_id']}.
- Reports: reports/tables/regression/{ctx['run_id']}.
- Models fit on 2018; early stopping January-March 2019.
- Comparison forecasts cover April-June 2019; final 2026 was not modeled.
- Machine checks passed; output review remains pending.
- Next: review forecasting performance and coverage before further experiments.
- No pricing, inventory or exit policy was implemented.
- No upstream numerical outputs were changed.
"""

    for path, marker, body in (
        (
            "docs/design_decisions.md",
            "notebook_06_design_v1",
            decisions,
        ),
        (
            "CONTRIBUTIONS.md",
            "notebook_06_contributions_v1",
            contributions,
        ),
        (
            "docs/project_state.md",
            "notebook_06_state_v1",
            state,
        ),
        (
            "README.md",
            "notebook_06_method_v1",
            "## Forecasting development\n"
            "- Notebook 05: complete 30/60-second labels and chronological splits.\n"
            "- Notebook 06: two MAE-trained LightGBM models and benchmark comparisons.\n"
            "- Training: 2018; early stopping: January-March 2019; comparison: April-June 2019.\n"
            "- Final evaluation: 2026, reserved. Forecasts do not predict P&L.\n"
            "- Results pending execution and validation.",
        ),
    ):
        util.write_section(
            ctx["root"] / path,
            marker,
            body.strip(),
        )

    util.write_section(
        ctx["root"] / "docs/project_state.md",
        "risk-interpretation-and-targets-2026-11-05",
        "## Earlier target discussion\n"
        "Superseded by the completed Notebook 05 and current Notebook 06 records.\n"
        "The side-dependent interpretation and 1/30/60-second exit agenda remains for 09-10.",
    )

    util.write_section(
        ctx["root"] / "docs/project_state.md",
        "notebook_05_state_v2",
        "## Notebook 05 — current state\n"
        "- Full-history labels: targets_05_v2, 130 months through October 16, 2026.\n"
        "- Codex reviewed saved checks, coverage and checkpoint/file metadata.\n"
        "- No reconciliation failures found; upstream label definitions unchanged.\n"
        "- Notebook 06 uses the reviewed outputs and an additional April 1 information cutoff.",
    )

    tg.write_json(
        ctx["reports"] / "execution_summary.json",
        summary,
    )
