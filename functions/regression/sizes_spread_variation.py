import copy
import gc
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from functions.regression import fx_regression as r
from functions.regression import group_significance as g

SOURCE_RUN = "regression_06_group_tests_v1"
SOURCE_FP = "f761bd52ecbf0f47277b5b617626c45a75644d7519394a01c3887bb8fc9138b1"

COLUMNS = [
    "bid_size_million_eur",
    "ask_size_million_eur",
    "spread_pips",
]

MONTHS = ["2020-04", "2020-05", "2020-06"]
START = r.tg.ns("2020-04-01")
END = r.tg.ns("2020-07-01")

def eligible(table, h):
    t = r.tg.get(table, "issue_ns")
    info = r.tg.get(
        table, f"label_available_ns_{h}", 0
    )

    return (
        (t >= START)
        & (t < END)
        & r.tg.get(table, f"usable_{h}")
        & (info >= t)
        & (info < END)
    )

def check(ctx, built=False):
    g.check(ctx, built=built)
    assert (
        r.util.sha(__file__)
        == ctx["config"]["variation_helper_sha"]
    )

def synthetic_checks():
    cut = END

    table = pa.table({
        "issue_ns": [
            cut - r.STEP,
            cut - r.STEP,
            START - r.STEP,
        ],
        "usable_30s": [True, True, True],
        "label_available_ns_30s": [
            cut, cut - 1, START
        ],
    })

    assert np.array_equal(
        eligible(table, "30s"),
        [False, True, False],
    )

    base = g.synthetic_checks()
    assert base["passed"]

    base["checks"][
        "fresh_quarter_information_cutoff"
    ] = True

    base["checks"]["exactly_three_named_inputs"] = (
        len(COLUMNS) == 3
        and len(set(COLUMNS)) == 3
    )

    base["passed"] = all(base["checks"].values())
    return base

def prepare(
    root, run_id,
    memory="2GB", threads=2, working_gb=3.0,
):
    root = Path(root).resolve()

    assert run_id and Path(run_id).name == run_id
    assert run_id not in (
        SOURCE_RUN,
        g.REFERENCE_RUN,
        "regression_06_v1",
    )

    source_dir = (
        root / "outputs" / "features" / SOURCE_RUN
    )
    source_cfg_path = (
        source_dir / "configuration.json"
    )
    source_reports = (
        root / "reports" / "tables"
        / "regression" / SOURCE_RUN
    )

    old = r.tg.fx.load_json(source_cfg_path)
    summary = r.tg.fx.load_json(
        source_reports / "execution_summary.json"
    )

    assert summary["fingerprint"] == SOURCE_FP

    assert hashlib.sha256(
        json.dumps(
            old,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest() == SOURCE_FP

    assert summary["nonzero_machine_checks"] == 0
    assert (
        old["test_helper_sha"]
        == r.util.sha(g.__file__)
    )

    ref_dir = (
        root / "outputs" / "features"
        / g.REFERENCE_RUN
    )
    ref_cfg = r.tg.fx.load_json(
        ref_dir / "configuration.json"
    )

    original = (
        root / "outputs" / "features"
        / "regression_06_v1"
    )
    original_cfg_path = (
        original / "configuration.json"
    )

    assert (
        r.util.sha(original_cfg_path)
        == ref_cfg["parent_config_sha"]
    )

    original_cfg = r.tg.fx.load_json(
        original_cfg_path
    )

    assert {
        k: list(v)
        for k, v in r.PERIODS.items()
    } == original_cfg["periods_ns"]

    training = {
        "root": root,
        "feature_dir": original,
        "reports": (
            root / "reports" / "tables"
            / "regression" / "regression_06_v1"
        ),
        "months": [
            str(m)
            for m in pd.period_range(
                "2018-01", "2019-06", freq="M"
            )
        ],
        "config": original_cfg,
        "fingerprint": ref_cfg["parent_fingerprint"],
        "memory": memory,
        "threads": threads,
    }

    r.check_evidence(training)

    artifacts = copy.deepcopy(
        old["reference_artifacts"]
    )

    for path in (
        source_cfg_path,
        source_reports / "execution_summary.json",
        source_reports / "omissions_primary_inference.csv",
        source_reports / "confirmation_metrics.csv",
    ):
        artifacts.append({
            "path": path.relative_to(root).as_posix(),
            "sha": r.util.sha(path),
        })

    assert len({
        v["path"] for v in artifacts
    }) == len(artifacts)

    inputs = copy.deepcopy(old["inputs"])

    sample = (
        root / "data" / "processed"
        / r.SAMPLE_RUN
    )

    raw = {
        v["year_month"]: v
        for v in r.tg.fx.load_json(
            sample / "configuration.json"
        )["inputs"]
    }

    additions = []

    for m in MONTHS:
        cp = sample / "_checkpoints" / f"{m}.json"
        saved = r.tg.fx.load_json(cp)

        assert (
            saved["fingerprint"]
            == r.EXPECTED["sampling"]
        )
        assert all(
            v == 0 for v in saved["checks"].values()
        )

        for kind in (
            "sampled_10s", "candles_3min"
        ):
            additions.append((
                kind,
                sample / kind / f"{m}.parquet",
                saved["outputs"][kind],
                cp,
            ))

        cp = (
            root / "data" / "processed"
            / "etl_01_v3" / "_checkpoints"
            / f"observed_{m}.json"
        )

        assert (
            r.util.sha(cp)
            == raw[m]["checkpoint_sha256"]
        )

        additions.append((
            "observed",
            root / raw[m]["path"],
            raw[m],
            cp,
        ))

    folders = {
        "targets": (
            root / "data" / "processed"
            / r.TARGET_RUN
        ),
        "entropy": (
            root / "outputs" / "features"
            / "entropy" / r.ENTROPY_RUN
        ),
    }

    for day in pd.date_range(
        "2020-04-01", "2020-06-30", freq="D"
    ):
        date = day.strftime("%Y-%m-%d")

        for kind, folder in folders.items():
            cp = (
                folder / "_checkpoints"
                / f"{date}.json"
            )
            saved = r.tg.fx.load_json(cp)

            assert (
                saved["fingerprint"]
                == r.EXPECTED[kind]
            )
            assert all(
                v == 0
                for v in saved["checks"].values()
            )

            additions.append((
                kind,
                folder / f"{date}.parquet",
                saved["output"],
                cp,
            ))

    for kind, path, expected, cp in additions:
        info = r.util.file_info(path)

        assert all(
            info[k] == expected[k] for k in info
        ), str(path)

        inputs.append({
            "kind": kind,
            "path": path.relative_to(root).as_posix(),
            **info,
            "checkpoint": (
                cp.relative_to(root).as_posix()
            ),
            "checkpoint_sha": r.util.sha(cp),
        })

    assert len({
        (v["kind"], v["path"])
        for v in inputs
    }) == len(inputs)

    config = {
        k: copy.deepcopy(old[k])
        for k in (
            "upstream",
            "contracts",
            "dependencies",
            "versions",
            "periods_ns",
            "params",
            "maximum_trees",
            "early_stopping_rounds",
            "model_features",
            "model_feature_groups",
            "test_helper_sha",
            "reference_fingerprint",
        )
    }

    assert (
        config["params"]["objective"]
        == "regression_l1"
    )
    assert len(config["model_features"]) == 21
    assert all(
        c in config["model_features"]
        for c in COLUMNS
    )

    config.update({
        "definition": "sizes_spread_variation_v1",
        "run_id": run_id,
        "parent_run_id": SOURCE_RUN,
        "parent_fingerprint": SOURCE_FP,
        "parent_config_sha": r.util.sha(
            source_cfg_path
        ),
        "reference_run": g.REFERENCE_RUN,
        "reference_artifacts": artifacts,
        "candidate_features": COLUMNS,
        "inputs": inputs,
        "variation_helper_sha": r.util.sha(__file__),
        "evaluation_periods_ns": {
            "fresh_development": [START, END]
        },
        "architecture_information_cutoff_ns": START - 1,
        "model_available_ns": START,
        "inference_latency_ns": 0,
        "comparison_scope": (
            "April-June 2020 fresh development comparison"
        ),
        "selection": (
            "User-requested fixed sizes/spread recipe; "
            "no automatic promotion"
        ),
        "missing_features": (
            "native NaN; valid target rows retained"
        ),
        "primary_block_days": 5,
        "sensitivity_block_days": 1,
        "resamples": 9999,
        "alpha": 0.05,
        "bootstrap_seed": 1729,
        "multiplicity": (
            "Holm across four tests: "
            "two horizons by two benchmarks"
        ),
        "interval": (
            "pointwise basic block-bootstrap; "
            "not simultaneous"
        ),
        "uncertainty": (
            "approximate and conditional on fixed "
            "models and this quarter"
        ),
        "forecast_estimate": (
            "conditional median price-change "
            "point forecast"
        ),
    })

    config = json.loads(json.dumps(config))

    fp = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    feature_dir = (
        root / "outputs" / "features" / run_id
    )
    model_dir = (
        root / "outputs" / "models" / run_id
    )
    forecast_dir = (
        root / "outputs" / "forecasts" / run_id
    )
    reports = (
        root / "reports" / "tables"
        / "regression" / run_id
    )

    for folder in (
        feature_dir / "_checkpoints",
        model_dir,
        reports,
        forecast_dir / "reference",
        forecast_dir / "candidate",
    ):
        folder.mkdir(
            parents=True, exist_ok=True
        )

    path = feature_dir / "configuration.json"

    if path.exists():
        previous = r.tg.fx.load_json(path)
        changed = [
            k for k in config
            if previous.get(k) != config[k]
        ]

        assert not changed, (
            f"Changed configuration {changed}; "
            "use a new RUN_ID."
        )
    else:
        r.tg.write_json(path, config)

    ctx = {
        "root": root,
        "feature_dir": feature_dir,
        "model_dir": model_dir,
        "forecast_dir": forecast_dir,
        "reports": reports,
        "config": config,
        "fingerprint": fp,
        "run_id": run_id,
        "months": MONTHS,
        "training": training,
        "reference_models": (
            root / "outputs" / "models"
            / g.REFERENCE_RUN
        ),
        "reference_features": {
            "feature_dir": ref_dir,
            "months": [
                "2019-07", "2019-08", "2019-09"
            ],
            "fingerprint": g.REFERENCE_FP,
        },
        "memory": memory,
        "threads": threads,
        "working_gb": working_gb,
    }

    check(ctx)

    r.tg.write_csv(
        pd.DataFrame(inputs),
        reports / "input_inventory.csv",
    )
    return ctx

def build(ctx):
    check(ctx)

    evidence = r.tg.fx.load_json(
        ctx["reports"] / "synthetic_checks.json"
    )
    assert evidence["passed"]
    assert (
        evidence["fingerprint"]
        == ctx["fingerprint"]
    )

    records = []

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": str(ctx["threads"]),
    }) as con:
        for m in MONTHS:
            path = (
                ctx["feature_dir"] / f"{m}.parquet"
            )
            cp = (
                ctx["feature_dir"]
                / "_checkpoints" / f"{m}.json"
            )

            if cp.exists():
                saved = r.tg.fx.load_json(cp)

                assert (
                    saved["fingerprint"]
                    == ctx["fingerprint"]
                )
                assert (
                    r.util.file_info(path)
                    == saved["output"]
                )
                assert all(
                    v == 0
                    for v in saved["checks"].values()
                )
            else:
                table, checks, _, _ = r.make_month(
                    ctx, m, con
                )

                counts = {
                    "month": m,
                    "rows": table.num_rows,
                }
                counts.update({
                    f"valid_{h}": int(
                        eligible(table, h).sum()
                    )
                    for h in ("30s", "60s")
                })

                r.tg.write_parquet(table, path)

                saved = {
                    "fingerprint": ctx["fingerprint"],
                    "checks": checks,
                    "counts": counts,
                    "output": r.util.file_info(path),
                }
                r.tg.write_json(cp, saved)

            records.append(saved)
            print(f"{m}: inputs ready.")

    check(ctx, built=True)

    counts = pd.DataFrame([
        v["counts"] for v in records
    ])
    checks = pd.DataFrame([
        {
            "month": v["counts"]["month"],
            **v["checks"],
        }
        for v in records
    ])

    r.tg.write_csv(
        counts,
        ctx["reports"] / "feature_counts.csv",
    )
    r.tg.write_csv(
        checks,
        ctx["reports"] / "feature_checks.csv",
    )
    return counts

def fit(ctx):
    check(ctx, built=True)
    g.memory_check(ctx)

    references = g.load_reference(ctx)
    bundles, rows = {}, []

    for h in ("30s", "60s"):
        case = {
            "id": f"sizes_spread_{h}",
            "horizon": h,
            "features": COLUMNS,
            "group": "Displayed sizes and spread",
            "ready_ns": START,
        }

        bundles[h] = g.fit_case(
            ctx, case, references
        )
        meta = bundles[h]["meta"]

        assert meta["features"] == COLUMNS
        assert (
            bundles[h]["model"].feature_name()
            == COLUMNS
        )
        assert (
            bundles[h]["model"].num_feature()
            == 3
        )

        rows.append({
            "Horizon": h,
            "Inputs": 3,
            "Training examples": meta[
                "training_rows"
            ],
            "Early-stopping examples": meta[
                "early_stopping_rows"
            ],
            "Trees": meta["best_iteration"],
        })

        print(f"{h}: three-input model ready.")
        gc.collect()

    check(ctx, built=True)

    report = pd.DataFrame(rows)

    r.tg.write_csv(
        report,
        ctx["reports"] / "model_fit_summary.csv",
    )
    return bundles, report

def evaluate(ctx, bundles):
    check(ctx, built=True)
    references = g.load_reference(ctx)

    monthly, daily, checks, examples = [], [], [], []

    descriptors = [
        {
            "id": f"{h}_vs_{key}",
            "horizon": h,
            "group": label,
        }
        for h in ("30s", "60s")
        for key, label in (
            ("reference", "Vs reference 21"),
            ("zero", "Vs zero change"),
        )
    ]

    for m in MONTHS:
        with duckdb.connect(config={
            "memory_limit": ctx["memory"],
            "threads": str(ctx["threads"]),
        }) as con:
            table = r.read(
                con,
                [
                    ctx["feature_dir"]
                    / f"{m}.parquet"
                ],
                [
                    *r.BASE,
                    *r.FEATURES,
                    "input_information_max_ns",
                ],
                "issue_ns",
            )

        times = r.tg.get(table, "issue_ns")
        dates = np.asarray(
            pd.to_datetime(
                times, unit="ns", utc=True
            ).tz_convert(r.TZ).strftime(
                "%Y-%m-%d"
            )
        )

        month_checks = {
            "month": m,
            "late_inputs": int(
                np.count_nonzero(
                    r.tg.get(
                        table,
                        "input_information_max_ns",
                    ) > times
                )
            ),
            "outside_comparison_period": int(
                np.count_nonzero(
                    (times < START)
                    | (times >= END)
                )
            ),
        }

        saved_predictions = {
            "reference": {},
            "candidate": {},
        }

        for h in ("30s", "60s"):
            candidate = bundles[h]
            reference = references[h]

            assert (
                candidate["meta"]["features"]
                == COLUMNS
            )

            values = {
                "Sizes + spread": g.predict(
                    candidate, table
                ),
                "Reference 21": g.predict(
                    reference, table
                ),
                "Zero change": np.zeros(
                    table.num_rows
                ),
            }

            saved_predictions["reference"][h] = (
                values["Reference 21"]
            )
            saved_predictions["candidate"][h] = (
                values["Sizes + spread"]
            )

            for key, model in (
                ("candidate", candidate),
                ("reference", reference),
            ):
                meta = model["meta"]

                assert (
                    meta["training_cutoff_ns"]
                    < meta["model_available_ns"]
                )

                month_checks[
                    f"{key}_{h}_late_model"
                ] = int(np.count_nonzero(
                    (
                        times
                        < meta["model_available_ns"]
                    )
                    | (
                        times
                        <= meta["training_cutoff_ns"]
                    )
                ))

            mask = eligible(table, h)
            actual = r.tg.get(
                table,
                f"target_change_{h}_pips",
                np.nan,
            )[mask]

            assert len(actual)
            assert np.isfinite(actual).all()

            errors = {
                key: prediction[mask] - actual
                for key, prediction in values.items()
            }

            for key, error in errors.items():
                monthly.append({
                    "month": m,
                    "horizon": h,
                    "model": key,
                    "count": len(actual),
                    "absolute_error": float(
                        np.abs(error).sum()
                    ),
                    "squared_error": float(
                        np.square(error).sum()
                    ),
                })

            days, index = np.unique(
                dates[mask],
                return_inverse=True,
            )
            counts = np.bincount(index)

            for key, benchmark in (
                ("reference", "Reference 21"),
                ("zero", "Zero change"),
            ):
                sums = np.bincount(
                    index,
                    weights=(
                        np.abs(
                            errors["Sizes + spread"]
                        )
                        - np.abs(
                            errors[benchmark]
                        )
                    ),
                )

                daily.extend({
                    "date": day,
                    "case_id": f"{h}_vs_{key}",
                    "horizon": h,
                    "count": int(n),
                    "difference_sum": float(s),
                } for day, n, s in zip(
                    days, counts, sums
                ))

            if m == "2020-04":
                first = r.tg.ns(
                    "2020-04-06T10:00:00"
                )
                take = np.flatnonzero(
                    mask
                    & (times >= first)
                    & (
                        times
                        < first + 40 * r.NS
                    )
                )

                mid = r.tg.get(
                    table, "reference_mid"
                )
                future = r.tg.get(
                    table,
                    f"target_future_mid_{h}",
                    np.nan,
                )
                synthetic = r.tg.get(
                    table,
                    f"endpoint_is_interpolated_{h}",
                    False,
                )

                for i in take:
                    examples.append({
                        "Athens time": pd.Timestamp(
                            times[i],
                            unit="ns",
                            tz="UTC",
                        ).tz_convert(
                            r.TZ
                        ).strftime(
                            "%d %b %Y %H:%M:%S"
                        ),
                        "Horizon": h,
                        "Current mid": mid[i],
                        "Predicted change (pips)": (
                            values[
                                "Sizes + spread"
                            ][i]
                        ),
                        "Predicted EURUSD": (
                            mid[i]
                            + r.PIP
                            * values[
                                "Sizes + spread"
                            ][i]
                        ),
                        "Observed future EURUSD": (
                            future[i]
                        ),
                        "Interpolated outcome": bool(
                            synthetic[i]
                        ),
                    })

        assert all(
            v == 0
            for k, v in month_checks.items()
            if k != "month"
        ), month_checks

        # Future-label validity never selects exported forecasts.
        for key, models in (
            ("reference", references),
            ("candidate", bundles),
        ):
            forecast = g.forecast_table(
                table,
                models,
                saved_predictions[key],
            )
            r.tg.write_parquet(
                forecast,
                ctx["forecast_dir"]
                / key / f"{m}.parquet",
            )

        checks.append(month_checks)
        print(
            f"{m}: scored both horizons; "
            "forecasts saved."
        )

    monthly = pd.DataFrame(monthly)
    daily = pd.DataFrame(daily)

    assert monthly.groupby(
        ["month", "horizon"]
    )["count"].nunique().eq(1).all()

    assert daily.groupby(
        ["date", "horizon"]
    )["count"].nunique().eq(1).all()

    totals = monthly.groupby(
        ["horizon", "model"],
        sort=False,
    )[[
        "count",
        "absolute_error",
        "squared_error",
    ]].sum().reset_index()

    totals["mae"] = (
        totals["absolute_error"]
        / totals["count"]
    )
    totals["rmse"] = np.sqrt(
        totals["squared_error"]
        / totals["count"]
    )

    for case in descriptors:
        h = case["horizon"]
        case_id = case["id"]

        benchmark = (
            "Reference 21"
            if case_id.endswith("reference")
            else "Zero change"
        )

        d = daily.loc[
            daily["case_id"].eq(case_id)
        ]
        scores = totals.loc[
            totals["horizon"].eq(h)
        ].set_index("model")

        assert int(d["count"].sum()) == int(
            scores.loc[benchmark, "count"]
        )

        assert np.isclose(
            d["difference_sum"].sum(),
            scores.loc[
                "Sizes + spread",
                "absolute_error",
            ]
            - scores.loc[
                benchmark,
                "absolute_error",
            ],
            rtol=0,
            atol=1e-7,
        )

    for name, frame in (
        ("performance_summary.csv", totals),
        ("monthly_scores.csv", monthly),
        ("paired_daily_losses.csv", daily),
        (
            "evaluation_checks.csv",
            pd.DataFrame(checks),
        ),
        (
            "price_examples.csv",
            pd.DataFrame(examples),
        ),
    ):
        r.tg.write_csv(
            frame, ctx["reports"] / name
        )

    primary, sensitivity = g.infer(
        ctx,
        daily,
        descriptors,
        stem="confirmation",
        family_size=4,
    )

    month_table = monthly.assign(
        mae=(
            monthly["absolute_error"]
            / monthly["count"]
        )
    ).pivot(
        index=["month", "horizon"],
        columns="model",
        values="mae",
    ).reset_index()

    month_table.columns.name = None
    month_table = month_table.rename(
        columns={
            "month": "Month",
            "horizon": "Horizon",
        }
    )

    r.tg.write_csv(
        month_table,
        ctx["reports"] / "monthly_mae.csv",
    )

    metrics = totals.rename(columns={
        "horizon": "Horizon",
        "model": "Model",
        "count": "Forecasts",
        "mae": "MAE (pips)",
        "rmse": "RMSE (pips)",
    })[[
        "Horizon",
        "Model",
        "Forecasts",
        "MAE (pips)",
        "RMSE (pips)",
    ]]

    check(ctx, built=True)

    return (
        metrics,
        month_table,
        primary,
        sensitivity,
        pd.DataFrame(examples),
    )

def finish(ctx):
    check(ctx, built=True)
    failures = {}

    for name in (
        "feature_checks.csv",
        "evaluation_checks.csv",
    ):
        frame = pd.read_csv(
            ctx["reports"] / name
        )
        failures[name] = int(
            frame.drop(
                columns="month"
            ).to_numpy().sum()
        )

    evidence = r.tg.fx.load_json(
        ctx["reports"]
        / "synthetic_checks.json"
    )

    assert evidence["passed"]
    assert (
        evidence["fingerprint"]
        == ctx["fingerprint"]
    )
    assert all(
        v == 0 for v in failures.values()
    ), failures

    report = {
        "notebook": (
            "06_gradient_boosted_regression.ipynb"
        ),
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "reference_run": g.REFERENCE_RUN,
        "candidate_features": COLUMNS,
        "horizons": ["30s", "60s"],
        "training": "all of 2018",
        "early_stopping": (
            "January-March 2019"
        ),
        "fresh_development_comparison": (
            "April-June 2020"
        ),
        "primary_metric": "MAE",
        "secondary_metric": "RMSE",
        "objective": "regression_l1",
        "comparisons": 4,
        "multiplicity": (
            ctx["config"]["multiplicity"]
        ),
        "nonzero_machine_checks": sum(
            failures.values()
        ),
        "synthetic_checks_passed": True,
        "forecasts_exported_without_future_label_filter": True,
        "final_2026_period_accessed": False,
        "model_promoted": False,
        "human_review": "pending",
        "uncertainty": (
            ctx["config"]["uncertainty"]
        ),
    }

    r.tg.write_json(
        ctx["reports"]
        / "execution_summary.json",
        report,
    )

    text = (
        f"\n- Run: {ctx['run_id']}; "
        f"fingerprint {ctx['fingerprint']}.\n"
        "- Raymond Fendi requested a two-model "
        "variation using only bid size, ask size "
        "and spread following the reported "
        "group-removal results.\n"
        "- This is a requested joint-removal "
        "experiment; the omitted groups have "
        "not been established as irrelevant.\n"
        "- Training remains all of 2018; stopping "
        "remains January-March 2019. Parameters, "
        "MAE objective and valid training rows "
        "remain fixed.\n"
        "- April-June 2020 is the fresh development "
        "comparison; Q4 2019 and Q1 2020 informed "
        "this recipe and are already used "
        "development evidence. 2026 remains reserved.\n"
        "- Compare both horizons with frozen "
        "21-input models and zero change; paired "
        "five-active-date bootstrap, one-date "
        "sensitivity, Holm family of four.\n"
        "- MAE is primary and RMSE secondary. "
        "Missing inputs remain missing. Forecasts "
        "retain availability and information cutoffs.\n"
        "- Execution checks passed. Results await "
        "human interpretation; no model was promoted "
        "and no upstream output was replaced.\n"
    )

    for path in (
        ctx["root"] / "CONTRIBUTIONS.md",
        ctx["root"] / "docs" / "design_decisions.md",
        ctx["root"] / "docs" / "project_state.md",
    ):
        r.util.write_section(
            path,
            f"notebook06_{ctx['run_id']}",
            text,
        )

    return report
