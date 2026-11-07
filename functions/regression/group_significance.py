import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from functions.regression import fx_regression as r

REFERENCE_RUN = "regression_06_short_ma_v1"
REFERENCE_FP = "650d88ee4e8ae4272136e4e9936f039aee522d00a6482c8e0bc93292cf4c4faf"

PHASES = {
    "test": (r.tg.ns("2019-10-01"), r.tg.ns("2020-01-01")),
    "confirmation": (r.tg.ns("2020-01-01"), r.tg.ns("2020-04-01")),
}

def project(table, columns):
    return np.column_stack([
        r.tg.get(table, c, np.nan) for c in columns
    ]).astype(np.float32, copy=False)

def eligible(table, phase, horizon):
    lower, upper = PHASES[phase]
    t = r.tg.get(table, "issue_ns")

    return (
        (t >= lower)
        & (t < upper)
        & r.tg.get(table, f"usable_{horizon}")
        & (
            r.tg.get(table, f"label_available_ns_{horizon}", 0)
            < upper
        )
    )

def check(ctx, built=False):
    r.check_snapshot(ctx)

    assert (
        r.util.sha(__file__)
        == ctx["config"]["test_helper_sha"]
    )

    r.check_features(ctx["training"])
    r.check_features(ctx["reference_features"])

    for item in ctx["config"]["reference_artifacts"]:
        assert (
            r.util.sha(ctx["root"] / item["path"])
            == item["sha"]
        ), item["path"]

    if built:
        r.check_features(ctx)

def load_reference(ctx):
    result = {}

    for h in ("30s", "60s"):
        path = ctx["reference_models"] / f"{h}_metadata.json"
        meta = r.tg.fx.load_json(path)

        model_path = (
            ctx["reference_models"] / f"{h}_lightgbm.txt"
        )

        assert meta["fingerprint"] == REFERENCE_FP
        assert meta["features"] == ctx["config"]["model_features"]
        assert r.util.sha(model_path) == meta["model_sha"]

        assert (
            meta["training_cutoff_ns"]
            < meta["model_available_ns"]
            <= PHASES["test"][0]
        )

        model = lgb.Booster(
            model_str=model_path.read_text(encoding="utf-8")
        )
        assert model.feature_name() == meta["features"]

        result[h] = {
            "model": model,
            "meta": meta,
            "kind": "boost",
        }

    return result

def predict(bundle, table):
    if bundle["kind"] == "zero":
        return np.zeros(table.num_rows)

    x = project(table, bundle["meta"]["features"])
    result = bundle["model"].predict(x)

    assert np.isfinite(result).all()
    return result

def memory_check(ctx):
    reference = load_reference(ctx)
    rows = []

    for h, bundle in reference.items():
        meta = bundle["meta"]
        n = meta["training_rows"] + meta["early_stopping_rows"]

        estimate = (
            6 * n * (4 * len(r.FEATURES) + 8) / 1024**3
        )

        rows.append({
            "Horizon": h,
            "Estimated working GiB": estimate,
            "Budget GiB": ctx["working_gb"],
        })

    frame = pd.DataFrame(rows)

    assert frame["Estimated working GiB"].le(
        ctx["working_gb"]
    ).all(), "Review the memory budget before fitting."

    r.tg.write_csv(
        frame, ctx["reports"] / "memory_estimate.csv"
    )
    return frame

def load_training(ctx, horizon):
    x, y, fit_info = r.load_role(
        ctx["training"], "train", horizon
    )
    sx, sy, stop_info = r.load_role(
        ctx["training"], "early_stopping", horizon
    )
    return x, y, fit_info, sx, sy, stop_info

def prepare(
    root, run_id, memory="2GB", threads=2, working_gb=3.0,
    resamples=9999, alpha=0.05, block_days=5,
    sensitivity_days=1, seed=1729,
):
    root = Path(root).resolve()

    assert run_id and Path(run_id).name == run_id
    assert run_id not in (REFERENCE_RUN, "regression_06_v1")
    assert 0 < alpha < 1 and resamples >= 999
    assert block_days > 0 and sensitivity_days > 0

    ref_features = root / "outputs" / "features" / REFERENCE_RUN
    ref_models = root / "outputs" / "models" / REFERENCE_RUN
    ref_reports = (
        root / "reports" / "tables" / "regression" / REFERENCE_RUN
    )

    ref_cfg_path = ref_features / "configuration.json"
    ref_cfg = r.tg.fx.load_json(ref_cfg_path)
    summary = r.tg.fx.load_json(
        ref_reports / "execution_summary.json"
    )

    assert summary["fingerprint"] == REFERENCE_FP
    assert summary["nonzero_machine_checks"] == 0
    assert len(ref_cfg["model_features"]) == 21
    assert ref_cfg["params"]["objective"] == "regression_l1"

    original = (
        root / "outputs" / "features" / "regression_06_v1"
    )
    original_cfg_path = original / "configuration.json"

    assert (
        r.util.sha(original_cfg_path)
        == ref_cfg["parent_config_sha"]
    )

    original_cfg = r.tg.fx.load_json(original_cfg_path)

    assert {
        k: list(v) for k, v in r.PERIODS.items()
    } == original_cfg["periods_ns"]
    assert r.GROUPS == original_cfg["features"]

    training = {
        "root": root,
        "feature_dir": original,
        "reports": (
            root / "reports" / "tables" / "regression"
            / "regression_06_v1"
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
    r.check_snapshot(training)
    r.check_features(training)

    artifacts = copy.deepcopy(ref_cfg["parent_artifacts"])

    for path in [
        ref_cfg_path,
        original_cfg_path,
        *[
            ref_models / f"{h}_{suffix}"
            for h in ("30s", "60s")
            for suffix in ("metadata.json", "lightgbm.txt")
        ],
        *[
            ref_features / "_checkpoints" / f"{m}.json"
            for m in ("2019-07", "2019-08", "2019-09")
        ],
    ]:
        artifacts.append({
            "path": path.relative_to(root).as_posix(),
            "sha": r.util.sha(path),
        })

    assert len({
        p["path"] for p in artifacts
    }) == len(artifacts)

    months = [
        str(m)
        for m in pd.period_range(
            "2019-10", "2020-03", freq="M"
        )
    ]

    sample = root / "data" / "processed" / r.SAMPLE_RUN
    target = root / "data" / "processed" / r.TARGET_RUN
    entropy = (
        root / "outputs" / "features" / "entropy"
        / r.ENTROPY_RUN
    )

    raw = {
        p["year_month"]: p
        for p in r.tg.fx.load_json(
            sample / "configuration.json"
        )["inputs"]
    }

    additions = []

    for month in months:
        cp = sample / "_checkpoints" / f"{month}.json"
        saved = r.tg.fx.load_json(cp)

        assert saved["fingerprint"] == r.EXPECTED["sampling"]
        assert all(v == 0 for v in saved["checks"].values())

        for kind in ("sampled_10s", "candles_3min"):
            additions.append((
                kind,
                sample / kind / f"{month}.parquet",
                saved["outputs"][kind],
                cp,
            ))

        cp = (
            root / "data" / "processed" / "etl_01_v3"
            / "_checkpoints" / f"observed_{month}.json"
        )

        assert (
            r.util.sha(cp)
            == raw[month]["checkpoint_sha256"]
        )

        additions.append((
            "observed",
            root / raw[month]["path"],
            raw[month],
            cp,
        ))

    for day in pd.date_range(
        "2019-10-01", "2020-03-31", freq="D"
    ):
        date = day.strftime("%Y-%m-%d")

        for kind, folder in (
            ("targets", target),
            ("entropy", entropy),
        ):
            cp = folder / "_checkpoints" / f"{date}.json"
            saved = r.tg.fx.load_json(cp)

            assert saved["fingerprint"] == r.EXPECTED[kind]
            assert all(
                v == 0 for v in saved["checks"].values()
            )

            additions.append((
                kind,
                folder / f"{date}.parquet",
                saved["output"],
                cp,
            ))

    inputs = copy.deepcopy(ref_cfg["inputs"])

    for kind, path, expected, cp in additions:
        info = r.util.file_info(path)

        assert all(
            info[k] == expected[k] for k in info
        ), str(path)

        inputs.append({
            "kind": kind,
            "path": path.relative_to(root).as_posix(),
            **info,
            "checkpoint": cp.relative_to(root).as_posix(),
            "checkpoint_sha": r.util.sha(cp),
        })

    assert len({
        (v["kind"], v["path"]) for v in inputs
    }) == len(inputs)

    groups = ref_cfg["model_feature_groups"]

    assert len(groups) == 6
    assert [
        c for columns in groups.values() for c in columns
    ] == ref_cfg["model_features"]

    cases = []

    for h in ("30s", "60s"):
        for i, (group, columns) in enumerate(groups.items()):
            keep = [
                c for c in ref_cfg["model_features"]
                if c not in columns
            ]
            cases.append({
                "id": f"omit_{h}_{i}",
                "horizon": h,
                "group": group,
                "features": keep,
            })

    config = copy.deepcopy(ref_cfg)

    config.update({
        "definition": "group_significance_v1",
        "run_id": run_id,
        "inputs": inputs,
        "reference_run": REFERENCE_RUN,
        "parent_run_id": REFERENCE_RUN,
        "parent_fingerprint": REFERENCE_FP,
        "parent_config_sha": r.util.sha(ref_cfg_path),
        "parent_artifacts": artifacts,
        "model_available_ns": PHASES["test"][0],
        "architecture_information_cutoff_ns": (
            PHASES["test"][0] - 1
        ),
        "reference_fingerprint": REFERENCE_FP,
        "reference_artifacts": artifacts,
        "test_helper_sha": r.util.sha(__file__),
        "hypotheses": cases,
        "evaluation_periods_ns": PHASES,
        "comparison_scope": (
            "Q4 2019 group tests; Q1 2020 combined confirmation"
        ),
        "resamples": resamples,
        "alpha": alpha,
        "bootstrap_seed": seed,
        "primary_block_days": block_days,
        "sensitivity_block_days": sensitivity_days,
        "bootstrap": (
            "noncircular moving blocks of consecutive "
            "nonempty Athens dates"
        ),
        "interval": (
            "pointwise basic block-bootstrap; not simultaneous"
        ),
        "p_value": (
            "two-sided equal-tail centered-null bootstrap; "
            "finite-resample correction"
        ),
        "multiplicity": (
            "Holm across 12 primary tests; "
            "separate 2-test confirmation family"
        ),
        "selection": (
            "negative delta MAE and primary Holm p <= alpha"
        ),
        "zero_feature_candidate": "zero-change forecast",
        "uncertainty": (
            "conditional on fixed fitted models; "
            "approximate within each quarter"
        ),
    })

    config = json.loads(json.dumps(config))
    fingerprint = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    feature_dir = root / "outputs" / "features" / run_id
    model_dir = root / "outputs" / "models" / run_id
    forecast_dir = root / "outputs" / "forecasts" / run_id
    reports = (
        root / "reports" / "tables" / "regression" / run_id
    )

    for path in (
        feature_dir,
        feature_dir / "_checkpoints",
        model_dir,
        reports,
        forecast_dir / "reference",
        forecast_dir / "selected",
    ):
        path.mkdir(parents=True, exist_ok=True)

    path = feature_dir / "configuration.json"

    if path.exists():
        old = r.tg.fx.load_json(path)
        changed = [
            k for k in config if old.get(k) != config[k]
        ]
        assert not changed, (
            f"Configuration changed {changed}; "
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
        "fingerprint": fingerprint,
        "run_id": run_id,
        "months": months,
        "training": training,
        "reference_models": ref_models,
        "reference_features": {
            "feature_dir": ref_features,
            "months": ["2019-07", "2019-08", "2019-09"],
            "fingerprint": REFERENCE_FP,
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
    assert evidence["fingerprint"] == ctx["fingerprint"]

    records = []

    with duckdb.connect(config={
        "memory_limit": ctx["memory"],
        "threads": str(ctx["threads"]),
    }) as con:
        for month in ctx["months"]:
            path = ctx["feature_dir"] / f"{month}.parquet"
            cp = (
                ctx["feature_dir"] / "_checkpoints"
                / f"{month}.json"
            )

            if cp.exists():
                saved = r.tg.fx.load_json(cp)

                assert (
                    saved["fingerprint"] == ctx["fingerprint"]
                )
                assert (
                    r.util.file_info(path) == saved["output"]
                )
                assert all(
                    v == 0 for v in saved["checks"].values()
                )
            else:
                table, checks, _, _ = r.make_month(
                    ctx, month, con
                )

                counts = {
                    "month": month,
                    "rows": table.num_rows,
                }

                for phase in PHASES:
                    for h in ("30s", "60s"):
                        counts[f"{phase}_{h}"] = int(
                            eligible(table, phase, h).sum()
                        )

                r.tg.write_parquet(table, path)

                saved = {
                    "fingerprint": ctx["fingerprint"],
                    "checks": checks,
                    "counts": counts,
                    "output": r.util.file_info(path),
                }

                r.tg.write_json(cp, saved)
                print(f"{month}: inputs saved.")

            records.append(saved)

    check(ctx, built=True)

    r.tg.write_csv(
        pd.DataFrame([s["counts"] for s in records]),
        ctx["reports"] / "feature_counts.csv",
    )
    r.tg.write_csv(
        pd.DataFrame([
            {
                "month": s["counts"]["month"],
                **s["checks"],
            }
            for s in records
        ]),
        ctx["reports"] / "feature_checks.csv",
    )

    return memory_check(ctx)

def fit_case(ctx, case, references, data=None):
    key = case["id"]
    h = case["horizon"]
    columns = case["features"]

    path = ctx["model_dir"] / f"{key}.txt"
    meta_path = ctx["model_dir"] / f"{key}.json"

    if meta_path.exists():
        meta = r.tg.fx.load_json(meta_path)

        assert meta["fingerprint"] == ctx["fingerprint"]
        assert meta["features"] == columns
        assert meta["model_available_ns"] == case["ready_ns"]
        assert (
            meta["training_cutoff_ns"]
            < meta["model_available_ns"]
        )
        assert (
            meta["selection_report_sha"]
            == case.get("selection_report_sha")
        )

        if meta["kind"] == "zero":
            return {
                "model": None,
                "meta": meta,
                "kind": "zero",
            }

        assert r.util.sha(path) == meta["model_sha"]

        model = lgb.Booster(
            model_str=path.read_text(encoding="utf-8")
        )
        assert model.feature_name() == columns

        return {
            "model": model,
            "meta": meta,
            "kind": meta["kind"],
        }

    started = time.perf_counter()
    reference = references[h]
    ref_meta = reference["meta"]
    new_fit = False

    if not columns:
        model, kind, trees = None, "zero", 0
        fit_info = ref_meta["fit_information_cutoff_ns"]
        stop_info = ref_meta[
            "early_stopping_information_cutoff_ns"
        ]
        n_train, n_stop = 0, 0

    elif columns == ctx["config"]["model_features"]:
        model = reference["model"]
        kind = "reference_copy"
        trees = ref_meta["best_iteration"]
        fit_info = ref_meta["fit_information_cutoff_ns"]
        stop_info = ref_meta[
            "early_stopping_information_cutoff_ns"
        ]
        n_train = ref_meta["training_rows"]
        n_stop = ref_meta["early_stopping_rows"]

    else:
        if data is None:
            data = load_training(ctx, h)

        full_x, y, fit_info, full_stop, stop_y, stop_info = data
        indices = [r.FEATURES.index(c) for c in columns]

        x = np.ascontiguousarray(full_x[:, indices])
        stop_x = np.ascontiguousarray(full_stop[:, indices])

        assert len(y) == ref_meta["training_rows"]
        assert len(stop_y) == ref_meta["early_stopping_rows"]
        assert np.isfinite(x).any(axis=0).all()

        train_data = lgb.Dataset(
            x, label=y, feature_name=columns
        )
        stop_data = lgb.Dataset(
            stop_x,
            label=stop_y,
            reference=train_data,
            feature_name=columns,
        )

        model = lgb.train(
            copy.deepcopy(ctx["config"]["params"]),
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

        assert model.best_iteration >= 1
        trees = int(model.best_iteration)
        kind, new_fit = "boost", True
        n_train, n_stop = len(y), len(stop_y)

        del train_data, stop_data, x

    ready = case["ready_ns"]
    cutoff = max(fit_info, stop_info, ready - 1)
    assert cutoff < ready

    if kind != "zero":
        text = model.model_to_string(num_iteration=trees)
        restored = lgb.Booster(model_str=text)

        if new_fit:
            assert np.allclose(
                model.predict(stop_x[:1000]),
                restored.predict(stop_x[:1000]),
                rtol=0,
                atol=1e-10,
            )

        r.save_text(path, text)
        model = restored

    meta = {
        "fingerprint": ctx["fingerprint"],
        "case_id": key,
        "horizon": h,
        "group": case.get("group"),
        "features": columns,
        "kind": kind,
        "new_fit": new_fit,
        "best_iteration": trees,
        "training_rows": n_train,
        "early_stopping_rows": n_stop,
        "training_period": (
            "Not fitted: zero-change benchmark"
            if kind == "zero"
            else "January-December 2018"
        ),
        "objective": "regression_l1",
        "estimate": "conditional median point forecast",
        "fit_information_cutoff_ns": fit_info,
        "early_stopping_information_cutoff_ns": stop_info,
        "selection_information_cutoff_ns": ready - 1,
        "training_cutoff_ns": cutoff,
        "model_available_ns": ready,
        "model_identifier": f"{ctx['run_id']}_{key}",
        "inference_latency_ns": 0,
        "selection_report_sha": case.get("selection_report_sha"),
        "model_sha": (
            None if kind == "zero" else r.util.sha(path)
        ),
        "fit_minutes": (
            time.perf_counter() - started
        ) / 60,
    }

    r.tg.write_json(meta_path, meta)
    gc.collect()

    return {
        "model": model,
        "meta": meta,
        "kind": kind,
    }

def fit_omissions(ctx):
    check(ctx, built=True)
    memory_check(ctx)

    references = load_reference(ctx)
    result, rows = {}, []

    for h in ("30s", "60s"):
        cases = [
            dict(c, ready_ns=PHASES["test"][0])
            for c in ctx["config"]["hypotheses"]
            if c["horizon"] == h
        ]

        pending = any(
            not (
                ctx["model_dir"] / f"{c['id']}.json"
            ).exists()
            for c in cases
        )

        data = load_training(ctx, h) if pending else None

        for case in cases:
            bundle = fit_case(
                ctx, case, references, data
            )
            result[case["id"]] = bundle
            meta = bundle["meta"]

            rows.append({
                "Group": case["group"],
                "Horizon": h,
                "Inputs": len(meta["features"]),
                "Trees": meta["best_iteration"],
                "Training examples": meta["training_rows"],
            })

            print(f"{h} | omit {case['group']}: ready.")

        del data
        gc.collect()

    check(ctx, built=True)

    report = pd.DataFrame(rows)
    r.tg.write_csv(
        report,
        ctx["reports"] / "omission_fit_summary.csv",
    )
    return result

def score_omissions(ctx, models):
    check(ctx, built=True)
    references = load_reference(ctx)

    daily, monthly, checks = [], [], []

    for month in ("2019-10", "2019-11", "2019-12"):
        with duckdb.connect(config={
            "memory_limit": ctx["memory"],
            "threads": str(ctx["threads"]),
        }) as con:
            table = r.read(
                con,
                [ctx["feature_dir"] / f"{month}.parquet"],
                [
                    *r.BASE,
                    *r.FEATURES,
                    "input_information_max_ns",
                ],
                "issue_ns",
            )

        times = r.tg.get(table, "issue_ns")
        dates = np.asarray(
            pd.to_datetime(times, unit="ns", utc=True)
            .tz_convert(r.TZ)
            .strftime("%Y-%m-%d")
        )

        issue_checks = {
            "month": month,
            "late_inputs": int(np.count_nonzero(
                r.tg.get(table, "input_information_max_ns")
                > times
            )),
        }

        for h in ("30s", "60s"):
            mask = eligible(table, "test", h)
            actual = r.tg.get(
                table, f"target_change_{h}_pips", np.nan
            )[mask]

            ref_pred = predict(references[h], table)
            ref_error = ref_pred[mask] - actual

            assert len(actual) and np.isfinite(actual).all()

            days, inverse = np.unique(
                dates[mask], return_inverse=True
            )
            counts = np.bincount(inverse)

            cases = [
                c for c in ctx["config"]["hypotheses"]
                if c["horizon"] == h
            ]

            for case in cases:
                bundle = models[case["id"]]

                assert np.all(
                    times >= bundle["meta"]["model_available_ns"]
                )

                pred = predict(bundle, table)
                error = pred[mask] - actual

                sums = np.bincount(
                    inverse,
                    weights=(
                        np.abs(error) - np.abs(ref_error)
                    ),
                )

                for day, total, count in zip(
                    days, sums, counts
                ):
                    daily.append({
                        "date": day,
                        "case_id": case["id"],
                        "horizon": h,
                        "group": case["group"],
                        "count": int(count),
                        "difference_sum": float(total),
                    })

                monthly.append({
                    "month": month,
                    "case_id": case["id"],
                    "horizon": h,
                    "group": case["group"],
                    "count": len(actual),
                    "reference_absolute_error": float(
                        np.abs(ref_error).sum()
                    ),
                    "without_absolute_error": float(
                        np.abs(error).sum()
                    ),
                    "reference_squared_error": float(
                        np.square(ref_error).sum()
                    ),
                    "without_squared_error": float(
                        np.square(error).sum()
                    ),
                })

                issue_checks[
                    case["id"] + "_late_model"
                ] = int(np.count_nonzero(
                    times
                    <= bundle["meta"]["training_cutoff_ns"]
                ))

        assert all(
            v == 0 for k, v in issue_checks.items()
            if k != "month"
        ), issue_checks

        checks.append(issue_checks)
        print(f"{month}: all 12 comparisons scored.")

    daily = pd.DataFrame(daily)
    monthly = pd.DataFrame(monthly)

    monthly["delta_mae_pips"] = (
        monthly["without_absolute_error"]
        - monthly["reference_absolute_error"]
    ) / monthly["count"]

    assert monthly.groupby(
        ["month", "horizon"]
    )["count"].nunique().eq(1).all()

    assert daily.groupby(
        ["date", "horizon"]
    )["count"].nunique().eq(1).all()

    for case_id, item in monthly.groupby(
        "case_id", sort=False
    ):
        d = daily.loc[daily["case_id"].eq(case_id)]

        assert int(d["count"].sum()) == int(
            item["count"].sum()
        )
        assert np.isclose(
            d["difference_sum"].sum(),
            (
                item["without_absolute_error"]
                - item["reference_absolute_error"]
            ).sum(),
            rtol=0,
            atol=1e-7,
        )

    r.tg.write_csv(
        daily,
        ctx["reports"] / "paired_daily_test_losses.csv",
    )
    r.tg.write_csv(
        monthly,
        ctx["reports"] / "monthly_omission_scores.csv",
    )
    r.tg.write_csv(
        pd.DataFrame(checks),
        ctx["reports"] / "omission_checks.csv",
    )

    check(ctx, built=True)
    return daily

def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)

    assert np.isfinite(p_values).all()
    assert np.all((p_values >= 0) & (p_values <= 1))

    order = np.argsort(p_values)
    ranked = p_values[order] * (
        len(p_values) - np.arange(len(p_values))
    )
    adjusted = np.minimum(
        1.0, np.maximum.accumulate(ranked)
    )

    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result

def block_test(sums, counts, block, draws, alpha, seed):
    sums = np.asarray(sums, dtype=float)
    counts = np.asarray(counts, dtype=float)

    assert sums.shape == counts.shape and sums.ndim == 2
    assert np.isfinite(sums).all()
    assert np.isfinite(counts).all()
    assert np.all(counts >= 0)

    active = counts.sum(axis=1) > 0
    sums, counts = sums[active], counts[active]
    n, k = sums.shape

    assert n >= 2 * block, (
        "Too few active dates for the chosen block length."
    )
    assert np.all(counts.sum(axis=0) > 0)
    assert np.all(sums[counts == 0] == 0)

    # Preserve forecast weighting: sum of losses / sum of rows.
    observed = sums.sum(axis=0) / counts.sum(axis=0)
    centered = sums - counts * observed

    null = np.empty((draws, k))
    rng = np.random.default_rng(seed)
    done, rejected = 0, 0
    blocks = (n + block - 1) // block

    while done < draws:
        batch = min(256, draws - done)

        starts = rng.integers(
            0, n - block + 1,
            size=(batch, blocks),
        )
        index = (
            starts[:, :, None] + np.arange(block)
        ).reshape(batch, -1)[:, :n]

        denominator = counts[index].sum(axis=1)
        good = np.all(denominator > 0, axis=1)
        rejected += int((~good).sum())

        if not good.any():
            assert rejected < 100 * draws, (
                "Bootstrap cohorts need review."
            )
            continue

        index = index[good]
        denominator = denominator[good]

        values = (
            centered[index].sum(axis=1)
            / denominator
        )

        take = min(len(values), draws - done)
        null[done:done + take] = values[:take]
        done += take

    assert rejected <= max(1, draws // 100), (
        "Too many zero-count bootstrap draws; review cohorts."
    )

    # Basic intervals invert the centered error distribution.
    lower = observed - np.quantile(
        null, 1 - alpha / 2, axis=0
    )
    upper = observed - np.quantile(
        null, alpha / 2, axis=0
    )

    lower_tail = (
        1 + (null <= observed).sum(axis=0)
    ) / (draws + 1)
    upper_tail = (
        1 + (null >= observed).sum(axis=0)
    ) / (draws + 1)

    p = np.minimum(
        1.0, 2 * np.minimum(lower_tail, upper_tail)
    )

    return {
        "delta": observed,
        "lower": lower,
        "upper": upper,
        "p": p,
        "active_dates": n,
        "rejected_draws": rejected,
    }

def infer(ctx, daily, descriptors, stem, family_size):
    ids = [case["id"] for case in descriptors]

    assert len(ids) == family_size
    assert len(set(ids)) == family_size

    sums = daily.pivot(
        index="date",
        columns="case_id",
        values="difference_sum",
    )
    counts = daily.pivot(
        index="date",
        columns="case_id",
        values="count",
    )

    dates = sorted(set(sums.index) | set(counts.index))

    # Zero here means no scored forecasts on that date.
    # It does not fill any market data or missing feature.
    sums = sums.reindex(
        index=dates, columns=ids
    ).fillna(0).to_numpy(dtype=float)

    counts = counts.reindex(
        index=dates, columns=ids
    ).fillna(0).to_numpy(dtype=float)

    results = {}

    for name, block in (
        ("primary", ctx["config"]["primary_block_days"]),
        (
            "sensitivity",
            ctx["config"]["sensitivity_block_days"],
        ),
    ):
        value = block_test(
            sums,
            counts,
            block,
            ctx["config"]["resamples"],
            ctx["config"]["alpha"],
            ctx["config"]["bootstrap_seed"] + block,
        )

        adjusted = holm(value["p"])
        rows = []

        for i, case in enumerate(descriptors):
            delta = float(value["delta"][i])
            significant = bool(
                adjusted[i] <= ctx["config"]["alpha"]
            )

            if significant:
                evidence = (
                    "Removal improves MAE"
                    if delta < 0 else "Group helps"
                )
            else:
                evidence = "Inconclusive"

            if stem == "confirmation":
                if significant:
                    evidence = (
                        "Candidate improves MAE"
                        if delta < 0
                        else "Candidate worsens MAE"
                    )
                else:
                    evidence = "Inconclusive"

            rows.append({
                "case_id": case["id"],
                "horizon": case["horizon"],
                "group": case.get(
                    "group", "Selected composite"
                ),
                "delta_mae_pips": delta,
                "ci_low": float(value["lower"][i]),
                "ci_high": float(value["upper"][i]),
                "p_raw": float(value["p"][i]),
                "p_holm": float(adjusted[i]),
                "evidence": evidence,
                "drop_candidate": bool(
                    stem != "confirmation"
                    and delta < 0
                    and significant
                ),
                "block_days": block,
                "active_dates": value["active_dates"],
                "scored_forecasts": int(
                    counts[:, i].sum()
                ),
                "rejected_draws": value["rejected_draws"],
                "interval_scope": (
                    "pointwise; not simultaneous"
                ),
            })

        frame = pd.DataFrame(rows)

        r.tg.write_csv(
            frame,
            ctx["reports"]
            / f"{stem}_{name}_inference.csv",
        )
        results[name] = frame

    return results["primary"], results["sensitivity"]

def inference_display(frame):
    return pd.DataFrame({
        "Group": frame["group"],
        "Horizon": frame["horizon"],
        "Delta MAE (pips)": frame[
            "delta_mae_pips"
        ].map(lambda v: f"{v:+.7f}"),
        "95% interval": [
            f"[{a:+.7f}, {b:+.7f}]"
            for a, b in zip(
                frame["ci_low"], frame["ci_high"]
            )
        ],
        "Adjusted p": frame["p_holm"].round(4),
        "Evidence": frame["evidence"],
    })

def select_and_fit(ctx, inference):
    check(ctx, built=True)
    memory_check(ctx)
    references = load_reference(ctx)

    report_path = (
        ctx["reports"] / "omissions_primary_inference.csv"
    )

    assert inference["case_id"].tolist() == [
        c["id"] for c in ctx["config"]["hypotheses"]
    ]

    saved = pd.read_csv(report_path)

    assert np.array_equal(
        saved["drop_candidate"].to_numpy(),
        inference["drop_candidate"].to_numpy(),
    )
    assert np.allclose(
        saved["delta_mae_pips"],
        inference["delta_mae_pips"],
        rtol=0,
        atol=1e-12,
    )

    report_sha = r.util.sha(report_path)
    selected, rows, specifications = {}, [], []

    for h in ("30s", "60s"):
        chosen = inference.loc[
            inference["horizon"].eq(h)
            & inference["drop_candidate"],
            "group",
        ].tolist()

        removed = {
            c
            for group in chosen
            for c in ctx["config"]["model_feature_groups"][group]
        }
        columns = [
            c for c in ctx["config"]["model_features"]
            if c not in removed
        ]

        case = {
            "id": f"selected_{h}",
            "horizon": h,
            "features": columns,
            "ready_ns": PHASES["confirmation"][0],
            "selection_report_sha": report_sha,
        }

        selected[h] = fit_case(
            ctx, case, references
        )

        specifications.append({
            **case,
            "removed_groups": chosen,
        })
        rows.append({
            "Horizon": h,
            "Removal candidate groups": (
                ", ".join(chosen) or "None"
            ),
            "Remaining inputs": len(columns),
            "Candidate type": selected[h]["kind"],
        })

    r.tg.write_json(
        ctx["reports"] / "selected_specs.json",
        specifications,
    )

    check(ctx, built=True)
    return selected, pd.DataFrame(rows)

def synthetic_checks():
    assert np.allclose(
        holm([0.01, 0.04, 0.03]),
        [0.03, 0.06, 0.06],
    )
    checks = {"holm_known_example": True}

    n = np.arange(1, 21, dtype=float)
    counts = np.column_stack([n, n])
    sums = -0.1 * counts

    result = block_test(
        sums, counts,
        block=5, draws=199, alpha=0.05, seed=17,
    )

    assert np.allclose(result["delta"], [-0.1, -0.1])
    assert np.allclose(result["lower"], [-0.1, -0.1])
    assert np.allclose(result["upper"], [-0.1, -0.1])
    assert result["p"][0] == result["p"][1]

    checks[
        "paired_common_draws_and_constant_difference"
    ] = True

    sums = np.column_stack([
        np.arange(20, dtype=float),
        np.arange(20, dtype=float),
    ])

    result = block_test(
        sums, counts, 1, 199, 0.05, 17
    )

    assert abs(
        result["delta"][0] - 190 / 210
    ) < 1e-12

    checks["observation_weighted_daily_ratio"] = True

    result = block_test(
        np.zeros_like(counts),
        counts, 5, 199, 0.05, 17,
    )

    assert np.array_equal(
        result["p"], [1.0, 1.0]
    )
    assert np.all(result["lower"] == 0)
    assert np.all(result["upper"] == 0)

    checks[
        "identical_forecasts_are_not_significant"
    ] = True

    cut = PHASES["test"][1]
    table = pa.table({
        "issue_ns": [cut - r.STEP, cut - r.STEP],
        "usable_30s": [True, True],
        "label_available_ns_30s": [cut, cut - 1],
    })

    assert np.array_equal(
        eligible(table, "test", "30s"),
        [False, True],
    )

    checks["quarter_information_cutoff"] = True

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }

def forecast_table(table, bundles, predictions):
    times = r.tg.get(table, "issue_ns")

    assert np.all(
        r.tg.get(table, "input_information_max_ns")
        <= times
    )

    reference = r.tg.get(
        table, "reference_mid", np.nan
    )

    data = {
        c: table[c]
        for c in (
            "issue_ns",
            "reference_quote_ns",
            "reference_mid",
            "reference_age_ns",
        )
    }

    for h in ("30s", "60s"):
        meta = bundles[h]["meta"]

        assert (
            meta["training_cutoff_ns"]
            < meta["model_available_ns"]
        )
        assert np.all(
            times >= meta["model_available_ns"]
        )

        data.update({
            f"predicted_change_{h}_pips": pa.array(
                predictions[h]
            ),
            f"expected_eurusd_rate_{h}": pa.array(
                r.tg.predicted_mid(
                    reference, predictions[h]
                )
            ),
            f"forecast_available_ns_{h}": pa.array(
                times + meta["inference_latency_ns"]
            ),
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

    result = pa.table(data)

    assert result.num_columns == 14
    assert result.num_rows == table.num_rows
    return result

def confirm(ctx, selected):
    check(ctx, built=True)
    references = load_reference(ctx)

    specs = r.tg.fx.load_json(
        ctx["reports"] / "selected_specs.json"
    )
    assert {
        v["horizon"] for v in specs
    } == {"30s", "60s"}

    selection_sha = r.util.sha(
        ctx["reports"] / "omissions_primary_inference.csv"
    )

    assert all(
        v["selection_report_sha"] == selection_sha
        for v in specs
    )

    daily, monthly, checks, examples = [], [], [], []

    columns = [
        *r.BASE,
        *r.FEATURES,
        "input_information_max_ns",
    ]

    for month in ("2020-01", "2020-02", "2020-03"):
        with duckdb.connect(config={
            "memory_limit": ctx["memory"],
            "threads": str(ctx["threads"]),
        }) as con:
            table = r.read(
                con,
                [ctx["feature_dir"] / f"{month}.parquet"],
                columns,
                "issue_ns",
            )

        times = r.tg.get(table, "issue_ns")
        dates = np.asarray(
            pd.to_datetime(times, unit="ns", utc=True)
            .tz_convert(r.TZ)
            .strftime("%Y-%m-%d")
        )

        month_checks = {
            "month": month,
            "late_inputs": int(np.count_nonzero(
                r.tg.get(table, "input_information_max_ns")
                > times
            )),
        }

        saved_predictions = {
            "reference": {},
            "selected": {},
        }

        for h in ("30s", "60s"):
            ref = references[h]
            candidate = selected[h]

            assert (
                candidate["meta"]["selection_report_sha"]
                == selection_sha
            )
            assert (
                candidate["meta"]["case_id"]
                == f"selected_{h}"
            )

            predictions = {
                "Reference 21": predict(ref, table),
                "Selected candidate": predict(
                    candidate, table
                ),
                "Zero change": np.zeros(table.num_rows),
            }

            if candidate["kind"] == "reference_copy":
                assert np.allclose(
                    predictions["Reference 21"],
                    predictions["Selected candidate"],
                    rtol=0,
                    atol=1e-10,
                )

                # These are the same fitted model.
                # Avoid treating serialization rounding as a change.
                predictions["Selected candidate"] = (
                    predictions["Reference 21"].copy()
                )

            saved_predictions["reference"][h] = (
                predictions["Reference 21"]
            )
            saved_predictions["selected"][h] = (
                predictions["Selected candidate"]
            )

            for label, bundle in (
                ("reference", ref),
                ("selected", candidate),
            ):
                month_checks[
                    f"{label}_{h}_late_model"
                ] = int(np.count_nonzero(
                    times
                    <= bundle["meta"]["training_cutoff_ns"]
                ))

            mask = eligible(
                table, "confirmation", h
            )
            actual = r.tg.get(
                table, f"target_change_{h}_pips", np.nan
            )[mask]

            assert len(actual) and np.isfinite(actual).all()

            errors = {
                key: value[mask] - actual
                for key, value in predictions.items()
            }

            days, inverse = np.unique(
                dates[mask], return_inverse=True
            )
            counts = np.bincount(inverse)

            differences = (
                np.abs(errors["Selected candidate"])
                - np.abs(errors["Reference 21"])
            )
            sums = np.bincount(
                inverse, weights=differences
            )

            daily.extend({
                "date": day,
                "case_id": f"selected_{h}",
                "horizon": h,
                "count": int(n),
                "difference_sum": float(value),
            } for day, n, value in zip(
                days, counts, sums
            ))

            for model, error in errors.items():
                monthly.append({
                    "month": month,
                    "horizon": h,
                    "model": model,
                    "count": len(actual),
                    "absolute_error": float(
                        np.abs(error).sum()
                    ),
                    "squared_error": float(
                        np.square(error).sum()
                    ),
                })

            if month == "2020-01":
                lower = r.tg.ns(
                    "2020-01-06T10:00:00"
                )
                chosen = np.flatnonzero(
                    mask
                    & (times >= lower)
                    & (times < lower + 40 * r.NS)
                )

                for i in chosen:
                    examples.append({
                        "Athens time": pd.Timestamp(
                            times[i], unit="ns", tz="UTC"
                        ).tz_convert(r.TZ).strftime(
                            "%d %b %Y %H:%M:%S"
                        ),
                        "Horizon": h,
                        "Current mid": r.tg.get(
                            table, "reference_mid"
                        )[i],
                        "Reference forecast": (
                            r.tg.get(
                                table, "reference_mid"
                            )[i]
                            + predictions["Reference 21"][i]
                            * r.PIP
                        ),
                        "Candidate forecast": (
                            r.tg.get(
                                table, "reference_mid"
                            )[i]
                            + predictions[
                                "Selected candidate"
                            ][i] * r.PIP
                        ),
                        "Observed future mid": r.tg.get(
                            table,
                            f"target_future_mid_{h}",
                            np.nan,
                        )[i],
                        "Interpolated outcome": bool(
                            r.tg.get(
                                table,
                                f"endpoint_is_interpolated_{h}",
                                False,
                            )[i]
                        ),
                    })

        # Export every issue-grid row.
        # Future-label validity never selects exported forecasts.
        for label, bundles in (
            ("reference", references),
            ("selected", selected),
        ):
            forecasts = forecast_table(
                table, bundles, saved_predictions[label]
            )
            r.tg.write_parquet(
                forecasts,
                ctx["forecast_dir"] / label
                / f"{month}.parquet",
            )

        assert all(
            v == 0 for k, v in month_checks.items()
            if k != "month"
        ), month_checks

        checks.append(month_checks)
        print(
            f"{month}: confirmation scored; "
            "all-grid forecasts saved."
        )

    monthly = pd.DataFrame(monthly)
    daily = pd.DataFrame(daily)

    assert monthly.groupby(
        ["month", "horizon"]
    )["count"].nunique().eq(1).all()

    for h in ("30s", "60s"):
        d = daily.loc[daily["horizon"].eq(h)]

        a = monthly.loc[
            monthly["horizon"].eq(h)
        ].groupby("model")[
            ["absolute_error", "count"]
        ].sum()

        assert int(d["count"].sum()) == int(
            a.loc["Reference 21", "count"]
        )
        assert np.isclose(
            d["difference_sum"].sum(),
            a.loc["Selected candidate", "absolute_error"]
            - a.loc["Reference 21", "absolute_error"],
            rtol=0,
            atol=1e-7,
        )

    r.tg.write_csv(
        daily,
        ctx["reports"]
        / "paired_daily_confirmation_losses.csv",
    )
    r.tg.write_csv(
        monthly,
        ctx["reports"]
        / "monthly_confirmation_scores.csv",
    )
    r.tg.write_csv(
        pd.DataFrame(checks),
        ctx["reports"] / "confirmation_checks.csv",
    )

    totals = monthly.groupby(
        ["horizon", "model"], sort=False
    )[[
        "count",
        "absolute_error",
        "squared_error",
    ]].sum().reset_index()

    totals["mae"] = (
        totals["absolute_error"] / totals["count"]
    )
    totals["rmse"] = np.sqrt(
        totals["squared_error"] / totals["count"]
    )

    r.tg.write_csv(
        totals,
        ctx["reports"] / "confirmation_metrics.csv",
    )

    example_frame = pd.DataFrame(examples)

    r.tg.write_csv(
        example_frame,
        ctx["reports"]
        / "confirmation_price_examples.csv",
    )

    descriptors = [
        {
            "id": f"selected_{h}",
            "horizon": h,
        }
        for h in ("30s", "60s")
    ]

    primary, sensitivity = infer(
        ctx, daily, descriptors, "confirmation", 2
    )

    check(ctx, built=True)

    lean = totals.rename(columns={
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

    return (
        lean, primary, sensitivity, example_frame
    )

def finish(ctx):
    check(ctx, built=True)
    reports = ctx["reports"]
    checks = {}

    for filename, index in (
        ("feature_checks.csv", "month"),
        ("omission_checks.csv", "month"),
        ("confirmation_checks.csv", "month"),
    ):
        frame = pd.read_csv(reports / filename)
        checks[filename] = int(
            frame.drop(columns=index).to_numpy().sum()
        )

    evidence = r.tg.fx.load_json(
        reports / "synthetic_checks.json"
    )

    assert evidence["passed"]
    assert evidence["fingerprint"] == ctx["fingerprint"]
    assert all(
        v == 0 for v in checks.values()
    ), checks

    fits = pd.read_csv(
        reports / "omission_fit_summary.csv"
    )
    assert len(fits) == 12

    selected = r.tg.fx.load_json(
        reports / "selected_specs.json"
    )

    summary = {
        "notebook": "06_gradient_boosted_regression.ipynb",
        "experiment": (
            "group removal significance and combined confirmation"
        ),
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "reference_run_id": REFERENCE_RUN,
        "reference_fingerprint": REFERENCE_FP,
        "training": "all of 2018",
        "early_stopping": "January-March 2019",
        "group_test_period": "October-December 2019",
        "confirmation_period": "January-March 2020",
        "omission_comparisons": 12,
        "selected_removals": {
            v["horizon"]: v["removed_groups"]
            for v in selected
        },
        "primary_loss": "MAE; regression_l1",
        "secondary_metric": "RMSE",
        "bootstrap_draws": ctx["config"]["resamples"],
        "primary_block_active_dates": (
            ctx["config"]["primary_block_days"]
        ),
        "sensitivity_block_active_dates": (
            ctx["config"]["sensitivity_block_days"]
        ),
        "multiplicity": ctx["config"]["multiplicity"],
        "nonzero_machine_checks": sum(checks.values()),
        "synthetic_checks_passed": True,
        "confirmation_forecasts_include_unscorable_rows": True,
        "final_2026_period_accessed": False,
        "human_review": "pending",
        "model_promoted": False,
        "uncertainty": ctx["config"]["uncertainty"],
        "source_stability_check": (
            "file metadata and checkpoint/artifact hashes; "
            "no bulk content hashes"
        ),
    }

    r.tg.write_json(
        reports / "execution_summary.json", summary
    )

    marker = (
        f"notebook06_group_tests_{ctx['run_id']}"
    )

    text = (
        f"\n- Experiment: {ctx['run_id']}; "
        f"fingerprint {ctx['fingerprint']}.\n"
        "- Raymond Fendi requested input-removal tests "
        "to investigate MAE improvement.\n"
        "- Agreed: 21-input reference, six grouped omissions "
        "at each horizon; 2018 training and January-March "
        "2019 stopping remain fixed.\n"
        "- Q4 2019 selects removals; Q1 2020 checks the "
        "combined candidate. 2026 remains reserved.\n"
        "- MAE is primary, RMSE secondary. "
        "Undefined inputs remain missing.\n"
        "- Five-active-date paired bootstrap is primary; "
        "one-date sensitivity; Holm correction across 12 "
        "tests and a separate two-test confirmation family.\n"
        "- Intervals are pointwise and approximate, "
        "conditional on fixed models and the evaluation "
        "quarter. Inconclusive does not mean irrelevant.\n"
        "- Selected groups: "
        + json.dumps(summary["selected_removals"])
        + ".\n"
        "- Execution checks passed. Human interpretation "
        "and any model promotion remain pending; no upstream "
        "definitions or models were replaced.\n"
    )

    for path in (
        ctx["root"] / "CONTRIBUTIONS.md",
        ctx["root"] / "docs" / "design_decisions.md",
        ctx["root"] / "docs" / "project_state.md",
    ):
        r.util.write_section(path, marker, text)

    return summary
