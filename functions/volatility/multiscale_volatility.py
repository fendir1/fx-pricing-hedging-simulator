
import json
import hashlib
import importlib.metadata
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

from functions.etl import clean_quotes as fx

NS = 1_000_000_000
STEP = 10 * NS
PIP = 0.0001
WINDOWS = {"30s": 3, "1m": 6, "5m": 30, "15m": 90}
MAX_N = 90

SOURCE_RUN = "sampling_02_v1"
SOURCE_FINGERPRINT = (
    "ad5235dc14f854d9f26ef04dd2c1fdf846faf05cff43eb9f84afccc229628e54"
)
SOURCE_COLUMNS = [
    "sample_ns", "available_ns", "mid", "has_price",
    "is_interpolated", "in_long_gap",
    "project_excluded", "project_eligible",
]

# These tolerances validate floating-point arithmetic.
# They do not round data or define market regimes.
VALUE_TOL = 1e-8
SHARE_TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {
        "python": sys.version.split()[0],
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "duckdb", "matplotlib")
        },
    }


def file_info(path):
    path = Path(path)
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": pq.ParquetFile(path).metadata.num_rows,
    }


def ddl(con, sql):
    con.execute(sql)
    con.complete()


def frame(con, sql):
    return con.sql(sql).to_arrow_table().to_pandas(
        types_mapper=pd.ArrowDtype
    )


def display_times(data, columns):
    out = data.copy()
    for column in columns:
        if column in out:
            out[column.replace("_ns", "_athens")] = (
                pd.to_datetime(
                    out[column].astype("Int64"), unit="ns", utc=True
                ).dt.tz_convert("Europe/Athens")
            )
    return out


def write_section(path, marker, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = start + "\n" + body.strip() + "\n" + end

    if start in text:
        assert end in text, f"Unfinished documentation marker: {path}"
        left = text.index(start)
        right = text.index(end, left) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text = text.rstrip() + "\n\n" + block

    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_views(con, context_query, lower, upper):
    lower, upper = int(lower), int(upper)
    ddl(con, "CREATE OR REPLACE TEMP VIEW vol_context AS " + context_query)

    source_checks = fx.record(con, f"""
        WITH ordered AS (
            SELECT *,
                lag(sample_ns) OVER (ORDER BY sample_ns) AS previous_ns
            FROM vol_context
        )
        SELECT
            count(*) - count(DISTINCT sample_ns)
                AS duplicate_input_timestamps,
            count(*) FILTER (
                WHERE sample_ns IS NULL
                   OR available_ns IS NULL
                   OR available_ns < sample_ns
                   OR sample_ns % {STEP} <> 0
            ) AS invalid_input_timing,
            count(*) FILTER (
                WHERE previous_ns IS NOT NULL
                  AND sample_ns - previous_ns <> {STEP}
            ) AS broken_input_grid,
            count(*) FILTER (
                WHERE has_price IS NULL
                   OR in_long_gap IS NULL
                   OR project_excluded IS NULL
                   OR project_eligible IS NULL
            ) AS missing_input_flags,
            count(*) FILTER (
                WHERE has_price AND (
                    mid IS NULL OR NOT isfinite(mid) OR mid <= 0
                    OR is_interpolated IS NULL
                )
            ) AS invalid_present_prices,
            count(*) FILTER (
                WHERE NOT has_price AND mid IS NOT NULL
            ) AS prices_on_missing_rows,
            count(*) FILTER (
                WHERE has_price AND in_long_gap
            ) AS prices_inside_long_gaps,
            count(*) FILTER (
                WHERE project_eligible
                  AND (NOT has_price OR project_excluded)
            ) AS inconsistent_scope_flags
        FROM ordered
    """)
    assert all(value == 0 for value in source_checks.values()), source_checks

    statistics = []
    window_sql = [
        """
        history AS (
            ORDER BY sample_ns
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        """,
        """
        supporting_prices AS (
            ORDER BY sample_ns
            ROWS BETWEEN 90 PRECEDING AND CURRENT ROW
        )
        """,
    ]

    for key, n in WINDOWS.items():
        window_sql.append(f"""
            w_{key} AS (
                ORDER BY sample_ns
                ROWS BETWEEN {n - 1} PRECEDING AND CURRENT ROW
            )
        """)
        statistics.extend([
            f"count(change_10s_pips) OVER w_{key} AS n_changes_{key}",
            f"avg(change_10s_pips) OVER w_{key} AS raw_mean_{key}",
            f"avg(change_square) OVER w_{key} AS raw_mean_square_{key}",
            f"stddev_pop(change_10s_pips) OVER w_{key} AS raw_sd_{key}",
            f"sum(change_square) OVER w_{key} AS raw_sum_square_{key}",
            f"max(change_square) OVER w_{key} AS raw_max_square_{key}",
            (
                "count(*) FILTER (WHERE interpolated_change) "
                f"OVER w_{key} AS raw_interp_count_{key}"
            ),
        ])

    ddl(con, f"""
        CREATE OR REPLACE TEMP VIEW vol_raw_stats AS
        WITH lagged AS (
            SELECT
                sample_ns,
                available_ns AS input_available_ns,
                mid, has_price, is_interpolated,
                in_long_gap, project_excluded, project_eligible,
                lag(sample_ns) OVER time_order AS previous_ns,
                lag(mid) OVER time_order AS previous_mid,
                lag(has_price) OVER time_order AS previous_has_price,
                lag(is_interpolated) OVER time_order
                    AS previous_is_interpolated
            FROM vol_context
            WINDOW time_order AS (ORDER BY sample_ns)
        ),
        changes AS (
            SELECT *,
                CASE
                    WHEN has_price
                     AND previous_has_price
                     AND sample_ns - previous_ns = {STEP}
                    THEN (mid - previous_mid) / {PIP}
                END AS change_10s_pips
            FROM lagged
        ),
        quality AS (
            SELECT *,
                change_10s_pips * change_10s_pips AS change_square,
                CASE WHEN change_10s_pips IS NOT NULL THEN
                    coalesce(is_interpolated, false)
                    OR coalesce(previous_is_interpolated, false)
                ELSE false END AS interpolated_change
            FROM changes
        )
        SELECT *,
            min(CASE WHEN has_price THEN sample_ns END)
                OVER history AS first_context_price_ns,
            max(CASE WHEN NOT has_price THEN sample_ns END)
                OVER history AS last_missing_ns,
            max(input_available_ns)
                OVER supporting_prices AS support_available_ns,
            {", ".join(statistics)}
        FROM quality
        WINDOW {", ".join(window_sql)}
    """)

    metrics = []
    for key, n in WINDOWS.items():
        ready = f"n_changes_{key} = {n}"
        metrics.extend([
            f"n_changes_{key}",
            f"({ready}) AS ready_{key}",
            (
                f"CASE WHEN {ready} THEN sqrt(raw_mean_square_{key}) "
                f"END AS rms_{key}_pips"
            ),
            f"CASE WHEN {ready} THEN raw_sd_{key} END AS sd_{key}_pips",
            (
                f"CASE WHEN {ready} AND raw_sum_square_{key} > 0 "
                f"THEN raw_max_square_{key} / raw_sum_square_{key} "
                f"END AS shock_share_{key}"
            ),
            (
                f"CASE WHEN {ready} "
                f"THEN raw_interp_count_{key}::DOUBLE / {n} "
                f"END AS interp_change_fraction_{key}"
            ),
            (
                f"CASE WHEN {ready} THEN raw_sum_square_{key} = 0 "
                f"END AS flat_{key}"
            ),
        ])

    ddl(con, f"""
        CREATE OR REPLACE TEMP VIEW vol_output AS
        WITH measured AS (
            SELECT
                sample_ns,
                input_available_ns,
                greatest(sample_ns, support_available_ns) AS available_ns,
                mid, has_price, is_interpolated,
                in_long_gap, project_excluded, project_eligible,
                change_10s_pips,
                has_price AND change_10s_pips IS NULL AS restart,
                CAST(
                    CASE WHEN has_price THEN
                        least(
                            90,
                            greatest(
                                0,
                                (
                                    sample_ns - coalesce(
                                        last_missing_ns + {STEP},
                                        first_context_price_ns
                                    )
                                ) // {STEP}
                            )
                        )
                    ELSE 0 END AS BIGINT
                ) AS fresh_changes_capped,
                {", ".join(metrics)}
            FROM vol_raw_stats
        )
        SELECT *,
            (available_ns - sample_ns)::DOUBLE / 1e9
                AS publication_delay_seconds,
            CASE
                WHEN ready_1m AND ready_15m AND rms_15m_pips > 0
                THEN rms_1m_pips / rms_15m_pips
            END AS rms_ratio_1m_15m,
            CASE
                WHEN ready_1m AND ready_15m
                THEN rms_15m_pips = 0
            END AS rms_ratio_1m_15m_denominator_zero
        FROM measured
        WHERE sample_ns >= {lower}
          AND sample_ns < {upper}
    """)

    return source_checks


def validate_output(con, meta):
    result = fx.record(con, f"""
        SELECT
            count(*) AS rows,
            min(sample_ns) AS first_ns,
            max(sample_ns) AS last_ns,
            count(*) - count(DISTINCT sample_ns) AS duplicate_output_times,
            count(*) FILTER (
                WHERE available_ns < sample_ns
                   OR available_ns < input_available_ns
            ) AS availability_errors,
            count(*) FILTER (
                WHERE change_10s_pips IS NOT NULL AND NOT has_price
            ) AS changes_without_current_price
        FROM saved_features
    """)

    checks = {
        "row_count_difference": result["rows"] - meta["rows"],
        "first_timestamp_difference": result["first_ns"] - meta["first_ns"],
        "last_timestamp_difference": result["last_ns"] - meta["last_ns"],
        "duplicate_output_times": result["duplicate_output_times"],
        "availability_errors": result["availability_errors"],
        "changes_without_current_price": result["changes_without_current_price"],
    }

    source_fields = ", ".join(SOURCE_COLUMNS)
    saved_fields = ", ".join(
        "input_available_ns AS available_ns" if name == "available_ns" else name
        for name in SOURCE_COLUMNS
    )
    checks["changed_source_fields"] = con.execute(f"""
        SELECT count(*) FROM (
            (SELECT {source_fields} FROM current_input
             EXCEPT ALL SELECT {saved_fields} FROM saved_features)
            UNION ALL
            (SELECT {saved_fields} FROM saved_features
             EXCEPT ALL SELECT {source_fields} FROM current_input)
        )
    """).fetchall()[0][0]

    for key, n in WINDOWS.items():
        metrics = f"""
            rms_{key}_pips, sd_{key}_pips,
            shock_share_{key}, interp_change_fraction_{key}
        """
        result = fx.record(con, f"""
            SELECT
                count(*) FILTER (
                    WHERE ready_{key}
                        IS DISTINCT FROM (fresh_changes_capped >= {n})
                ) AS incorrect_readiness,
                count(*) FILTER (
                    WHERE NOT ready_{key} AND (
                        rms_{key}_pips IS NOT NULL
                        OR sd_{key}_pips IS NOT NULL
                        OR shock_share_{key} IS NOT NULL
                        OR interp_change_fraction_{key} IS NOT NULL
                        OR flat_{key} IS NOT NULL
                    )
                ) AS premature_values,
                count(*) FILTER (
                    WHERE ready_{key} AND (
                        rms_{key}_pips IS NULL
                        OR sd_{key}_pips IS NULL
                        OR interp_change_fraction_{key} IS NULL
                        OR flat_{key} IS NULL
                    )
                ) AS missing_ready_values,
                count(*) FILTER (
                    WHERE ready_{key} AND (
                        NOT isfinite(rms_{key}_pips)
                        OR NOT isfinite(sd_{key}_pips)
                        OR rms_{key}_pips < 0
                        OR sd_{key}_pips < 0
                        OR sd_{key}_pips > rms_{key}_pips + {VALUE_TOL}
                        OR interp_change_fraction_{key} < 0
                        OR interp_change_fraction_{key} > 1
                    )
                ) AS invalid_values,
                count(*) FILTER (
                    WHERE ready_{key} AND flat_{key} AND (
                        rms_{key}_pips <> 0
                        OR sd_{key}_pips <> 0
                        OR shock_share_{key} IS NOT NULL
                    )
                ) AS incorrect_flat_windows,
                count(*) FILTER (
                    WHERE ready_{key} AND NOT flat_{key} AND (
                        shock_share_{key} IS NULL
                        OR NOT isfinite(shock_share_{key})
                        OR shock_share_{key} < {1.0 / n} - {SHARE_TOL}
                        OR shock_share_{key} > 1 + {SHARE_TOL}
                    )
                ) AS invalid_shock_shares
            FROM saved_features
        """)
        checks.update({
            f"{key}_{name}": value for name, value in result.items()
        })

        checks[f"{key}_second_moment_identity_errors"] = con.execute(f"""
            SELECT count(*)
            FROM vol_raw_stats
            WHERE sample_ns >= {meta["lower_ns"]}
              AND sample_ns < {meta["upper_ns"]}
              AND n_changes_{key} = {n}
              AND abs(
                    raw_mean_square_{key}
                    - raw_sd_{key} * raw_sd_{key}
                    - raw_mean_{key} * raw_mean_{key}
                  ) > {VALUE_TOL} * (1 + raw_mean_square_{key})
        """).fetchall()[0][0]

    ratio_checks = fx.record(con, f"""
        SELECT
            count(*) FILTER (
                WHERE (
                    ready_1m AND ready_15m AND rms_15m_pips > 0
                ) IS DISTINCT FROM (rms_ratio_1m_15m IS NOT NULL)
            ) AS ratio_presence_errors,
            count(*) FILTER (
                WHERE rms_ratio_1m_15m IS NOT NULL AND (
                    NOT isfinite(rms_ratio_1m_15m)
                    OR rms_ratio_1m_15m < 0
                    OR rms_ratio_1m_15m > sqrt(15.0) + {VALUE_TOL}
                )
            ) AS ratio_value_errors,
            count(*) FILTER (
                WHERE rms_ratio_1m_15m_denominator_zero
                  AND rms_ratio_1m_15m IS NOT NULL
            ) AS zero_denominator_errors
        FROM saved_features
    """)
    checks.update(ratio_checks)
    assert all(value == 0 for value in checks.values()), checks
    return checks


def monthly_statistics(con, label):
    expressions = [
        "count(*) AS sample_rows",
        "count(*) FILTER (WHERE has_price) AS priced_rows",
        "count(*) FILTER (WHERE NOT has_price) AS missing_price_rows",
        "count(*) FILTER (WHERE change_10s_pips IS NOT NULL) AS valid_change_rows",
        "count(*) FILTER (WHERE restart) AS restart_rows",
        "count(*) FILTER (WHERE rms_ratio_1m_15m IS NOT NULL) AS ratio_defined_rows",
        "max(publication_delay_seconds) AS max_publication_delay_seconds",
    ]
    for key in WINDOWS:
        expressions.extend([
            f"count(*) FILTER (WHERE ready_{key}) AS ready_{key}_rows",
            (
                f"count(*) FILTER (WHERE ready_{key} AND project_eligible) "
                f"AS project_ready_{key}_rows"
            ),
            f"count(*) FILTER (WHERE flat_{key}) AS flat_{key}_rows",
            (
                f"count(*) FILTER (WHERE ready_{key} "
                f"AND interp_change_fraction_{key} = 1) "
                f"AS all_changes_touch_interpolation_{key}_rows"
            ),
        ])
    result = fx.record(
        con, "SELECT " + ", ".join(expressions) + " FROM saved_features"
    )
    assert result["valid_change_rows"] + result["restart_rows"] == result["priced_rows"]
    return {"year_month": label, **result}


def prepare(
    root, run_id, stop_after_month="2016-01",
    memory="2GB", threads=2, spill_limit="20GB"
):
    root = Path(root).resolve()
    assert run_id and Path(run_id).name == run_id

    source = root / "data" / "processed" / SOURCE_RUN
    source_reports = root / "reports" / "tables" / "sampling" / SOURCE_RUN

    source_state = fx.load_json(source / "run_state.json")
    source_summary = fx.load_json(source_reports / "execution_summary.json")
    source_configuration = fx.load_json(source / "configuration.json")
    completion = fx.load_json(
        source_reports / "full_history_completion_checks.json"
    )

    assert source_state["fingerprint"] == SOURCE_FINGERPRINT
    assert source_state["status"] == "full_history_processed"
    assert source_summary["full_history_complete"]
    assert source_summary["nonzero_machine_checks"] == 0
    assert completion["fingerprint"] == SOURCE_FINGERPRINT
    assert completion["machine_checks_passed"]
    assert all(completion["checks"].values())

    first_real = int(source_configuration["inputs"][0]["first_ns"])
    last_real = int(source_configuration["inputs"][-1]["last_ns"])

    inputs = []
    checkpoint_paths = sorted((source / "_checkpoints").glob("*.json"))

    for checkpoint_path in checkpoint_paths:
        label = checkpoint_path.stem
        checkpoint = fx.load_json(checkpoint_path)
        assert checkpoint["fingerprint"] == SOURCE_FINGERPRINT
        assert all(value == 0 for value in checkpoint["checks"].values())

        path = source / "sampled_10s" / f"{label}.parquet"
        actual = file_info(path)
        expected = checkpoint["outputs"]["sampled_10s"]
        assert all(actual[key] == expected[key] for key in actual), path

        schema = pq.read_schema(path)
        assert set(SOURCE_COLUMNS).issubset(schema.names), path
        assert pa.types.is_int64(schema.field("sample_ns").type)
        assert pa.types.is_int64(schema.field("available_ns").type)

        lower, upper = fx.month_bounds(label)
        first = max(lower, ((first_real + STEP - 1) // STEP) * STEP)
        last = min(upper - STEP, (last_real // STEP) * STEP)

        assert actual["rows"] == (last - first) // STEP + 1
        inputs.append({
            "year_month": label,
            "path": path.relative_to(root).as_posix(),
            **actual,
            "first_ns": first,
            "last_ns": last,
            "lower_ns": lower,
            "upper_ns": upper,
            "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": sha(checkpoint_path),
        })

    labels = [item["year_month"] for item in inputs]
    assert labels == [
        str(period) for period in pd.period_range(labels[0], labels[-1], freq="M")
    ]
    assert len(inputs) == source_summary["processed_months"]
    assert sum(item["rows"] for item in inputs) == source_summary["sample_rows"]
    if stop_after_month is not None:
        assert stop_after_month in labels

    run = root / "outputs" / "features" / "volatility" / run_id
    tables = root / "reports" / "tables" / "volatility" / run_id
    figures = root / "reports" / "figures" / "volatility" / run_id
    for directory in (run, tables, figures, run / "_checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)

    configuration = {
        "definition": "multiscale_volatility_v1",
        "run_id": run_id,
        "source_run": SOURCE_RUN,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "source_configuration_sha256": sha(source / "configuration.json"),
        "helper_sha256": sha(__file__),
        "etl_utilities_sha256": sha(fx.__file__),
        "versions": versions(),
        "inputs": inputs,
        "step_seconds": 10,
        "windows_changes": WINDOWS,
        "pip_size": PIP,
        "standard_deviation": "population; division by N",
        "full_window_required": True,
        "excluded_hours_with_data_continue_history": True,
        "interpolation_remains_eligible": True,
        "ratio": "RMS 1m / RMS 15m",
        "zero_denominator_result": None,
        "availability": "integer max of preceding 91 supporting grid rows",
        "learned_thresholds": False,
    }
    fingerprint = hashlib.sha256(json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    path = run / "configuration.json"
    if path.exists():
        existing = fx.load_json(path)
        changed = [
            key for key in sorted(set(existing) | set(configuration))
            if existing.get(key) != configuration.get(key)
        ]
        assert not changed, (
            f"Configuration changed in {changed}. "
            "Use a new RUN_ID; do not combine incompatible outputs."
        )
    else:
        fx.dump_json(path, configuration)

    pd.DataFrame(inputs).to_csv(tables / "input_inventory.csv", index=False)

    return {
        "root": root, "run": run, "tables": tables, "figures": figures,
        "run_id": run_id, "fingerprint": fingerprint,
        "inputs": inputs,
        "target": [
            item for item in inputs
            if stop_after_month is None or item["year_month"] <= stop_after_month
        ],
        "memory": memory, "threads": threads, "spill_limit": spill_limit,
        "configuration": configuration,
        "source_summary": source_summary,
    }


def check_inputs(ctx):
    for item in ctx["inputs"]:
        path = ctx["root"] / item["path"]
        actual = file_info(path)
        assert all(actual[key] == item[key] for key in actual), path
        assert sha(
            ctx["root"] / item["checkpoint_path"]
        ) == item["checkpoint_sha256"]
    assert sha(__file__) == ctx["configuration"]["helper_sha256"]
    assert sha(fx.__file__) == ctx["configuration"]["etl_utilities_sha256"]


def process(ctx):
    test_path = ctx["tables"] / "synthetic_checks.json"
    if not test_path.exists():
        self_checks(ctx)
    evidence = fx.load_json(test_path)
    assert evidence["fingerprint"] == ctx["fingerprint"]
    assert all(evidence["checks"].values())

    check_inputs(ctx)
    started = time.perf_counter()
    completed = []
    new_months = 0

    for index, meta in enumerate(ctx["target"]):
        label = meta["year_month"]
        checkpoint_path = ctx["run"] / "_checkpoints" / f"{label}.json"
        destination = ctx["run"] / f"{label}.parquet"

        if checkpoint_path.exists():
            done = fx.load_json(checkpoint_path)
            assert done["fingerprint"] == ctx["fingerprint"]
            assert all(value == 0 for value in done["checks"].values())
            assert file_info(destination) == done["output"]
            print(label, "resumed")
        else:
            month_started = time.perf_counter()
            spill = ctx["run"] / "_work" / "spill"
            spill.mkdir(parents=True, exist_ok=True)

            with duckdb.connect(config={
                "memory_limit": ctx["memory"],
                "threads": ctx["threads"],
                "temp_directory": str(spill),
                "max_temp_directory_size": ctx["spill_limit"],
            }) as con:
                current_path = ctx["root"] / meta["path"]
                con.read_parquet(
                    str(current_path), hive_partitioning=False
                ).create_view("current_input", replace=True)

                columns = ", ".join(SOURCE_COLUMNS)
                context_query = f"SELECT {columns} FROM current_input"

                if index:
                    previous = ctx["inputs"][index - 1]
                    previous_path = ctx["root"] / previous["path"]
                    cutoff = meta["lower_ns"] - (MAX_N + 1) * STEP
                    context_query += f"""
                        UNION ALL
                        SELECT {columns}
                        FROM read_parquet(
                            {fx.literal(previous_path)},
                            hive_partitioning=false
                        )
                        WHERE sample_ns >= {cutoff}
                    """

                checks = build_views(
                    con, context_query, meta["lower_ns"], meta["upper_ns"]
                )
                fx.write_query(
                    con, "SELECT * FROM vol_output ORDER BY sample_ns", destination
                )
                con.read_parquet(
                    str(destination), hive_partitioning=False
                ).create_view("saved_features", replace=True)

                checks.update(validate_output(con, meta))
                statistics = monthly_statistics(con, label)

            statistics["processing_minutes"] = round(
                (time.perf_counter() - month_started) / 60, 3
            )
            done = {
                "fingerprint": ctx["fingerprint"],
                "summary": statistics,
                "checks": checks,
                "output": file_info(destination),
            }
            fx.dump_json(checkpoint_path, done)
            new_months += 1
            print(
                label,
                f"rows={statistics['sample_rows']:,}",
                f"ready 30s={statistics['ready_30s_rows']:,}",
                f"ready 15m={statistics['ready_15m_rows']:,}",
            )

        completed.append(done)

    check_inputs(ctx)
    return finish(
        ctx, completed, new_months, (time.perf_counter() - started) / 60
    )


def write_review_tables(ctx, completed):
    paths = [
        str(ctx["run"] / f"{item['summary']['year_month']}.parquet")
        for item in completed
    ]
    tables = ctx["tables"]

    coverage = []
    for key, n in WINDOWS.items():
        coverage.append({
            "window": key,
            "required_changes": n,
            "required_prices": n + 1,
            "ready_rows": sum(
                item["summary"][f"ready_{key}_rows"] for item in completed
            ),
            "project_ready_rows": sum(
                item["summary"][f"project_ready_{key}_rows"] for item in completed
            ),
            "flat_rows": sum(
                item["summary"][f"flat_{key}_rows"] for item in completed
            ),
            "all_changes_touch_interpolation_rows": sum(
                item["summary"][f"all_changes_touch_interpolation_{key}_rows"]
                for item in completed
            ),
        })
    pd.DataFrame(coverage).to_csv(tables / "window_coverage.csv", index=False)

    with duckdb.connect(config={
        "memory_limit": ctx["memory"], "threads": ctx["threads"]
    }) as con:
        con.read_parquet(
            paths, hive_partitioning=False
        ).create_view("review_features")

        warmup = []
        for key, n in WINDOWS.items():
            row = fx.record(con, f"""
                SELECT
                    sample_ns, available_ns, fresh_changes_capped,
                    n_changes_{key} AS valid_changes,
                    rms_{key}_pips AS rms_pips,
                    sd_{key}_pips AS sd_pips,
                    shock_share_{key} AS shock_share,
                    interp_change_fraction_{key} AS interpolation_fraction
                FROM review_features
                WHERE ready_{key}
                ORDER BY sample_ns LIMIT 1
            """)
            if row is not None:
                assert row["valid_changes"] == n
                warmup.append({"window": key, **row})

        warmup = pd.DataFrame(warmup)
        if not warmup.empty:
            for column in (
                "sample_ns", "available_ns", "fresh_changes_capped", "valid_changes"
            ):
                warmup[column] = warmup[column].astype("Int64")
        display_times(
            warmup, ["sample_ns", "available_ns"]
        ).to_csv(tables / "first_ready_windows.csv", index=False)

        delayed = frame(con, """
            SELECT
                sample_ns, available_ns, publication_delay_seconds,
                rms_30s_pips, sd_30s_pips,
                interp_change_fraction_30s, project_eligible
            FROM review_features
            WHERE ready_30s AND available_ns > sample_ns
            ORDER BY publication_delay_seconds DESC, sample_ns
            LIMIT 10
        """)
        display_times(
            delayed, ["sample_ns", "available_ns"]
        ).to_csv(tables / "delayed_examples.csv", index=False)

        interpolated = frame(con, """
            SELECT
                sample_ns, available_ns, rms_30s_pips, sd_30s_pips,
                interp_change_fraction_30s, ready_30s, project_eligible
            FROM review_features
            WHERE ready_30s AND interp_change_fraction_30s = 1
            ORDER BY sample_ns LIMIT 10
        """)
        display_times(
            interpolated, ["sample_ns", "available_ns"]
        ).to_csv(tables / "interpolation_examples.csv", index=False)

        restart = fx.record(con, f"""
            SELECT sample_ns
            FROM review_features
            WHERE restart AND sample_ns > {ctx["inputs"][0]["first_ns"]}
            ORDER BY sample_ns LIMIT 1
        """)
        restart_columns = """
            sample_ns, available_ns, mid, has_price, change_10s_pips,
            fresh_changes_capped,
            ready_30s, ready_1m, ready_5m, ready_15m,
            rms_30s_pips, rms_1m_pips, rms_5m_pips, rms_15m_pips,
            project_eligible
        """
        if restart is None:
            restart_query = (
                f"SELECT {restart_columns} FROM review_features LIMIT 0"
            )
        else:
            at = int(restart["sample_ns"])
            restart_query = f"""
                SELECT {restart_columns}
                FROM review_features
                WHERE sample_ns >= {at - 20 * NS}
                  AND sample_ns <= {at + 16 * 60 * NS}
                ORDER BY sample_ns
            """
        restart_data = frame(con, restart_query)
        display_times(
            restart_data, ["sample_ns", "available_ns"]
        ).to_csv(tables / "restart_example.csv", index=False)

        # Exploratory sampling sensitivity on one fully supported 15-minute block.
        # Production calculations remain on the ten-second grid.
        block = fx.record(con, f"""
            SELECT sample_ns AS end_ns
            FROM review_features
            WHERE ready_15m AND sample_ns % {900 * NS} = 0
            ORDER BY sample_ns LIMIT 1
        """)
        sensitivity = []
        if block is not None:
            end = int(block["end_ns"])
            start = end - 900 * NS
            prices = frame(con, f"""
                SELECT sample_ns, mid, is_interpolated
                FROM review_features
                WHERE sample_ns BETWEEN {start} AND {end}
                ORDER BY sample_ns
            """)
            assert len(prices) == 91
            assert prices["mid"].notna().all()

            times = prices["sample_ns"].to_numpy(dtype=np.int64)
            for seconds in (10, 30, 60):
                chosen = prices.loc[
                    (times - start) % (seconds * NS) == 0
                ]
                changes = np.diff(chosen["mid"].to_numpy(dtype=float)) / PIP
                sensitivity.append({
                    "block_start_athens": fx.athens(start),
                    "block_end_athens": fx.athens(end),
                    "change_interval_seconds": seconds,
                    "number_of_changes": len(changes),
                    "realised_movement_over_same_15m_pips": float(
                        np.sqrt(np.sum(changes * changes))
                    ),
                    "selected_price_interpolation_fraction": float(
                        chosen["is_interpolated"].fillna(False).mean()
                    ),
                })
        pd.DataFrame(sensitivity).to_csv(
            tables / "sampling_sensitivity.csv", index=False
        )


def record_documents(ctx, summary):
    root = ctx["root"]
    scope = (
        "full available history"
        if summary["full_history_complete"]
        else f"prefix through {summary['processed_through']}"
    )
    marker = f"notebook-03-{ctx['run_id']}"

    write_section(root / "docs" / "design_decisions.md", marker, f"""
## Notebook 03 — {ctx["run_id"]}

- Input: all-hours ten-second sampled mids from `{SOURCE_RUN}`.
- Lookbacks: 30 seconds, 1 minute, 5 minutes and 15 minutes.
- Measurements: RMS movement and centred population standard deviation.
- Units: pips of ten-second change; no annualisation.
- Diagnostics: largest-change share and interpolation-supported-change fraction.
- Comparison: RMS 1-minute / RMS 15-minute, undefined at a zero denominator.
- History: require complete consecutive windows after missing periods.
- Excluded hours with available prices continue the measurement history.
- Interpolation remains eligible, including entirely reconstructed support.
- Availability: conservative exact integer maximum over supporting prices.
- No learned thresholds, regime classifications or execution rules are fitted.
- Status of these definitions: agreed.
- Forecast-priced limit exits and execution clips remain provisional applications.
- Execution scope: {scope}.
- Saved machine checks: passed.
- Human review of Notebook 03 outputs: pending.
- Fingerprint: `{ctx["fingerprint"]}`.
""")

    write_section(root / "CONTRIBUTIONS.md", marker, f"""
## Notebook 03 — {ctx["run_id"]}

Raymond Fendi:
- Defined volatility's intended use in execution urgency and regime research.
- Selected 30-second, 1-minute, 5-minute and 15-minute evaluations.
- Agreed to ten-second inputs and updates.
- Agreed to RMS and population SD as separate saved measurements.
- Required fresh full lookbacks after missing-market periods.
- Retained available excluded-hour data and eligible interpolation.
- Agreed to fast/slow and largest-move diagnostics.
- Described forecast-priced limit exits and multiple clips as future applications.

OpenAI Codex:
- Prepared the implementation, inline synthetic checks and documentation.
- Distinguished movement magnitude, dispersion, direction and execution assumptions.
- Preserved input availability and context across processing boundaries.

Execution scope: {scope}.
Machine checks passed when this record was written.
Raymond's acceptance of Notebook 03 execution outputs remains pending.
""")

    write_section(root / "README.md", marker, f"""
## Notebook 03 progress

Multiscale RMS movement, population SD and supporting diagnostics are implemented.
Execution scope: {scope}.
Machine checks passed; human output review remains pending.
Measurements retain missing-data, reconstruction and availability information.
No execution policy or forecasting model is implemented by this notebook.

Results pending execution and validation.
""")

    state = f"""# Project state

- Notebook 00, 01 and 01a: executed and reviewed.
- Notebook 02: full-history machine checks and Codex technical review completed.
- Current stage: Notebook 03 output review.
- Notebook 03 run: `{ctx["run_id"]}`.
- Notebook 03 execution scope: {scope}.
- Notebook 03 machine checks: passed.
- Notebook 03 human review: pending.

## Locations

- Input: `data/processed/{SOURCE_RUN}/sampled_10s/`.
- Features: `outputs/features/volatility/{ctx["run_id"]}/`.
- Reports: `reports/tables/volatility/{ctx["run_id"]}/`.
- Figures: `reports/figures/volatility/{ctx["run_id"]}/`.

## Outstanding matters

- Review Notebook 03 outputs before expanding processing.
- Exact historical provider closure completeness remains a limitation.
- Interpolation's statistical effects remain visible through diagnostics.
- Regime thresholds, entropy, forecasting and execution policies remain for design.

## Next step

Review the saved Notebook 03 outputs.
If this is a prefix run, expand using a complete continuation cell.
After full-history review, discuss Notebook 04 entropy.

## Outputs invalidated

None from the accepted upstream runs.
Notebook 03 is a new versioned feature output.
Changed definitions or upstream inputs require new compatible outputs.
"""
    (root / "docs" / "project_state.md").write_text(state, encoding="utf-8")


def finish(ctx, completed, new_months, minutes):
    monthly = pd.DataFrame([item["summary"] for item in completed])
    checks = pd.DataFrame([
        {"year_month": item["summary"]["year_month"], **item["checks"]}
        for item in completed
    ])
    assert checks.drop(columns="year_month").eq(0).all().all()

    expected_rows = sum(item["rows"] for item in ctx["target"])
    assert int(monthly["sample_rows"].sum()) == expected_rows

    full = len(completed) == len(ctx["inputs"])
    if full:
        assert expected_rows == ctx["source_summary"]["sample_rows"]

    monthly.to_csv(ctx["tables"] / "monthly_summary.csv", index=False)
    checks.to_csv(ctx["tables"] / "monthly_checks.csv", index=False)

    count_columns = [
        column for column in monthly
        if column.endswith("_rows")
    ]
    annual = (
        monthly.assign(year=monthly["year_month"].str[:4])
        .groupby("year", as_index=False)[count_columns].sum()
    )
    annual.to_csv(ctx["tables"] / "annual_summary.csv", index=False)

    summary = {
        "notebook": "03_multiscale_volatility.ipynb",
        "run_id": ctx["run_id"],
        "fingerprint": ctx["fingerprint"],
        "source_run": SOURCE_RUN,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "mode": "full_history" if full else "prefix_review",
        "available_input_months": len(ctx["inputs"]),
        "processed_months": len(completed),
        "processed_through": completed[-1]["summary"]["year_month"],
        "full_history_complete": full,
        "new_months_processed_this_call": new_months,
        "processed_sample_rows": expected_rows,
        "priced_rows": int(monthly["priced_rows"].sum()),
        "missing_price_rows": int(monthly["missing_price_rows"].sum()),
        "valid_change_rows": int(monthly["valid_change_rows"].sum()),
        "restart_rows": int(monthly["restart_rows"].sum()),
        "ready_rows_by_window": {
            key: int(monthly[f"ready_{key}_rows"].sum())
            for key in WINDOWS
        },
        "ratio_defined_rows": int(monthly["ratio_defined_rows"].sum()),
        "max_publication_delay_seconds": float(
            monthly["max_publication_delay_seconds"].max()
        ),
        "nonzero_machine_checks": int(
            checks.drop(columns="year_month").ne(0).sum().sum()
        ),
        "synthetic_checks_passed": True,
        "machine_checks_passed": True,
        "human_review": "pending",
        "processing_call_minutes": round(minutes, 3),
        "source_stability_check": "size and modification time; no data content hash",
    }

    write_review_tables(ctx, completed)
    fx.dump_json(ctx["tables"] / "execution_summary.json", summary)
    fx.dump_json(ctx["run"] / "run_state.json", {
        "fingerprint": ctx["fingerprint"],
        "status": "full_history_processed" if full else "prefix_processed",
        "processed_through": summary["processed_through"],
        "machine_checks_passed": True,
        "human_review": "pending",
    })
    fx.dump_json(ctx["run"] / "output_contract.json", {
        "timestamp_units": "UTC Unix nanoseconds",
        "price_change_units": "pips per ten-second change",
        "windows_changes": WINDOWS,
        "population_sd": True,
        "full_consecutive_windows_required": True,
        "common_publication_timestamp": "available_ns",
        "missing_values_are_not_zero": True,
        "flat_windows": "ready=true, RMS=SD=0; shock share undefined",
        "ratio": "RMS 1m / RMS 15m; undefined at denominator zero",
        "project_scope": (
            "History includes excluded hours. For an experiment, require the "
            "relevant ready flag, project_eligible and availability by decision time."
        ),
        "interp_change_fraction": (
            "Fraction of changes with at least one interpolated supporting "
            "sampled-price source; not the fraction of raw ticks."
        ),
        "fresh_changes_capped": "consecutive valid changes since a break, capped at 90",
        "no_execution_or_forecast_claim": True,
    })
    record_documents(ctx, summary)

    return {
        "context": ctx, "completed": completed,
        "monthly": monthly, "checks": checks,
        "annual": annual, "summary": summary,
    }


def fixture(prices, origin, missing=None, interpolated=None, availability=None,
            excluded=None):
    prices = np.asarray(prices, dtype=float)
    count = len(prices)
    times = int(origin) + np.arange(count, dtype=np.int64) * STEP
    missing = np.zeros(count, dtype=bool) if missing is None else missing
    interpolated = (
        np.zeros(count, dtype=bool) if interpolated is None else interpolated
    )
    excluded = np.zeros(count, dtype=bool) if excluded is None else excluded
    availability = times.copy() if availability is None else availability

    return pa.table({
        "sample_ns": pa.array(times, type=pa.int64()),
        "available_ns": pa.array(availability, type=pa.int64()),
        "mid": pa.array([
            None if missing[i] else float(prices[i]) for i in range(count)
        ], type=pa.float64()),
        "has_price": ~missing,
        "is_interpolated": pa.array([
            None if missing[i] else bool(interpolated[i]) for i in range(count)
        ], type=pa.bool_()),
        "in_long_gap": missing,
        "project_excluded": excluded,
        "project_eligible": (~missing) & (~excluded),
    })


def evaluate_fixture(con, data, lower=None, upper=None):
    con.register("fixture_input", data)
    times = data["sample_ns"].to_numpy()
    lower = int(times[0]) if lower is None else int(lower)
    upper = int(times[-1] + STEP) if upper is None else int(upper)
    build_views(con, "SELECT * FROM fixture_input", lower, upper)
    return frame(con, "SELECT * FROM vol_output ORDER BY sample_ns")


def self_checks(ctx):
    path = ctx["tables"] / "synthetic_checks.json"
    if path.exists():
        saved = fx.load_json(path)
        assert saved["fingerprint"] == ctx["fingerprint"]
        assert all(saved["checks"].values())
        print("Saved synthetic-check evidence reused.")
        return saved

    origin = int(pd.Timestamp(
        "2016-01-31 23:40", tz="Europe/Athens"
    ).value)
    count = 240
    scenarios = {
        "flat": np.zeros(count - 1),
        "steady_up": np.ones(count - 1),
        "alternating": np.where(np.arange(count - 1) % 2 == 0, 1.0, -1.0),
        "single_jump": np.r_[np.zeros(count - 2), 5.0],
    }
    example_rows = []

    with duckdb.connect() as con:
        for name, changes in scenarios.items():
            prices = 1.10 + np.r_[0.0, np.cumsum(changes)] * PIP
            output = evaluate_fixture(con, fixture(prices, origin))
            last = output.iloc[-1]

            for key, n in WINDOWS.items():
                expected = changes[-n:]
                expected_rms = float(np.sqrt(np.mean(expected * expected)))
                expected_sd = float(np.std(expected, ddof=0))
                assert bool(last[f"ready_{key}"])
                np.testing.assert_allclose(
                    float(last[f"rms_{key}_pips"]),
                    expected_rms, atol=VALUE_TOL, rtol=1e-9,
                )
                np.testing.assert_allclose(
                    float(last[f"sd_{key}_pips"]),
                    expected_sd, atol=VALUE_TOL, rtol=1e-9,
                )
                if np.sum(expected * expected) == 0:
                    assert pd.isna(last[f"shock_share_{key}"])
                else:
                    expected_share = float(
                        np.max(expected * expected) / np.sum(expected * expected)
                    )
                    np.testing.assert_allclose(
                        float(last[f"shock_share_{key}"]),
                        expected_share, atol=SHARE_TOL, rtol=1e-9,
                    )

                first_ready = output.loc[output[f"ready_{key}"]].iloc[0]
                assert int(first_ready["sample_ns"]) == origin + n * STEP
                example_rows.append({
                    "scenario": name, "window": key,
                    "rms_pips": float(last[f"rms_{key}_pips"]),
                    "sd_pips": float(last[f"sd_{key}_pips"]),
                    "expected_rms_pips": expected_rms,
                    "expected_sd_pips": expected_sd,
                })

            if name == "flat":
                assert pd.isna(last["rms_ratio_1m_15m"])
            if name == "single_jump":
                np.testing.assert_allclose(
                    float(last["rms_ratio_1m_15m"]),
                    np.sqrt(15.0), atol=VALUE_TOL,
                )

        # Missing period: no return may connect the two price levels.
        count = 360
        prices = 1.10 + np.arange(count) * PIP
        missing = np.zeros(count, dtype=bool)
        missing[101:141] = True
        prices[141:] += 0.01
        data = fixture(prices, origin, missing=missing)
        whole = evaluate_fixture(con, data)

        assert pd.isna(whole.iloc[141]["change_10s_pips"])
        assert bool(whole.iloc[141]["restart"])

        for key, n in WINDOWS.items():
            assert not bool(whole.iloc[141 + n - 1][f"ready_{key}"])
            assert bool(whole.iloc[141 + n][f"ready_{key}"])
            np.testing.assert_allclose(
                float(whole.iloc[141 + n][f"rms_{key}_pips"]),
                1.0, atol=VALUE_TOL,
            )

        # Processing partitions include a real calendar-month boundary.
        pieces = []
        for start, stop in ((0, 120), (120, 197), (197, count)):
            context_start = max(0, start - 91)
            piece = data.slice(context_start, stop - context_start)
            times = data["sample_ns"].to_numpy()
            pieces.append(evaluate_fixture(
                con, piece,
                lower=int(times[start]),
                upper=int(times[stop - 1] + STEP),
            ))
        split = pd.concat(pieces, ignore_index=True)

        float_columns = [
            "mid", "change_10s_pips", "publication_delay_seconds",
            "rms_ratio_1m_15m",
            *[
                column
                for key in WINDOWS
                for column in (
                    f"rms_{key}_pips", f"sd_{key}_pips",
                    f"shock_share_{key}", f"interp_change_fraction_{key}",
                )
            ],
        ]
        for column in whole:
            if column in float_columns:
                np.testing.assert_allclose(
                    whole[column].to_numpy(dtype=float, na_value=np.nan),
                    split[column].to_numpy(dtype=float, na_value=np.nan),
                    atol=VALUE_TOL, rtol=1e-9, equal_nan=True,
                )
            else:
                pd.testing.assert_series_equal(
                    whole[column].reset_index(drop=True),
                    split[column].reset_index(drop=True),
                )

        # Excluded hours containing prices do not reset measurements.
        excluded = np.zeros(240, dtype=bool)
        excluded[40:80] = True
        prices = 1.10 + np.arange(240) * PIP
        normal = evaluate_fixture(con, fixture(prices, origin))
        flagged = evaluate_fixture(
            con, fixture(prices, origin, excluded=excluded)
        )
        for key in WINDOWS:
            np.testing.assert_allclose(
                normal[f"rms_{key}_pips"].to_numpy(dtype=float, na_value=np.nan),
                flagged[f"rms_{key}_pips"].to_numpy(dtype=float, na_value=np.nan),
                atol=VALUE_TOL, equal_nan=True,
            )
            pd.testing.assert_series_equal(
                normal[f"ready_{key}"], flagged[f"ready_{key}"]
            )

        # Eligible reconstructed support and exact nanosecond availability.
        interpolated = np.zeros(120, dtype=bool)
        interpolated[1:21] = True
        times = origin + np.arange(120, dtype=np.int64) * STEP
        availability = times.copy()
        right_anchor = int(times[20]) + 7
        availability[1:21] = right_anchor

        reconstructed = evaluate_fixture(con, fixture(
            1.10 + np.arange(120) * PIP,
            origin,
            interpolated=interpolated,
            availability=availability,
        ))
        row = reconstructed.iloc[3]
        assert bool(row["ready_30s"])
        assert float(row["interp_change_fraction_30s"]) == 1.0
        assert int(row["available_ns"]) == right_anchor

    checks = {
        "flat_trend_alternating_and_shock": True,
        "population_sd_and_rms_hand_checks": True,
        "warmup_lengths": True,
        "zero_denominators": True,
        "no_cross_gap_return": True,
        "fresh_window_after_missing_period": True,
        "whole_versus_partitioned_processing": True,
        "excluded_hours_do_not_reset": True,
        "interpolation_remains_eligible": True,
        "exact_integer_availability": True,
    }
    result = {"fingerprint": ctx["fingerprint"], "checks": checks}
    pd.DataFrame(example_rows).to_csv(
        ctx["tables"] / "synthetic_examples.csv", index=False
    )
    fx.dump_json(path, result)
    print("SYNTHETIC_VOLATILITY_CHECKS_PASSED")
    return result
