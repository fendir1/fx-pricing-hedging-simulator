
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

from functions.etl import clean_quotes as fx

NS = 1_000_000_000
STEP = 10 * NS
BAR = 180 * NS
LIMIT = 300 * NS
PIP = 0.0001
WINDOWS = (15, 30, 60, 100, 200)


def frame(con, query):
    # Preserve nullable integer nanoseconds without float conversion.
    return con.sql(query).to_arrow_table().to_pandas(
        types_mapper=pd.ArrowDtype
    )


def ddl(con, query):
    con.execute(query)
    con.complete()


def save_frame(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.parquet")
    pq.write_table(
        pa.Table.from_pandas(data, preserve_index=False),
        temporary,
        compression="zstd",
    )
    temporary.replace(path)
    fx.verify_parquet(path, len(data))


def file_info(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": pq.ParquetFile(path).metadata.num_rows,
    }


def overlap_flags(starts, ends, intervals):
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if not intervals:
        return np.zeros(len(starts), dtype=bool)

    intervals = np.asarray(intervals, dtype=np.int64)
    index = np.searchsorted(intervals[:, 1], starts, side="right")
    safe = np.minimum(index, len(intervals) - 1)
    return (
        (index < len(intervals))
        & (intervals[safe, 0] < ends)
    )


def initial_state():
    return {
        "valid_candles": 0,
        "tail_closes": [],
        "ema_values": {str(n): None for n in WINDOWS},
        "dependency_available_ns": None,
        "last_candle_end_ns": None,
    }


def display_times(data, columns):
    result = data.copy()
    for column in columns:
        if column in result:
            result[column.replace("_ns", "_athens")] = (
                pd.to_datetime(
                    result[column].astype("Int64"),
                    unit="ns",
                    utc=True,
                ).dt.tz_convert("Europe/Athens")
            )
    return result


def write_section(path, marker, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    replacement = start + "\n" + body.strip() + "\n" + end

    if start in text:
        assert end in text, f"Unfinished documentation marker: {path}"
        left = text.index(start)
        right = text.index(end, left) + len(end)
        text = text[:left] + replacement + text[right:]
    else:
        text = text.rstrip() + "\n\n" + replacement

    path.write_text(text.strip() + "\n", encoding="utf-8")


def fixture_rows(times, bids):
    times = [int(value) for value in times]
    bids = np.asarray(bids, dtype=float)
    gaps = [None] + [
        times[i] - times[i - 1] for i in range(1, len(times))
    ]
    segments = np.cumsum([
        0 if gap is None else int(gap > LIMIT) for gap in gaps
    ]).astype(np.int64)

    return pa.table({
        "ts_ns": pa.array(times, type=pa.int64()),
        "bid": bids,
        "ask": bids + 0.0001,
        "bid_size": np.ones(len(times)),
        "ask_size": np.ones(len(times)),
        "source_copies": np.ones(len(times), dtype=np.int64),
        "representative_source_id": np.zeros(len(times), dtype=np.int64),
        "representative_source_record": np.arange(
            1, len(times) + 1, dtype=np.int64
        ),
        "a_ns": pa.array([None] + times[:-1], type=pa.int64()),
        "gap_ns": pa.array(gaps, type=pa.int64()),
        "data_segment_id": segments,
    })


def fixture_meta(table, path, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    rows = table.to_pylist()
    return {
        "path": str(path),
        "year_month": label,
        "observed_rows": len(rows),
        "first": rows[0],
        "last": rows[-1],
    }


def reconstruct(con, meta, previous, following, root, previous_point, exclusions):
    root = Path(root)
    work = root / "_work"
    work.mkdir(parents=True, exist_ok=True)

    previous_analysis = None
    if previous_point is not None:
        previous_analysis = {
            key: previous_point[key]
            for key in ("ts_ns", "data_segment_id", "scope_id")
        }

    result = fx.analysis_month(
        con=con,
        meta=meta,
        previous=previous,
        following=following,
        windows=[],
        destination=work / "base.parquet",
        expected_path=work / "expected_keys.parquet",
        previous_analysis=previous_analysis,
        rebuild=True,
    )

    con.register("project_calendar", pa.table({
        "start_ns": pa.array(
            [int(x[0]) for x in exclusions], type=pa.int64()
        ),
        "end_ns": pa.array(
            [int(x[1]) for x in exclusions], type=pa.int64()
        ),
    }))

    destination = root / "quotes" / f"{meta['year_month']}.parquet"
    fx.write_query(con, """
        SELECT q.*,
            (q.bid + q.ask) / 2.0 AS mid,
            q.ask - q.bid AS spread,
            (q.ask - q.bid) / 0.0001 AS spread_pips,
            (
                e.end_ns IS NOT NULL AND q.ts_ns < e.end_ns
            ) AS project_excluded
        FROM actual_quotes q
        ASOF LEFT JOIN project_calendar e
            ON q.ts_ns >= e.start_ns
        ORDER BY q.ts_ns
    """, destination)

    con.read_parquet(
        str(destination), hive_partitioning=False
    ).create_view("points", replace=True)

    con.register("point_boundary", pa.Table.from_pylist(
        [] if previous_point is None else [previous_point],
        schema=pq.read_schema(destination),
    ))
    ddl(con, """
        CREATE OR REPLACE TEMP VIEW point_context AS
        SELECT * FROM points
        UNION ALL
        SELECT * FROM point_boundary
    """)
    ddl(con, """
        CREATE OR REPLACE TEMP VIEW real_links AS
        SELECT
            ts_ns AS a_ns,
            lead(ts_ns) OVER (ORDER BY ts_ns) AS b_ns
        FROM pairs_quotes
    """)

    last_point = fx.record(
        con, "SELECT * FROM points ORDER BY ts_ns DESC LIMIT 1"
    )
    assert result["retained_real_rows"] == meta["observed_rows"]
    fx.verify_parquet(destination, result["analysis_rows"])

    return result, last_point, destination


def derive_tables(con, meta, exclusions, first_ns, last_ns):
    lower, upper = fx.month_bounds(meta["year_month"])

    grid_start = max(lower, ((first_ns + STEP - 1) // STEP) * STEP)
    grid_stop = min(upper, (last_ns // STEP) * STEP + STEP)

    grid = frame(con, f"""
        WITH looked AS (
            SELECT
                g.sample_ns,
                q.ts_ns AS source_ns,
                q.available_ns AS source_available_ns,
                q.is_interpolated,
                q.left_anchor_ns,
                q.right_anchor_ns,
                q.bid, q.ask, q.mid, q.spread,
                q.bid_size, q.ask_size,
                r.a_ns, r.b_ns
            FROM range({grid_start}, {grid_stop}, {STEP}) g(sample_ns)
            ASOF LEFT JOIN point_context q
                ON g.sample_ns >= q.ts_ns
            ASOF LEFT JOIN real_links r
                ON g.sample_ns >= r.a_ns
        ),
        classified AS (
            SELECT *,
                coalesce(
                    sample_ns > a_ns AND b_ns - a_ns > {LIMIT},
                    false
                ) AS in_long_gap,
                CASE
                    WHEN sample_ns = a_ns THEN sample_ns
                    ELSE least(
                        coalesce(b_ns, a_ns + {LIMIT} + 1),
                        a_ns + {LIMIT} + 1
                    )
                END AS status_available_ns
            FROM looked
        ),
        valid AS (
            SELECT *,
                (
                    source_ns IS NOT NULL
                    AND NOT in_long_gap
                    AND sample_ns - source_ns >= 0
                    AND sample_ns - source_ns < {STEP}
                ) AS has_price
            FROM classified
        )
        SELECT
            sample_ns,
            has_price,
            in_long_gap,
            greatest(
                sample_ns,
                coalesce(source_available_ns, sample_ns),
                coalesce(status_available_ns, sample_ns)
            ) AS available_ns,
            status_available_ns,
            CASE WHEN has_price THEN source_ns END AS source_ts_ns,
            CASE WHEN has_price THEN source_available_ns END
                AS source_available_ns,
            CASE WHEN has_price THEN sample_ns - source_ns END
                AS source_age_ns,
            CASE WHEN has_price THEN is_interpolated END
                AS is_interpolated,
            CASE WHEN has_price THEN left_anchor_ns END AS left_anchor_ns,
            CASE WHEN has_price THEN right_anchor_ns END AS right_anchor_ns,
            CASE WHEN has_price THEN bid END AS bid,
            CASE WHEN has_price THEN ask END AS ask,
            CASE WHEN has_price THEN mid END AS mid,
            CASE WHEN has_price THEN spread END AS spread,
            CASE WHEN has_price THEN bid_size END AS bid_size,
            CASE WHEN has_price THEN ask_size END AS ask_size
        FROM valid
        ORDER BY sample_ns
    """)

    allowed, _ = fx.allowed_and_scope(
        grid["sample_ns"].to_numpy(dtype=np.int64), exclusions
    )
    grid["project_excluded"] = ~allowed
    grid["project_eligible"] = (
        grid["has_price"].to_numpy(dtype=bool) & allowed
    )

    bar_start = max(lower, (first_ns // BAR) * BAR)
    bar_stop = min(upper, (last_ns // BAR) * BAR + BAR)

    candles = frame(con, f"""
        WITH aggregated AS (
            SELECT
                (ts_ns // {BAR}) * {BAR} AS candle_start_ns,
                arg_min(mid, ts_ns) AS open_mid,
                max(mid) AS high_mid,
                min(mid) AS low_mid,
                arg_max(mid, ts_ns) AS close_mid,
                count(*)::BIGINT AS n_points,
                count(*) FILTER (WHERE NOT is_interpolated)::BIGINT AS n_real,
                count(*) FILTER (WHERE is_interpolated)::BIGINT AS n_synthetic,
                max(available_ns) AS points_available_ns
            FROM points
            GROUP BY 1
        ),
        clock AS (
            SELECT
                candle_start_ns,
                candle_start_ns + {BAR} AS candle_end_ns
            FROM range({bar_start}, {bar_stop}, {BAR}) t(candle_start_ns)
        )
        SELECT
            c.candle_start_ns,
            c.candle_end_ns,
            a.open_mid, a.high_mid, a.low_mid, a.close_mid,
            coalesce(a.n_points, 0)::BIGINT AS n_points,
            coalesce(a.n_real, 0)::BIGINT AS n_real,
            coalesce(a.n_synthetic, 0)::BIGINT AS n_synthetic,
            a.points_available_ns,
            c.candle_end_ns <= {last_ns} AS completed,
            c.candle_start_ns < {first_ns} AS left_coverage_boundary,
            CASE WHEN c.candle_end_ns <= {last_ns} THEN
                greatest(
                    c.candle_end_ns,
                    coalesce(a.points_available_ns, c.candle_end_ns),
                    coalesce(
                        least(
                            coalesce(r.b_ns, r.a_ns + {LIMIT} + 1),
                            r.a_ns + {LIMIT} + 1
                        ),
                        c.candle_end_ns
                    )
                )
            END AS available_ns
        FROM clock c
        LEFT JOIN aggregated a USING (candle_start_ns)
        ASOF LEFT JOIN real_links r
            ON c.candle_end_ns - 1 >= r.a_ns
        ORDER BY c.candle_start_ns
    """)

    long_context = frame(con, f"""
        SELECT
            left_ns AS a_ns,
            ts_ns AS b_ns,
            ts_ns - left_ns AS gap_ns,
            left_ns + {LIMIT} + 1 AS status_known_ns,
            ts_ns AS interval_available_ns
        FROM pairs_quotes
        WHERE ts_ns - left_ns > {LIMIT}
          AND left_ns < {upper}
          AND ts_ns > {lower}
        ORDER BY left_ns
    """)

    intervals = [
        [int(a), int(b)]
        for a, b in zip(long_context["a_ns"], long_context["b_ns"])
    ]
    starts = candles["candle_start_ns"].to_numpy(dtype=np.int64)
    ends = candles["candle_end_ns"].to_numpy(dtype=np.int64)

    candles["excluded_time_overlap"] = overlap_flags(
        starts, ends, exclusions
    )
    candles["long_gap_overlap"] = overlap_flags(
        starts, ends, intervals
    )
    populated = candles["n_points"].to_numpy(dtype=np.int64) > 0
    completed = candles["completed"].to_numpy(dtype=bool)

    candles["synthetic_only"] = (
        populated
        & (candles["n_real"].to_numpy(dtype=np.int64) == 0)
    )
    candles["ma_eligible"] = populated & completed
    candles["project_eligible"] = (
        populated
        & completed
        & ~candles["excluded_time_overlap"].to_numpy(dtype=bool)
        & ~candles["long_gap_overlap"].to_numpy(dtype=bool)
        & ~candles["left_coverage_boundary"].to_numpy(dtype=bool)
    )

    counts = candles["n_points"].to_numpy(dtype=np.int64)
    synthetic = candles["n_synthetic"].to_numpy(dtype=np.int64)
    candles["synthetic_fraction"] = np.divide(
        synthetic,
        counts,
        out=np.full(len(candles), np.nan),
        where=counts > 0,
    )

    owned_gaps = long_context.loc[
        (long_context["b_ns"] >= lower)
        & (long_context["b_ns"] < upper)
    ].reset_index(drop=True)

    return grid, candles, owned_gaps


def add_averages(candles, state):
    out = candles.copy()
    before = int(state["valid_candles"])
    tail = np.asarray(state["tail_closes"], dtype=float)
    assert len(tail) == min(before, 199)

    if out.empty:
        for n in WINDOWS:
            for column in (
                f"sma_{n}", f"ema_{n}",
                f"close_minus_sma_{n}_pips",
                f"close_minus_ema_{n}_pips",
            ):
                out[column] = pd.Series(index=out.index, dtype=float)
        out["ma_available_ns"] = pd.Series(index=out.index, dtype="Int64")
        out["valid_candle_number"] = pd.Series(index=out.index, dtype="Int64")
        return out, {
            **state,
            "tail_closes": list(state["tail_closes"]),
            "ema_values": dict(state["ema_values"]),
        }

    close = out["close_mid"].to_numpy(dtype=float)
    ends = out["candle_end_ns"].to_numpy(dtype=np.int64)
    availability = out["available_ns"].to_numpy(dtype=np.int64)

    assert np.isfinite(close).all() and (close > 0).all()
    assert (np.diff(ends) > 0).all()
    assert (availability >= ends).all()
    if state["last_candle_end_ns"] is not None:
        assert int(ends[0]) > state["last_candle_end_ns"]

    history = np.concatenate((tail, close))
    series = pd.Series(history, copy=False)
    counts = before + np.arange(1, len(close) + 1, dtype=np.int64)
    ema_after = {}

    for n in WINDOWS:
        sma = series.rolling(n, min_periods=n).mean().to_numpy()[len(tail):]
        old_ema = state["ema_values"][str(n)]
        alpha = 2.0 / (n + 1.0)

        if old_ema is not None:
            assert before >= n
            seeded = np.concatenate(([float(old_ema)], close))
            ema = (
                pd.Series(seeded)
                .ewm(alpha=alpha, adjust=False)
                .mean().to_numpy()[1:]
            )
        else:
            assert before < n
            ema = np.full(len(close), np.nan)
            needed = n - before
            if len(close) >= needed:
                seed_values = np.concatenate((tail, close[:needed]))
                assert len(seed_values) == n
                seed = float(seed_values.mean())
                seeded = np.concatenate(([seed], close[needed:]))
                ema[needed - 1:] = (
                    pd.Series(seeded)
                    .ewm(alpha=alpha, adjust=False)
                    .mean().to_numpy()
                )

        ready = counts >= n
        assert np.array_equal(np.isfinite(sma), ready)
        assert np.array_equal(np.isfinite(ema), ready)

        out[f"sma_{n}"] = sma
        out[f"ema_{n}"] = ema
        out[f"close_minus_sma_{n}_pips"] = (close - sma) / PIP
        out[f"close_minus_ema_{n}_pips"] = (close - ema) / PIP
        ema_after[str(n)] = (
            float(ema[-1]) if np.isfinite(ema[-1]) else old_ema
        )

    dependency = availability.copy()
    if state["dependency_available_ns"] is not None:
        dependency = np.maximum(
            dependency, np.int64(state["dependency_available_ns"])
        )
    dependency = np.maximum.accumulate(dependency)

    out["ma_available_ns"] = pd.array(dependency, dtype="Int64")
    out["valid_candle_number"] = pd.array(counts, dtype="Int64")

    next_state = {
        "valid_candles": int(counts[-1]),
        "tail_closes": history[-199:].tolist(),
        "ema_values": ema_after,
        "dependency_available_ns": int(dependency[-1]),
        "last_candle_end_ns": int(ends[-1]),
    }
    return out, next_state


def process_month(
    meta, previous, following, root, exclusions,
    first_ns, last_ns, state, previous_point,
    memory="2GB", threads=2, spill_limit="20GB"
):
    root = Path(root)
    work = root / "_work"
    spill = work / "spill"
    spill.mkdir(parents=True, exist_ok=True)
    label = meta["year_month"]

    with duckdb.connect(config={
        "memory_limit": memory,
        "threads": threads,
        "temp_directory": str(spill),
        "max_temp_directory_size": spill_limit,
        "preserve_insertion_order": False,
    }) as con:
        reconstruction, last_point, quote_path = reconstruct(
            con, meta, previous, following, root,
            previous_point, exclusions,
        )
        grid, candles, gaps = derive_tables(
            con, meta, exclusions, first_ns, last_ns
        )

        valid = candles.loc[candles["ma_eligible"]].reset_index(drop=True)
        features, state_after = add_averages(valid, state)

        extra_columns = [
            column for column in features
            if column not in candles
        ]
        candles = candles.merge(
            features[["candle_start_ns", *extra_columns]],
            on="candle_start_ns",
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values("candle_start_ns").reset_index(drop=True)

        checks = dict(reconstruction["checks"])
        checks["real_row_difference"] = (
            reconstruction["retained_real_rows"] - meta["observed_rows"]
        )
        checks["candle_point_count_difference"] = (
            int(candles["n_points"].sum()) - reconstruction["analysis_rows"]
        )
        checks["candle_component_count_errors"] = int((
            candles["n_points"]
            != candles["n_real"] + candles["n_synthetic"]
        ).sum())

        populated = candles["n_points"].to_numpy(dtype=np.int64) > 0
        prices = candles.loc[
            populated, ["open_mid", "high_mid", "low_mid", "close_mid"]
        ].to_numpy(dtype=float)
        checks["invalid_ohlc"] = int((
            ~np.isfinite(prices).all(axis=1)
            | (prices[:, 2] > prices[:, 0])
            | (prices[:, 2] > prices[:, 3])
            | (prices[:, 1] < prices[:, 0])
            | (prices[:, 1] < prices[:, 3])
            | (prices[:, 2] > prices[:, 1])
        ).sum())
        checks["prices_in_empty_candles"] = int(
            candles.loc[
                ~populated,
                ["open_mid", "high_mid", "low_mid", "close_mid"],
            ].notna().any(axis=1).sum()
        )

        complete = candles.loc[candles["completed"]]
        checks["candle_availability_before_end"] = int((
            complete["available_ns"] < complete["candle_end_ns"]
        ).sum())
        checks["candle_availability_before_inputs"] = int((
            complete["available_ns"]
            < complete["points_available_ns"]
        ).fillna(False).sum())
        checks["completed_candles_available_after_coverage"] = int((
            complete["available_ns"] > last_ns
        ).sum())

        priced_grid = grid.loc[grid["has_price"]]
        checks["grid_long_gap_fills"] = int((
            grid["has_price"] & grid["in_long_gap"]
        ).sum())
        checks["grid_staleness_errors"] = int((
            (priced_grid["source_age_ns"] < 0)
            | (priced_grid["source_age_ns"] >= STEP)
        ).sum())
        checks["grid_availability_before_time"] = int((
            grid["available_ns"] < grid["sample_ns"]
        ).sum())
        checks["grid_availability_before_source"] = int((
            priced_grid["available_ns"]
            < priced_grid["source_available_ns"]
        ).sum())
        checks["grid_spacing_errors"] = int((
            np.diff(grid["sample_ns"].to_numpy(dtype=np.int64)) != STEP
        ).sum())
        checks["ma_availability_before_candle"] = int((
            features["ma_available_ns"] < features["available_ns"]
        ).sum())
        checks["ma_count_difference"] = (
            state_after["valid_candles"]
            - state["valid_candles"]
            - len(valid)
        )

        assert all(value == 0 for value in checks.values()), checks

        paths = {
            "quotes": quote_path,
            "sampled_10s": root / "sampled_10s" / f"{label}.parquet",
            "candles_3min": root / "candles_3min" / f"{label}.parquet",
            "long_gaps": root / "long_gaps" / f"{label}.parquet",
        }
        save_frame(grid, paths["sampled_10s"])
        save_frame(candles, paths["candles_3min"])
        save_frame(gaps, paths["long_gaps"])

        summary = {
            "year_month": label,
            "real_rows": int(reconstruction["retained_real_rows"]),
            "synthetic_rows": int(reconstruction["synthetic_rows"]),
            "quote_rows": int(reconstruction["analysis_rows"]),
            "sample_rows": len(grid),
            "sample_rows_with_prices": int(grid["has_price"].sum()),
            "candle_rows": len(candles),
            "populated_completed_candles": len(valid),
            "synthetic_only_completed_candles": int((
                candles["synthetic_only"] & candles["completed"]
            ).sum()),
            "empty_candles": int((candles["n_points"] == 0).sum()),
            "project_eligible_candles": int(candles["project_eligible"].sum()),
            "all_hours_long_gap_parents": len(gaps),
            "nonzero_checks": sum(value != 0 for value in checks.values()),
        }

    # The caller publishes the checkpoint only after this returns.
    return {
        "summary": summary,
        "checks": checks,
        "state_after": state_after,
        "last_point": last_point,
        "outputs": {
            name: file_info(path) for name, path in paths.items()
        },
    }
