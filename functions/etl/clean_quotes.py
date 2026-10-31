
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

TZ = "Europe/Athens"
NY = "America/New_York"
STEP_NS = 10_000_000_000
LIMIT_NS = 300_000_000_000
BUCKETS = []

VALUES = ["bid", "ask", "bid_size", "ask_size"]

def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    temporary.replace(path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def literal(value):
    return "'" + str(value).replace("'", "''") + "'"

def write_query(con, query, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.parquet")
    temporary.unlink(missing_ok=True)

    con.execute(
        f"COPY ({query}) TO {literal(temporary)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.complete()
    temporary.replace(path)

def record(con, query):
    cursor = con.execute(query)
    names = [item[0] for item in cursor.description]
    rows = cursor.fetchall()

    assert len(rows) <= 1, "record() requires a single-row query."
    return None if not rows else dict(zip(names, rows[0]))

def verify_parquet(path, rows):
    assert Path(path).is_file(), f"Missing output: {path}"
    actual = pq.ParquetFile(path).metadata.num_rows
    assert actual == rows, f"Row count changed: {path}"

def athens(value):
    if value is None:
        return None
    return pd.Timestamp(
        int(value), unit="ns", tz="UTC"
    ).tz_convert(TZ).isoformat()

def month_bounds(label):
    start = pd.Timestamp(f"{label}-01", tz=TZ)
    end = start + pd.offsets.MonthBegin(1)
    return int(start.value), int(end.value)

def histogram(con, relation):
    counts = [0] * len(BUCKETS)
    clauses = " ".join(
        f"WHEN gap_ns <= {upper} THEN {index}"
        for index, upper in BUCKETS
    )
    rows = con.execute(
        f"SELECT CASE {clauses} END AS bucket_index, count(*) "
        f"FROM {relation} WHERE gap_ns > 0 GROUP BY 1"
    ).fetchall()
    for index, count in rows:
        assert index is not None
        counts[index] = count
    return counts


def merge_windows(rows):
    merged = []
    for start, end in sorted(rows):
        start, end = int(start), int(end)
        assert end > start
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

def allowed_and_scope(times, windows):
    times = np.asarray(times, dtype=np.int64)
    if not windows:
        return np.ones(len(times), dtype=bool), np.zeros(len(times), dtype=np.int64)
    starts = np.asarray([x[0] for x in windows], dtype=np.int64)
    ends = np.asarray([x[1] for x in windows], dtype=np.int64)
    previous = np.searchsorted(starts, times, side="right") - 1
    inside = (previous >= 0) & (times < ends[np.maximum(previous, 0)])
    scope = np.searchsorted(ends, times, side="right").astype(np.int64)
    return ~inside, scope

def make_windows(first_ns, last_ns, verified_path):
    first = pd.Timestamp(first_ns, unit="ns", tz="UTC").tz_convert(NY).date()
    last = pd.Timestamp(last_ns, unit="ns", tz="UTC").tz_convert(NY).date()
    day, stop = first - timedelta(days=10), last + timedelta(days=10)
    details = []

    def add(start, end, kind):
        details.append({
            "start_ns": int(start.value),
            "end_ns": int(end.value),
            "kind": kind
        })

    while day <= stop:
        if day.weekday() == 4:
            start = pd.Timestamp(f"{day} 17:00", tz=NY)
            end = pd.Timestamp(f"{day + timedelta(days=2)} 17:00", tz=NY)
            add(start, end, "expected_weekend")
        if day.weekday() in (0, 1, 2, 3, 4, 6):
            add(
                pd.Timestamp(f"{day} 16:00", tz=NY),
                pd.Timestamp(f"{day} 19:00", tz=NY),
                "research_daily_exclusion"
            )
        day += timedelta(days=1)

    for year in range(first.year - 1, last.year + 2):
        add(
            pd.Timestamp(f"{year}-01-01", tz=TZ),
            pd.Timestamp(f"{year}-01-02", tz=TZ),
            "january_1_exclusion"
        )

    verified = pd.read_csv(verified_path)
    for row in verified.to_dict("records"):
        start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
        assert start.tzinfo is not None and end.tzinfo is not None
        add(start, end, "verified_quote_closure")

    windows = merge_windows(
        (row["start_ns"], row["end_ns"]) for row in details
    )
    return windows, details

def allowed_pieces(start, end, windows):
    current = int(start)
    for left, right in windows:
        if right <= current:
            continue
        if left >= end:
            break
        if left > current:
            yield current, min(left, end)
        current = max(current, right)
        if current >= end:
            return
    if current < end:
        yield current, int(end)

def split_by_month(start, end):
    current = int(start)
    while current < end:
        local = pd.Timestamp(current, unit="ns", tz="UTC").tz_convert(TZ)
        label = f"{local.year:04d}-{local.month:02d}"
        _, boundary = month_bounds(label)
        piece_end = min(int(end), boundary)
        yield label, current, piece_end
        current = piece_end


def stage_source(repo, source, directory, block_bytes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "source_complete.json"

    path = Path(repo) / source["path"]
    assert path.stat().st_size == source["bytes"]
    assert path.stat().st_mtime_ns == source["mtime_ns"]

    if done_path.exists():
        done = load_json(done_path)
        for item in done["files"]:
            verify_parquet(directory / item["name"], item["rows"])
        return done

    types = {
        "Time (EET)": pa.string(),
        "Ask": pa.float64(), "Bid": pa.float64(),
        "AskVolume": pa.float64(), "BidVolume": pa.float64()
    }
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(
            block_size=block_bytes, use_threads=False
        ),
        convert_options=pacsv.ConvertOptions(column_types=types)
    )
    assert reader.schema.names == list(types), f"Unexpected columns: {path}"

    writers, output_rows = {}, {}
    rows = locked = both_zero = 0
    first_ns = last_ns = previous_ns = None

    try:
        for batch in reader:
            count = batch.num_rows
            if not count:
                continue

            try:
                local = pd.DatetimeIndex(pd.to_datetime(
                    batch.column("Time (EET)").to_pylist(),
                    format="%Y.%m.%d %H:%M:%S.%f",
                    errors="raise"
                ))
                assert not local.hasnans
                local = local.tz_localize(
                    TZ, ambiguous="raise", nonexistent="raise"
                )
                ticks = local.tz_convert("UTC").as_unit("ns").asi8
            except Exception as error:
                batch.slice(0, min(count, 100)).to_pandas().to_csv(
                    directory / "timestamp_error_sample.csv", index=False
                )
                raise RuntimeError(
                    f"Timestamp review required in {path.name}; "
                    f"parsed records {rows + 1}–{rows + count}."
                ) from error

            values = {
                "bid": batch.column("Bid").to_numpy(zero_copy_only=False),
                "ask": batch.column("Ask").to_numpy(zero_copy_only=False),
                "bid_size": batch.column("BidVolume").to_numpy(zero_copy_only=False),
                "ask_size": batch.column("AskVolume").to_numpy(zero_copy_only=False)
            }
            invalid = np.zeros(count, dtype=bool)
            for array in values.values():
                invalid |= ~np.isfinite(array)
            invalid |= values["bid"] <= 0
            invalid |= values["ask"] <= 0
            invalid |= values["ask"] < values["bid"]
            invalid |= values["bid_size"] < 0
            invalid |= values["ask_size"] < 0

            if invalid.any():
                positions = np.flatnonzero(invalid)[:50]
                preview = batch.take(pa.array(positions)).to_pandas()
                preview["source_record"] = rows + positions + 1
                preview.to_csv(directory / "invalid_quote_examples.csv", index=False)
                raise RuntimeError(
                    f"Invalid quotes in {path.name}; review saved examples."
                )

            backward = np.any(ticks[1:] < ticks[:-1])
            backward |= previous_ns is not None and ticks[0] < previous_ns
            if backward:
                raise RuntimeError(f"Backward UTC timestamps require review: {path.name}")

            table = pa.table({
                "ts_ns": pa.array(ticks, type=pa.int64()),
                **{name: pa.array(array, type=pa.float64())
                   for name, array in values.items()},
                "source_id": pa.array(
                    np.full(count, source["source_id"], dtype=np.int32)
                ),
                "source_record": pa.array(
                    np.arange(rows + 1, rows + count + 1, dtype=np.int64)
                )
            })

            labels = np.asarray(local.strftime("%Y-%m"))
            for label in np.unique(labels):
                positions = np.flatnonzero(labels == label)
                if label not in writers:
                    writers[label] = pq.ParquetWriter(
                        directory / f"{label}.parquet",
                        table.schema,
                        compression="zstd"
                    )
                    output_rows[label] = 0
                writers[label].write_table(table.take(pa.array(positions)))
                output_rows[label] += len(positions)

            locked += int(np.count_nonzero(values["ask"] == values["bid"]))
            both_zero += int(np.count_nonzero(
                (values["bid_size"] == 0) & (values["ask_size"] == 0)
            ))
            first_ns = int(ticks[0]) if first_ns is None else first_ns
            previous_ns = last_ns = int(ticks[-1])
            rows += count
    finally:
        for writer in writers.values():
            writer.close()
        reader.close()

    assert rows > 0, f"Empty input requires review: {path}"
    assert path.stat().st_size == source["bytes"], "Source changed during processing."
    assert path.stat().st_mtime_ns == source["mtime_ns"], "Source changed during processing."

    files = [
        {"name": f"{label}.parquet", "year_month": label, "rows": int(count)}
        for label, count in sorted(output_rows.items())
    ]
    assert sum(item["rows"] for item in files) == rows

    done = {
        **source, "rows": rows,
        "first_ns": first_ns, "last_ns": last_ns,
        "locked_rows": locked, "both_sizes_zero_rows": both_zero,
        "files": files
    }
    dump_json(done_path, done)
    return done


def merge_month(con, files, destination, previous):
    con.read_parquet(
        [str(path) for path in files], hive_partitioning=False
    ).create_view("raw_month", replace=True)

    raw_rows = con.execute("SELECT count(*) FROM raw_month").fetchall()[0][0]

    con.execute("""
        CREATE OR REPLACE TEMP TABLE unique_month AS
        SELECT
            ts_ns, bid, ask, bid_size, ask_size,
            count(*)::BIGINT AS source_copies,
            arg_min(
                source_id,
                struct_pack(s := source_id, r := source_record)
            ) AS representative_source_id,
            arg_min(
                source_record,
                struct_pack(s := source_id, r := source_record)
            ) AS representative_source_record
        FROM raw_month
        GROUP BY ts_ns, bid, ask, bid_size, ask_size
    """)

    conflicts = con.execute("""
        SELECT count(*) FROM (
            SELECT ts_ns FROM unique_month
            GROUP BY ts_ns HAVING count(*) > 1
        )
    """).fetchall()[0][0]

    if conflicts:
        evidence = Path(destination).with_suffix(".conflicts.parquet")
        write_query(con, """
            SELECT * FROM unique_month
            WHERE ts_ns IN (
                SELECT ts_ns FROM unique_month
                GROUP BY ts_ns HAVING count(*) > 1
            )
            ORDER BY ts_ns, representative_source_id
        """, evidence)
        raise RuntimeError(
            f"{conflicts} conflicting timestamps. Review {evidence}"
        )

    previous_ns = "NULL" if previous is None else str(previous["ts_ns"])
    segment_offset = 0 if previous is None else previous["data_segment_id"]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ordered_month AS
        SELECT *,
            lag(ts_ns, 1, {previous_ns}) OVER (ORDER BY ts_ns) AS a_ns
        FROM unique_month
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prepared_month AS
        SELECT *,
            ts_ns - a_ns AS gap_ns,
            CAST(({segment_offset} + sum(
                CASE WHEN ts_ns - a_ns > {LIMIT_NS} THEN 1 ELSE 0 END
            ) OVER (
                ORDER BY ts_ns ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )) AS BIGINT) AS data_segment_id
        FROM ordered_month
    """)

    write_query(
        con, "SELECT * FROM prepared_month ORDER BY ts_ns", destination
    )
    con.read_parquet(str(destination), hive_partitioning=False).create_view(
        "saved_observed", replace=True
    )

    fields = "ts_ns, bid, ask, bid_size, ask_size"
    differences = con.execute(f"""
        SELECT count(*) FROM (
            (SELECT {fields} FROM unique_month
             EXCEPT ALL SELECT {fields} FROM saved_observed)
            UNION ALL
            (SELECT {fields} FROM saved_observed
             EXCEPT ALL SELECT {fields} FROM unique_month)
        )
    """).fetchall()[0][0]
    assert differences == 0, "Saved observed values do not reconcile."

    result = record(con, """
        SELECT
            count(*) AS observed_rows,
            sum(source_copies)::BIGINT AS represented_source_rows,
            count(*) FILTER (WHERE ask = bid) AS locked_rows,
            count(*) FILTER (WHERE bid_size = 0 AND ask_size = 0) AS both_sizes_zero_rows,
            count(*) FILTER (WHERE gap_ns > 0) AS positive_intervals,
            count(*) FILTER (WHERE gap_ns <= 0) AS nonpositive_intervals
        FROM saved_observed
    """)
    assert result["represented_source_rows"] == raw_rows
    assert result["nonpositive_intervals"] == 0

    result.update({
        "path": str(destination),
        "raw_rows": raw_rows,
        "duplicate_copies_removed": raw_rows - result["observed_rows"],
        "first": record(con, "SELECT * FROM saved_observed ORDER BY ts_ns LIMIT 1"),
        "last": record(con, "SELECT * FROM saved_observed ORDER BY ts_ns DESC LIMIT 1")
    })
    return result


KEYS = [
    "ts_ns", "left_anchor_ns", "right_anchor_ns",
    "available_ns", "data_segment_id", "scope_id"
]
KEY_SCHEMA = pa.schema([(name, pa.int64()) for name in KEYS])

def prepare_context(con, meta, previous, following, windows):
    path = meta["path"]
    con.read_parquet(path, hive_partitioning=False).create_view(
        "cur_quotes", replace=True
    )
    boundary = [row for row in (previous, following) if row is not None]
    con.register(
        "real_boundary",
        pa.Table.from_pylist(boundary, schema=pq.read_schema(path))
    )
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pairs_quotes AS
        SELECT *,
            lag(ts_ns) OVER w AS left_ns,
            lag(bid) OVER w AS left_bid,
            lag(ask) OVER w AS left_ask,
            lag(bid_size) OVER w AS left_bid_size,
            lag(ask_size) OVER w AS left_ask_size
        FROM (
            SELECT * FROM cur_quotes
            UNION ALL SELECT * FROM real_boundary
        )
        WINDOW w AS (ORDER BY ts_ns)
    """)

    bad_adjacency = con.execute("""
        SELECT count(*) FROM pairs_quotes
        WHERE left_ns IS NOT NULL AND a_ns IS DISTINCT FROM left_ns
    """).fetchall()[0][0]
    assert bad_adjacency == 0, "Cross-partition real adjacency mismatch."

    con.register("exclusions", pa.table({
        "start_ns": pa.array([x[0] for x in windows], type=pa.int64()),
        "end_ns": pa.array([x[1] for x in windows], type=pa.int64()),
        "scope_id": pa.array(
            np.arange(1, len(windows) + 1, dtype=np.int64)
        )
    }))

def scope_sql(query):
    return f"""
        SELECT q.*, coalesce(e.scope_id, 0)::BIGINT AS scope_id
        FROM ({query}) q
        ASOF LEFT JOIN exclusions e ON q.ts_ns >= e.start_ns
        WHERE e.end_ns IS NULL OR q.ts_ns >= e.end_ns
    """

def write_expected_keys(con, path, lower, upper, windows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    query = f"""
        SELECT left_ns, ts_ns, data_segment_id
        FROM pairs_quotes
        WHERE ts_ns - left_ns > {STEP_NS}
          AND ts_ns - left_ns <= {LIMIT_NS}
          AND left_ns < {upper}
          AND ts_ns > {lower}
    """

    reader = con.sql(query).to_arrow_reader(batch_size=10_000)

    try:
        with pq.ParquetWriter(
            path, KEY_SCHEMA, compression="zstd"
        ) as writer:
            for batch in reader:
                a = batch.column(0).to_numpy()
                b = batch.column(1).to_numpy()
                segment = batch.column(2).to_numpy()

                counts = ((b - a - 1) // STEP_NS).astype(np.int64)
                total = int(counts.sum())
                if not total:
                    continue

                starts = np.cumsum(counts) - counts
                k = (
                    np.arange(total, dtype=np.int64)
                    - np.repeat(starts, counts)
                    + 1
                )

                left = np.repeat(a, counts)
                right = np.repeat(b, counts)
                ticks = left + k * STEP_NS

                allowed, scope = allowed_and_scope(ticks, windows)
                keep = allowed & (ticks >= lower) & (ticks < upper)

                arrays = [
                    ticks[keep],
                    left[keep],
                    right[keep],
                    right[keep],
                    np.repeat(segment, counts)[keep],
                    scope[keep]
                ]

                writer.write_table(pa.Table.from_arrays(
                    [
                        pa.array(array, type=pa.int64())
                        for array in arrays
                    ],
                    schema=KEY_SCHEMA
                ))
    finally:
        reader.close()

def analysis_month(
    con, meta, previous, following, windows,
    destination, expected_path, previous_analysis, rebuild
):
    prepare_context(con, meta, previous, following, windows)
    lower, upper = month_bounds(meta["year_month"])

    real = """
        SELECT
            ts_ns, ts_ns AS available_ns,
            bid, ask, bid_size, ask_size,
            false AS is_interpolated,
            ts_ns AS left_anchor_ns, ts_ns AS right_anchor_ns,
            data_segment_id,
            representative_source_id, representative_source_record,
            source_copies
        FROM cur_quotes
    """

    interpolated_values = ", ".join(
        f"left_{name} + ({name} - left_{name}) * "
        f"((k * {STEP_NS})::DOUBLE / (ts_ns - left_ns)::DOUBLE) AS {name}"
        for name in VALUES
    )
    synthetic = f"""
        SELECT
            left_ns + k * {STEP_NS} AS ts_ns,
            ts_ns AS available_ns,
            {interpolated_values},
            true AS is_interpolated,
            left_ns AS left_anchor_ns,
            ts_ns AS right_anchor_ns,
            data_segment_id,
            NULL::INTEGER AS representative_source_id,
            NULL::BIGINT AS representative_source_record,
            0::BIGINT AS source_copies
        FROM pairs_quotes,
        LATERAL range(
            1::BIGINT,
            ((ts_ns - left_ns - 1) // {STEP_NS} + 1)::BIGINT
        ) generated(k)
        WHERE ts_ns - left_ns > {STEP_NS}
          AND ts_ns - left_ns <= {LIMIT_NS}
    """

    candidates = f"""
        SELECT * FROM ({real} UNION ALL {synthetic})
        WHERE ts_ns >= {lower} AND ts_ns < {upper}
    """
    scoped = scope_sql(candidates)

    if rebuild:
        write_query(con, f"""
            SELECT *,
                ask = bid AS locked_quote,
                bid_size = 0 AND ask_size = 0 AS both_sizes_zero
            FROM ({scoped})
            ORDER BY ts_ns
        """, destination)

    write_expected_keys(con, expected_path, lower, upper, windows)
    con.read_parquet(str(destination), hive_partitioning=False).create_view(
        "actual_quotes", replace=True
    )
    con.read_parquet(str(expected_path), hive_partitioning=False).create_view(
        "expected_keys", replace=True
    )

    key_sql = ", ".join(KEYS)
    checks = {}

    checks["missing_required_fill_points"] = con.execute(f"""
        SELECT count(*) FROM (
            SELECT {key_sql} FROM expected_keys
            EXCEPT ALL
            SELECT {key_sql} FROM actual_quotes WHERE is_interpolated
        )
    """).fetchall()[0][0]

    checks["unexpected_synthetic_points"] = con.execute(f"""
        SELECT count(*) FROM (
            SELECT {key_sql} FROM actual_quotes WHERE is_interpolated
            EXCEPT ALL SELECT {key_sql} FROM expected_keys
        )
    """).fetchall()[0][0]

    checks["duplicate_analysis_timestamps"] = con.execute("""
        SELECT count(*) - count(DISTINCT ts_ns) FROM actual_quotes
    """).fetchall()[0][0]

    checks["rows_inside_exclusions"] = con.execute("""
        SELECT count(*)
        FROM actual_quotes q
        ASOF LEFT JOIN exclusions e ON q.ts_ns >= e.start_ns
        WHERE q.ts_ns < e.end_ns
    """).fetchall()[0][0]

    checks["invalid_analysis_values"] = con.execute("""
        SELECT count(*) FROM actual_quotes
        WHERE NOT isfinite(bid) OR NOT isfinite(ask)
           OR NOT isfinite(bid_size) OR NOT isfinite(ask_size)
           OR bid <= 0 OR ask < bid OR bid_size < 0 OR ask_size < 0
           OR available_ns < ts_ns
           OR (NOT is_interpolated AND available_ns <> ts_ns)
    """).fetchall()[0][0]

    expected_real = scope_sql(real)
    columns = [
        "ts_ns", "available_ns", *VALUES, "is_interpolated",
        "left_anchor_ns", "right_anchor_ns", "data_segment_id",
        "representative_source_id", "representative_source_record",
        "source_copies", "scope_id"
    ]
    columns_sql = ", ".join(columns)
    checks["changed_or_missing_real_rows"] = con.execute(f"""
        SELECT count(*) FROM (
            (SELECT {columns_sql} FROM ({expected_real})
             EXCEPT ALL
             SELECT {columns_sql} FROM actual_quotes WHERE NOT is_interpolated)
            UNION ALL
            (SELECT {columns_sql} FROM actual_quotes WHERE NOT is_interpolated
             EXCEPT ALL
             SELECT {columns_sql} FROM ({expected_real}))
        )
    """).fetchall()[0][0]

    value_errors = []
    for name in VALUES:
        expected_value = (
            f"p.left_{name} + (p.{name} - p.left_{name}) * "
            "((q.ts_ns - p.left_ns)::DOUBLE / "
            "(p.ts_ns - p.left_ns)::DOUBLE)"
        )
        value_errors.append(
            f"abs(q.{name} - ({expected_value})) > "
            f"1e-12 * greatest(1.0, abs({expected_value}))"
        )

    checks["interpolation_value_errors"] = con.execute(f"""
        SELECT count(*)
        FROM actual_quotes q
        JOIN pairs_quotes p
          ON q.right_anchor_ns = p.ts_ns
         AND q.left_anchor_ns = p.left_ns
        WHERE q.is_interpolated AND (
            q.available_ns <> p.ts_ns
            OR p.ts_ns - p.left_ns > {LIMIT_NS}
            OR q.ts_ns <= p.left_ns OR q.ts_ns >= p.ts_ns
            OR {" OR ".join(value_errors)}
        )
    """).fetchall()[0][0]

    boundary_schema = pa.schema([
        ("ts_ns", pa.int64()),
        ("data_segment_id", pa.int64()),
        ("scope_id", pa.int64())
    ])
    con.register("analysis_boundary", pa.Table.from_pylist(
        [] if previous_analysis is None else [previous_analysis],
        schema=boundary_schema
    ))
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW analysis_spacings AS
        WITH combined AS (
            SELECT ts_ns, data_segment_id, scope_id FROM actual_quotes
            UNION ALL SELECT * FROM analysis_boundary
        ), ordered AS (
            SELECT *,
                lag(ts_ns) OVER w AS previous_ns,
                lag(data_segment_id) OVER w AS previous_segment,
                lag(scope_id) OVER w AS previous_scope
            FROM combined WINDOW w AS (ORDER BY ts_ns)
        )
        SELECT ts_ns - previous_ns AS gap_ns
        FROM ordered
        WHERE ts_ns >= {lower}
          AND data_segment_id = previous_segment
          AND scope_id = previous_scope
    """)

    checks["unresolved_short_gap_instances"] = con.execute(f"""
        SELECT count(*) FROM analysis_spacings
        WHERE gap_ns > {STEP_NS} AND gap_ns <= {LIMIT_NS}
    """).fetchall()[0][0]
    checks["long_spacing_inside_continuous_segment"] = con.execute(f"""
        SELECT count(*) FROM analysis_spacings WHERE gap_ns > {LIMIT_NS}
    """).fetchall()[0][0]
    checks["nonpositive_analysis_spacing"] = con.execute("""
        SELECT count(*) FROM analysis_spacings WHERE gap_ns <= 0
    """).fetchall()[0][0]

    result = record(con, """
        SELECT
            count(*) AS analysis_rows,
            count(*) FILTER (WHERE is_interpolated) AS synthetic_rows,
            count(*) FILTER (WHERE NOT is_interpolated) AS retained_real_rows
        FROM actual_quotes
    """)
    result.update({
        "year_month": meta["year_month"],
        "path": str(destination),
        "expected_fill_points": con.execute(
            "SELECT count(*) FROM expected_keys"
        ).fetchall()[0][0],
        "checks": checks,
        "raw_interval_histogram": histogram(con, "cur_quotes"),
        "analysis_spacing_histogram": histogram(con, "analysis_spacings"),
        "last": record(con, """
            SELECT ts_ns, data_segment_id, scope_id
            FROM actual_quotes ORDER BY ts_ns DESC LIMIT 1
        """)
    })

    assert all(value == 0 for value in checks.values()), checks
    assert result["synthetic_rows"] == result["expected_fill_points"]
    verify_parquet(destination, result["analysis_rows"])
    return result
