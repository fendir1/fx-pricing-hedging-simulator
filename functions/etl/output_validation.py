
import json
from html import escape
from pathlib import Path

import pandas as pd

STYLE = """
<style>
.rv-section {font-family:Arial,sans-serif;color:#17212b;background:white;
             padding:14px;margin:12px 0;border:1px solid #d6dee6}
.rv-section h2 {margin-top:0}
.rv-section p {line-height:1.45}
.rv-scroll {overflow:auto;max-height:520px;margin:10px 0}
.rv-section table {border-collapse:collapse;font-size:13px;width:100%}
.rv-section th,.rv-section td {border:1px solid #d6dee6;padding:6px;
                              text-align:left;vertical-align:top}
.rv-section th {background:#edf2f7;position:sticky;top:0}
.rv-section td {white-space:nowrap}
.rv-section details {margin:12px 0}
.rv-section summary {cursor:pointer;font-weight:bold}
</style>
"""

def times(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(
            result[column], unit="ns", utc=True
        ).dt.tz_convert("Europe/Athens").map(
            lambda value: "" if pd.isna(value) else value.isoformat()
        )
        result = result.rename(columns={
            column: column.removesuffix("_ns") + "_athens"
        })
    return result

def section(title, status, notes, tables):
    parts = [
        '<section class="rv-section">',
        f"<h2>{escape(title)}</h2>",
        f"<p><strong>Status:</strong> {escape(status)}</p>"
    ]
    parts.extend(f"<p>{escape(note)}</p>" for note in notes)

    for label, frame in tables:
        parts.append(
            f"<details {'open' if len(frame) <= 25 else ''}>"
            f"<summary>{escape(label)} — {len(frame):,} rows</summary>"
        )
        if frame.empty:
            parts.append("<p>No records.</p>")
        else:
            parts.append('<div class="rv-scroll">')
            parts.append(frame.to_html(
                index=False,
                escape=True,
                render_links=True,
                border=0,
                float_format=lambda value: f"{value:.9g}"
            ))
            parts.append("</div>")
        parts.append("</details>")

    parts.append("</section>")
    return "".join(parts)

def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    temporary.replace(path)

def upsert_markdown(path, key, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {key}:start -->"
    end = f"<!-- {key}:end -->"
    block = f"{start}\n{body.strip()}\n{end}"

    if start in text:
        left = text.index(start)
        right = text.index(end, left) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"

    path.write_text(text, encoding="utf-8")
