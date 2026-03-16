"""Render pytest-benchmark JSON as a colored HTML report."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Current benchmark JSON path.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--compare", help="Optional baseline benchmark JSON path.")
    parser.add_argument("--metric", default="median", help="Comparison metric to display.")
    parser.add_argument("--threshold", help="Optional percentage threshold like 8%%.")
    return parser.parse_args()


def load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    abs_value = abs(value)
    if abs_value < 1e-6:
        return f"{value * 1e9:,.4f} ns"
    if abs_value < 1e-3:
        return f"{value * 1e6:,.4f} us"
    if abs_value < 1:
        return f"{value * 1e3:,.4f} ms"
    return f"{value:,.4f} s"


def format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.4f}"


def parse_threshold_percent(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("%"):
        text = text[:-1]
    return float(text)


def compute_delta_percent(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in (None, 0):
        return None
    return ((current - baseline) / baseline) * 100.0


def delta_class(delta_percent: float | None, threshold_percent: float | None) -> str:
    if delta_percent is None:
        return "na"
    if delta_percent <= -2.0:
        return "improved"
    if threshold_percent is not None and delta_percent > threshold_percent:
        return "regressed"
    if delta_percent > 2.0:
        return "slower"
    return "stable"


def metric_direction(metric: str) -> str:
    return "higher" if metric == "ops" else "lower"


def metric_class(value: float | None, best: float | None, metric: str) -> str:
    if value is None or best is None:
        return "na"
    if metric_direction(metric) == "higher":
        return "improved" if value >= best else "regressed"
    return "improved" if value <= best else "regressed"


def metric_ratio(value: float | None, best: float | None) -> str:
    if value is None or best in (None, 0):
        return "-"
    ratio = value / best
    if ratio > 1000:
        return ">1000.0"
    return f"{ratio:.2f}"


def grouped_benchmarks(benchmarks: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for bench in benchmarks:
        group = bench.get("group") or bench["fullname"]
        groups.setdefault(group, []).append(bench)
    return sorted(groups.items(), key=lambda item: item[0])


def benchmark_map(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not data:
        return {}
    return {bench["fullname"]: bench for bench in data.get("benchmarks", [])}


def render_html(
    current_data: dict[str, Any],
    compare_data: dict[str, Any] | None,
    metric: str,
    threshold_percent: float | None,
) -> str:
    current_benchmarks = current_data.get("benchmarks", [])
    compare_by_name = benchmark_map(compare_data)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grouped_sections: list[str] = []
    for group_name, benches in grouped_benchmarks(current_benchmarks):
        metric_bests: dict[str, float | None] = {}
        for metric_name in ("min", "median", "mean", "max", "stddev", "iqr", "ops"):
            values = [bench["stats"].get(metric_name) for bench in benches if bench["stats"].get(metric_name) is not None]
            if not values:
                metric_bests[metric_name] = None
            elif metric_direction(metric_name) == "higher":
                metric_bests[metric_name] = max(values)
            else:
                metric_bests[metric_name] = min(values)

        rows: list[str] = []
        for bench in sorted(benches, key=lambda b: b["fullname"]):
            stats = bench["stats"]
            current_metric = stats.get(metric)
            baseline = compare_by_name.get(bench["fullname"])
            baseline_metric = baseline["stats"].get(metric) if baseline else None
            delta_percent = compute_delta_percent(current_metric, baseline_metric)
            delta_klass = delta_class(delta_percent, threshold_percent)

            def metric_cell(metric_name: str, formatter) -> str:
                value = stats.get(metric_name)
                best = metric_bests.get(metric_name)
                klass = metric_class(value, best, metric_name) if len(benches) > 1 else "stable"
                ratio = metric_ratio(value, best) if len(benches) > 1 else "-"
                return f"<td class='{klass}'>{formatter(value)} <span class='ratio'>({ratio})</span></td>"

            rows.append(
                "<tr>"
                f"<td class='name'>{html.escape(bench['name'])}</td>"
                f"{metric_cell('min', format_seconds)}"
                f"{metric_cell('max', format_seconds)}"
                f"{metric_cell('mean', format_seconds)}"
                f"{metric_cell('stddev', format_seconds)}"
                f"{metric_cell('median', format_seconds)}"
                f"{metric_cell('iqr', format_seconds)}"
                f"{metric_cell('ops', format_number)}"
                f"<td>{format_number(stats.get('rounds'))}</td>"
                f"<td>{format_number(stats.get('iterations'))}</td>"
                f"<td>{format_seconds(baseline_metric)}</td>"
                f"<td class='{delta_klass}'>{'-' if delta_percent is None else f'{delta_percent:+.2f}%'}" "</td>"
                "</tr>"
            )

        grouped_sections.append(
            f"""
    <section class="card table-card">
      <h2>benchmark '{html.escape(group_name)}': {len(benches)} tests</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Min</th>
            <th>Max</th>
            <th>Mean</th>
            <th>StdDev</th>
            <th>Median</th>
            <th>IQR</th>
            <th>OPS</th>
            <th>Rounds</th>
            <th>Iters</th>
            <th>Baseline {html.escape(metric)}</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
"""
        )

    machine = current_data.get("machine_info", {})
    commit = current_data.get("commit_info", {})
    compare_label = html.escape(compare_data.get("datetime", "none")) if compare_data else "none"
    current_label = html.escape(current_data.get("datetime", "unknown"))
    threshold_label = "-" if threshold_percent is None else f"{threshold_percent:.2f}%"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Benchmark Report</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2523;
      --muted: #62706b;
      --grid: #d9d0c1;
      --stable: #f2f0ea;
      --slower: #fff0c7;
      --regressed: #c00;
      --improved: #0c6;
      --na: #f3f3f3;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(135deg, #efe7d4, #f7f4ed 58%, #e7efe8);
      color: var(--ink);
      font: 14px/1.45 "Iosevka", "IBM Plex Mono", "SFMono-Regular", monospace;
    }}
    .wrap {{
      width: max-content;
      min-width: 100%;
      margin: 0 auto;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(61, 67, 63, 0.12);
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(41, 43, 38, 0.08);
      padding: 20px 22px;
      margin-bottom: 18px;
    }}
    .table-card {{
      width: max-content;
      min-width: 100%;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px 18px;
      color: var(--muted);
      margin-top: 14px;
    }}
    .meta strong {{
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
      font-size: 13px;
    }}
    thead th {{
      background: #1f2523;
      color: #f7f4ed;
      position: sticky;
      top: 0;
      text-align: left;
      padding: 10px 12px;
      white-space: nowrap;
    }}
    tbody td {{
      border-top: 1px solid var(--grid);
      padding: 9px 12px;
      white-space: nowrap;
      vertical-align: top;
    }}
    tbody tr:nth-child(even) {{
      background: rgba(224, 214, 195, 0.18);
    }}
    td.name {{
      white-space: normal;
      min-width: 360px;
    }}
    td.stable {{ color: var(--ink); }}
    td.slower {{ color: #6f5600; font-weight: 700; }}
    td.regressed {{ color: var(--regressed); font-weight: 700; }}
    td.improved {{ color: var(--improved); font-weight: 700; }}
    td.na {{ color: var(--muted); }}
    .ratio {{
      color: var(--muted);
      font-weight: 400;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      margin-top: 12px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Benchmark Report</h1>
      <div>Generated {html.escape(now)}</div>
      <div class="meta">
        <div><strong>Current run:</strong> {current_label}</div>
        <div><strong>Baseline run:</strong> {compare_label}</div>
        <div><strong>Compare metric:</strong> {html.escape(metric)}</div>
        <div><strong>Regression threshold:</strong> {threshold_label}</div>
        <div><strong>Machine:</strong> {html.escape(machine.get('node', '-'))}</div>
        <div><strong>Python:</strong> {html.escape(machine.get('python_version', '-'))}</div>
        <div><strong>Commit:</strong> {html.escape(str(commit.get('id', '-')))}</div>
      </div>
      <div class="legend">
        <span class="chip"><span class="swatch" style="background: var(--improved);"></span> improved</span>
        <span class="chip"><span class="swatch" style="background: var(--stable);"></span> stable</span>
        <span class="chip"><span class="swatch" style="background: var(--slower);"></span> slower</span>
        <span class="chip"><span class="swatch" style="background: var(--regressed);"></span> regressed</span>
      </div>
    </section>

    {''.join(grouped_sections)}
  </div>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    current_data = load_json(args.input)
    if current_data is None:
        raise FileNotFoundError(args.input)
    compare_data = load_json(args.compare)
    threshold_percent = parse_threshold_percent(args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(current_data, compare_data, args.metric, threshold_percent),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
