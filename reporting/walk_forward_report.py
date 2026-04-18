"""Generate engineer-facing HTML reports for walk-forward sessions."""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")
    body = "".join(row_html)
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body}</tbody></table>"


def _load_window_payloads(session_folder: str) -> List[Dict[str, Any]]:
    models_dir = os.path.join(session_folder, "models")
    payloads: List[Dict[str, Any]] = []
    if not os.path.isdir(models_dir):
        return payloads

    for window_name in sorted(os.listdir(models_dir)):
        window_folder = os.path.join(models_dir, window_name)
        report_path = os.path.join(window_folder, "report_data.json")
        if not os.path.isfile(report_path):
            continue
        payload = _load_json(report_path)
        payload["window_folder"] = window_folder
        payloads.append(payload)
    return payloads


def _window_summary_rows(payloads: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for payload in payloads:
        metrics = payload["metrics"]
        completed_trades = metrics.get("economic_trade_count", metrics.get("completed_trades", metrics.get("trade_count", 0)))
        rebalances = metrics.get("rebalance_count", metrics.get("trade_count", 0))
        rows.append([
            payload["window"],
            _fmt_pct(metrics.get("return_pct")),
            _fmt_num(metrics.get("sortino_ratio")),
            _fmt_num(metrics.get("calmar_ratio")),
            _fmt_pct(metrics.get("max_drawdown")),
            completed_trades,
            rebalances,
            _fmt_pct(metrics.get("hit_rate")),
            payload.get("best_iteration", "n/a"),
        ])
    return rows


def _build_summary_metrics(payloads: List[Dict[str, Any]], summary_results: Dict[str, Any]) -> List[List[Any]]:
    profitable_windows = sum(1 for payload in payloads if payload["metrics"].get("return_pct", 0.0) > 0)
    all_returns = [payload["metrics"].get("return_pct", 0.0) for payload in payloads]
    all_drawdowns = [payload["metrics"].get("max_drawdown", 0.0) for payload in payloads]
    all_completed_trades = [
        payload["metrics"].get(
            "economic_trade_count",
            payload["metrics"].get("completed_trades", payload["metrics"].get("trade_count", 0)),
        )
        for payload in payloads
    ]
    all_rebalances = [payload["metrics"].get("rebalance_count", payload["metrics"].get("trade_count", 0)) for payload in payloads]
    return [
        ["Windows", summary_results.get("num_windows", len(payloads))],
        ["Profitable windows", f"{profitable_windows}/{len(payloads)}"],
        ["Average return", _fmt_pct(summary_results.get("avg_return"))],
        ["Average Sortino", _fmt_num(summary_results.get("avg_sortino"))],
        ["Average final portfolio", _fmt_money(summary_results.get("avg_portfolio"))],
        ["Average completed trades", _fmt_num(summary_results.get("avg_completed_trades", summary_results.get("avg_trades")))],
        ["Average rebalances", _fmt_num(summary_results.get("avg_rebalances", summary_results.get("avg_trades")))],
        ["Worst window return", _fmt_pct(min(all_returns) if all_returns else None)],
        ["Worst max drawdown", _fmt_pct(min(all_drawdowns) if all_drawdowns else None)],
        ["Max completed trades in a window", max(all_completed_trades) if all_completed_trades else 0],
        ["Max rebalances in a window", max(all_rebalances) if all_rebalances else 0],
    ]


def _aggregate_action_mix(payloads: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = {i: 0 for i in range(7)}
    for payload in payloads:
        for key, value in payload.get("action_counts", {}).items():
            counts[int(key)] = counts.get(int(key), 0) + int(value)
    total = max(1, sum(counts.values()))
    return {
        "LONG": 100.0 * sum(counts.get(i, 0) for i in (0, 1, 2)) / total,
        "SHORT": 100.0 * sum(counts.get(i, 0) for i in (3, 4, 5)) / total,
        "FLAT": 100.0 * counts.get(6, 0) / total,
    }


def _build_engineer_notes(payloads: List[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    collapse_windows = []
    degradation_windows = []
    overtrading_windows = []
    fallback_windows = []

    for payload in payloads:
        window = payload["window"]
        training_iterations = payload.get("training_iterations", [])
        if any(iteration.get("warning_policy_collapse") for iteration in training_iterations):
            collapse_windows.append(window)
        validation_sortino = payload.get("validation_results", {}).get("sortino_ratio")
        test_sortino = payload["metrics"].get("sortino_ratio")
        if validation_sortino is not None and test_sortino is not None and validation_sortino - test_sortino > 2.0:
            degradation_windows.append(window)
        if payload.get("validation_results", {}).get("selected_via_fallback", False):
            fallback_windows.append(window)
        if payload["metrics"].get("rebalance_count", payload["metrics"].get("trade_count", 0)) > 500 and payload["metrics"].get("return_pct", 0.0) <= 0:
            overtrading_windows.append(window)

    returns = [payload["metrics"].get("return_pct", 0.0) for payload in payloads]
    if returns and (max(returns) - min(returns) > 20.0):
        notes.append("Return dispersion across windows is high, which suggests regime sensitivity rather than stable edge.")
    if collapse_windows:
        notes.append(f"Policy-collapse warnings appear in windows {collapse_windows}; inspect action concentration and reward bias.")
    if degradation_windows:
        notes.append(f"Validation-to-test degradation is material in windows {degradation_windows}; check overfitting and tuning leakage.")
    if fallback_windows:
        notes.append(f"Fallback checkpoint selection was required in windows {fallback_windows}; the validation gates likely rejected all candidate checkpoints.")
    if overtrading_windows:
        notes.append(f"High trade-count with weak returns appears in windows {overtrading_windows}; turnover penalty or action stability likely needs work.")
    if not notes:
        notes.append("No major rule-based warnings fired; review per-window charts for subtler degradation patterns.")
    return notes


def _cross_window_figure(payloads: List[Dict[str, Any]]) -> go.Figure:
    windows = [payload["window"] for payload in payloads]
    returns = [payload["metrics"].get("return_pct", 0.0) for payload in payloads]
    sortinos = [payload["metrics"].get("sortino_ratio", 0.0) for payload in payloads]
    drawdowns = [payload["metrics"].get("max_drawdown", 0.0) for payload in payloads]
    trades = [payload["metrics"].get("economic_trade_count", payload["metrics"].get("trade_count", 0)) for payload in payloads]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Return by Window", "Sortino by Window", "Max Drawdown by Window", "Trade Count by Window"),
    )
    fig.add_trace(go.Bar(x=windows, y=returns, name="Return %", marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Bar(x=windows, y=sortinos, name="Sortino", marker_color="#2ca02c"), row=1, col=2)
    fig.add_trace(go.Bar(x=windows, y=drawdowns, name="Max DD %", marker_color="#d62728"), row=2, col=1)
    fig.add_trace(go.Bar(x=windows, y=trades, name="Trades", marker_color="#9467bd"), row=2, col=2)
    fig.update_layout(height=700, title="Cross-Window Diagnostics", showlegend=False)
    return fig


def _training_figure(payload: Dict[str, Any]) -> go.Figure:
    iterations = payload.get("training_iterations", [])
    x = [item.get("iteration", idx) for idx, item in enumerate(iterations)]
    returns = [item.get("return_pct", 0.0) for item in iterations]
    sortinos = [item.get("sortino_ratio", 0.0) for item in iterations]
    completed_trades = [item.get("economic_trade_count", item.get("trade_count", 0)) for item in iterations]
    rebalances = [item.get("rebalance_count", item.get("trade_count", 0)) for item in iterations]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=returns, name="Return %", line=dict(color="#1f77b4", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=sortinos, name="Sortino", line=dict(color="#2ca02c", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=completed_trades, name="Completed Trades", line=dict(color="#d62728", dash="dot")), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=rebalances, name="Rebalances", line=dict(color="#9467bd", dash="dash")), secondary_y=True)
    best_iteration = payload.get("best_iteration")
    if best_iteration is not None:
        fig.add_vline(x=best_iteration, line_dash="dash", line_color="#ff7f0e")
    fig.update_layout(height=360, title=f"Window {payload['window']} Training Diagnostics")
    fig.update_yaxes(title_text="Validation Metric / Return", secondary_y=False)
    fig.update_yaxes(title_text="Trade Events", secondary_y=True)
    return fig


def _replay_figure(payload: Dict[str, Any]) -> go.Figure:
    series = payload["series"]
    price_x = pd.to_datetime(series.get("timestamps", []))
    prices = series.get("prices", [])
    portfolio_x = pd.to_datetime(series.get("portfolio_timestamps", []))
    portfolio_y = series.get("portfolio_values", [])
    drawdown_x = pd.to_datetime(series.get("drawdown_timestamps", []))
    drawdown_y = series.get("drawdown_values", [])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("Price and Trades", "Portfolio Value", "Drawdown"),
    )
    fig.add_trace(go.Scatter(x=price_x, y=prices, name="Price", line=dict(color="#4c566a")), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_x, y=portfolio_y, name="Portfolio", line=dict(color="#5e81ac")), row=2, col=1)
    if drawdown_x is not None and len(drawdown_x) > 0 and len(drawdown_y) > 0:
        fig.add_trace(go.Scatter(x=drawdown_x, y=drawdown_y, name="Drawdown %", line=dict(color="#bf616a")), row=3, col=1)

    trades = pd.DataFrame(payload.get("trade_history", []))
    if not trades.empty and "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"], utc=True, errors="coerce")
        trades = trades.dropna(subset=["date"])
        if "old_contracts" in trades.columns and "new_contracts" in trades.columns:
            old_contracts = pd.to_numeric(trades["old_contracts"], errors="coerce").fillna(0.0)
            new_contracts = pd.to_numeric(trades["new_contracts"], errors="coerce").fillna(0.0)
            old_direction = np.sign(old_contracts)
            new_direction = np.sign(new_contracts)
            old_abs = old_contracts.abs()
            new_abs = new_contracts.abs()
            realized_mask = trades.get("realized_trade", pd.Series(False, index=trades.index)).astype(bool)

            marker_frames = [
                (
                    trades[(new_direction > 0) & ((old_direction <= 0) | (new_abs > old_abs))],
                    "Long Entry / Add",
                    "#2ca02c",
                    "triangle-up",
                ),
                (
                    trades[(new_direction < 0) & ((old_direction >= 0) | (new_abs > old_abs))],
                    "Short Entry / Add",
                    "#d62728",
                    "triangle-down",
                ),
                (
                    trades[realized_mask],
                    "Exit / Reduce",
                    "#ff7f0e",
                    "x",
                ),
            ]
        else:
            trade_type_series = trades.get("trade_type", pd.Series("", index=trades.index)).astype(str).str.lower()
            marker_frames = [
                (trades[trade_type_series.str.contains("long", na=False)], "Long Entry / Add", "#2ca02c", "triangle-up"),
                (trades[trade_type_series.str.contains("short", na=False)], "Short Entry / Add", "#d62728", "triangle-down"),
                (trades[trade_type_series.str.contains("exit|scale out|flat", na=False)], "Exit / Reduce", "#ff7f0e", "x"),
            ]

        for df, name, color, symbol in marker_frames:
            if not df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["price"],
                        name=name,
                        mode="markers",
                        marker=dict(color=color, symbol=symbol, size=9),
                    ),
                    row=1,
                    col=1,
                )

    fig.update_layout(height=850, title=f"Window {payload['window']} Trade Replay")
    return fig

def generate_walk_forward_report(session_folder: str) -> str:
    """Generate an HTML performance report for a completed walk-forward session."""
    reports_dir = os.path.join(session_folder, "reports")
    session_params_path = os.path.join(reports_dir, "session_parameters.json")
    summary_results_path = os.path.join(reports_dir, "summary_results.json")
    if not os.path.isfile(session_params_path) or not os.path.isfile(summary_results_path):
        raise FileNotFoundError("Missing required session report inputs")

    session_params = _load_json(session_params_path)
    summary_results = _load_json(summary_results_path)
    best_hyperparameters = {}
    best_hyperparameters_path = os.path.join(reports_dir, "best_hyperparameters.json")
    if os.path.isfile(best_hyperparameters_path):
        best_hyperparameters = _load_json(best_hyperparameters_path)

    payloads = _load_window_payloads(session_folder)
    if not payloads:
        raise FileNotFoundError("No per-window report_data.json artifacts found")

    summary_table = _table(
        ["Metric", "Value"],
        _build_summary_metrics(payloads, summary_results),
    )
    window_table = _table(
        ["Window", "Return", "Sortino", "Calmar", "Max DD", "Completed Trades", "Rebalances", "Hit Rate", "Best Iter"],
        _window_summary_rows(payloads),
    )
    action_mix = _aggregate_action_mix(payloads)
    action_mix_table = _table(
        ["Action", "Share"],
        [[name, _fmt_pct(value)] for name, value in action_mix.items()],
    )
    hyperparam_rows = [[key, value] for key, value in best_hyperparameters.items()]
    hyperparam_table = _table(["Hyperparameter", "Value"], hyperparam_rows) if hyperparam_rows else "<p>No tuned hyperparameter report found for this session.</p>"
    engineer_notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in _build_engineer_notes(payloads))

    cross_window_fig = _cross_window_figure(payloads)
    window_sections: List[str] = []
    for payload in payloads:
        metrics = payload["metrics"]
        validation_results = payload.get("validation_results", {})
        metrics_table = _table(
            ["Metric", "Value"],
            [
                ["Return", _fmt_pct(metrics.get("return_pct"))],
                ["Final portfolio", _fmt_money(metrics.get("final_portfolio_value"))],
                ["Sortino", _fmt_num(metrics.get("sortino_ratio"))],
                ["Calmar", _fmt_num(metrics.get("calmar_ratio"))],
                ["Max drawdown", _fmt_pct(metrics.get("max_drawdown"))],
                ["Completed trades", metrics.get("economic_trade_count", metrics.get("trade_count", 0))],
                ["Rebalances", metrics.get("rebalance_count", metrics.get("trade_count", 0))],
                ["Hit rate", _fmt_pct(metrics.get("hit_rate"))],
                ["Validation portfolio", _fmt_money(validation_results.get("final_portfolio_value"))],
                ["Validation return", _fmt_pct(validation_results.get("total_return_pct"))],
                ["Validation Sortino", _fmt_num(validation_results.get("sortino_ratio"))],
                ["Validation Calmar", _fmt_num(validation_results.get("calmar_ratio"))],
                ["Validation max drawdown", _fmt_pct(validation_results.get("max_drawdown"))],
            ],
        )
        fallback_note = ""
        if validation_results.get("selected_via_fallback", False):
            fallback_note = "<div class=\"note\">Validation checkpoint selection fell back to the least-bad candidate because all checkpoints failed the configured trade/action/drawdown gates.</div>"
        training_fig = _training_figure(payload)
        replay_fig = _replay_figure(payload)
        flag_rows = []
        for iteration in payload.get("training_iterations", []):
            flags = iteration.get("collapse_flags", [])
            if flags:
                flag_rows.append([iteration.get("iteration"), ", ".join(flags), _fmt_num(iteration.get("sortino_ratio"))])
        flags_html = _table(["Iteration", "Flags", "Sortino"], flag_rows) if flag_rows else "<p>No collapse flags were recorded for this window.</p>"
        window_sections.append(
            f"""
            <section class="window">
              <h2>Window {payload['window']}</h2>
              {metrics_table}
              {fallback_note}
              <div class="note">Train/Val/Test: {html.escape(str(payload['window_periods'].get('train_start')))} to {html.escape(str(payload['window_periods'].get('test_end')))}</div>
              {replay_fig.to_html(full_html=False, include_plotlyjs=False)}
              {training_fig.to_html(full_html=False, include_plotlyjs=False)}
              <h3>Collapse Flags</h3>
              {flags_html}
            </section>
            """
        )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Walk-Forward Report {html.escape(session_params.get('timestamp', ''))}</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 0; background: linear-gradient(180deg, #f5f1e8 0%, #efe6d5 100%); color: #1e1d1b; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 32px 28px 56px; }}
    h1, h2, h3 {{ font-family: 'Palatino Linotype', 'Book Antiqua', serif; }}
    h1 {{ margin-bottom: 8px; }}
    section {{ background: rgba(255, 252, 247, 0.86); border: 1px solid #d8c8ae; border-radius: 18px; padding: 20px 22px; margin: 20px 0; box-shadow: 0 10px 35px rgba(67, 53, 33, 0.08); }}
    .meta {{ color: #5a5245; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e7dbc8; text-align: left; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f4ead8; }}
    ul {{ margin: 10px 0 0 18px; }}
    .note {{ color: #5a5245; margin-top: 10px; }}
    .window {{ padding-bottom: 28px; }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 18px 14px 40px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Walk-Forward Performance Report</h1>
    <div class="meta">
      Session: {html.escape(session_folder)}<br>
      Timestamp: {html.escape(str(session_params.get('timestamp', 'n/a')))}<br>
      Data range: {html.escape(str(session_params.get('data_start', 'n/a')))} to {html.escape(str(session_params.get('data_end', 'n/a')))}<br>
      Evaluation metric: {html.escape(str(session_params.get('evaluation_metric', 'return')))}
    </div>

    <section>
      <h2>Session Summary</h2>
      <div class="grid">
        <div>
          {summary_table}
        </div>
        <div>
          <h3>Aggregate Action Mix</h3>
          {action_mix_table}
          <h3>Engineer Notes</h3>
          <ul>{engineer_notes_html}</ul>
        </div>
      </div>
    </section>

    <section>
      <h2>Window Comparison</h2>
      {window_table}
      {cross_window_fig.to_html(full_html=False, include_plotlyjs='inline')}
    </section>

    <section>
      <h2>Tuned Hyperparameters</h2>
      {hyperparam_table}
    </section>

    {''.join(window_sections)}
  </main>
</body>
</html>
"""

    output_path = os.path.join(reports_dir, "walk_forward_report.html")
    with open(output_path, "w") as f:
        f.write(report_html)
    return output_path
