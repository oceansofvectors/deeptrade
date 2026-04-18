import copy
import csv
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from walk_forward import load_tradingview_data, walk_forward_testing


ARTIFACTS_DIR = ROOT / "artifacts"
SCENARIOS_PATH = ROOT / "scenarios.yaml"

SMOKE_BUDGET = {
    "window_size": 120,
    "step_size": 24,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "embargo_days": 1,
    "initial_timesteps": 2000,
    "additional_timesteps": 1000,
    "max_iterations": 1,
    "n_stagnant_loops": 1,
    "improvement_threshold": 0.01,
    "run_hyperparameter_tuning": False,
    "tuning_trials": 0,
    "max_windows": 1,
}

VALIDATION_BUDGET = {
    "window_size": 120,
    "step_size": 24,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "embargo_days": 1,
    "initial_timesteps": 5000,
    "additional_timesteps": 2000,
    "max_iterations": 2,
    "n_stagnant_loops": 2,
    "improvement_threshold": 0.01,
    "run_hyperparameter_tuning": False,
    "tuning_trials": 0,
    "max_windows": 2,
}

SCENARIO_ORDER = [
    ("default_full_cycle", None),
    ("btc_recovery_2023", "btc_recovery_2023"),
    ("btc_post_run_chop_2024", "btc_post_run_chop_2024"),
    ("btc_bear_market_2022", "btc_bear_market_2022"),
]


def _json_default(value):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def load_scenarios() -> dict:
    with SCENARIOS_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("scenarios", {})


def filter_for_scenario(data: pd.DataFrame, scenario_name: str | None, scenarios: dict) -> pd.DataFrame:
    if not scenario_name:
        return data.copy()

    scenario = scenarios[scenario_name]
    filtered = data.copy()
    start_date = scenario.get("start_date")
    end_date = scenario.get("end_date")

    if start_date:
        start_ts = pd.Timestamp(start_date)
        if getattr(filtered.index, "tz", None) is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        filtered = filtered[filtered.index >= start_ts]

    if end_date:
        end_ts = pd.Timestamp(end_date)
        if getattr(filtered.index, "tz", None) is not None and end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        filtered = filtered[filtered.index <= end_ts + pd.Timedelta(days=1)]

    return filtered.copy()


def summarize_run(label: str, scenario_key: str, scenario_name: str | None, oversample_ratio: float, budget: dict, results: dict) -> dict:
    windows = results.get("all_window_results", [])
    session_folder = None
    html_report_path = results.get("html_report_path")
    if html_report_path:
        session_folder = str(Path(html_report_path).parent.parent)

    profitable_windows = sum(1 for window in windows if float(window.get("return", 0.0)) > 0)
    worst_window = min(windows, key=lambda window: float(window.get("return", 0.0)), default=None)
    best_window = max(windows, key=lambda window: float(window.get("return", 0.0)), default=None)

    return {
        "label": label,
        "scenario": scenario_key,
        "scenario_config_name": scenario_name,
        "oversample_ratio": oversample_ratio,
        "budget": copy.deepcopy(budget),
        "session_folder": session_folder,
        "html_report_path": html_report_path,
        "avg_return": float(results.get("avg_return", 0.0)),
        "avg_sortino": float(results.get("avg_sortino", 0.0)),
        "avg_portfolio": float(results.get("avg_portfolio", 0.0)),
        "avg_trades": float(results.get("avg_trades", 0.0)),
        "avg_completed_trades": float(results.get("avg_completed_trades", results.get("avg_trades", 0.0))),
        "avg_rebalances": float(results.get("avg_rebalances", results.get("avg_trades", 0.0))),
        "avg_hit_rate": float(results.get("avg_hit_rate", 0.0)),
        "num_windows": int(results.get("num_windows", len(windows))),
        "profitable_windows": profitable_windows,
        "best_window_return": float(best_window.get("return", 0.0)) if best_window else 0.0,
        "worst_window_return": float(worst_window.get("return", 0.0)) if worst_window else 0.0,
        "worst_window_max_drawdown": float(worst_window.get("max_drawdown", 0.0)) if worst_window else 0.0,
        "windows": [
            {
                "window": int(window.get("window", 0)),
                "return": float(window.get("return", 0.0)),
                "portfolio_value": float(window.get("portfolio_value", 0.0)),
                "sortino_ratio": float(window.get("sortino_ratio", 0.0)),
                "calmar_ratio": float(window.get("calmar_ratio", 0.0)),
                "max_drawdown": float(window.get("max_drawdown", 0.0)),
                "trade_count": int(window.get("trade_count", 0)),
                "rebalance_count": int(window.get("rebalance_count", window.get("trade_count", 0))),
                "completed_trades": int(
                    window.get("completed_trades", window.get("economic_trade_count", 0))
                ),
                "economic_trade_count": int(
                    window.get("economic_trade_count", window.get("completed_trades", 0))
                ),
                "hit_rate": float(window.get("hit_rate", 0.0)),
                "train_start": window.get("train_start"),
                "train_end": window.get("train_end"),
                "test_start": window.get("test_start"),
                "test_end": window.get("test_end"),
                "window_folder": window.get("window_folder"),
                "report_data_path": window.get("report_data_path"),
            }
            for window in windows
        ],
    }


def run_single(label: str, scenario_key: str, scenario_name: str | None, oversample_ratio: float, dataset: pd.DataFrame, budget: dict) -> dict:
    aug_cfg = config.setdefault("augmentation", {}).setdefault("synthetic_bears", {})
    aug_cfg["enabled"] = True
    aug_cfg["oversample_ratio"] = float(oversample_ratio)

    results = walk_forward_testing(
        data=dataset,
        **budget,
    )
    return summarize_run(
        label=label,
        scenario_key=scenario_key,
        scenario_name=scenario_name,
        oversample_ratio=float(oversample_ratio),
        budget=budget,
        results=results,
    )


def build_scenario_comparison(baseline: dict, candidate: dict) -> list[dict]:
    rows = []
    for scenario_key, baseline_run in baseline.items():
        candidate_run = candidate[scenario_key]
        rows.append(
            {
                "scenario": scenario_key,
                "baseline_session": baseline_run.get("session_folder"),
                "candidate_session": candidate_run.get("session_folder"),
                "baseline_avg_return_pct": baseline_run["avg_return"],
                "candidate_avg_return_pct": candidate_run["avg_return"],
                "delta_avg_return_pct": candidate_run["avg_return"] - baseline_run["avg_return"],
                "baseline_avg_sortino": baseline_run["avg_sortino"],
                "candidate_avg_sortino": candidate_run["avg_sortino"],
                "delta_avg_sortino": candidate_run["avg_sortino"] - baseline_run["avg_sortino"],
                "baseline_avg_trades": baseline_run["avg_trades"],
                "candidate_avg_trades": candidate_run["avg_trades"],
                "delta_avg_trades": candidate_run["avg_trades"] - baseline_run["avg_trades"],
                "baseline_avg_completed_trades": baseline_run["avg_completed_trades"],
                "candidate_avg_completed_trades": candidate_run["avg_completed_trades"],
                "delta_avg_completed_trades": candidate_run["avg_completed_trades"] - baseline_run["avg_completed_trades"],
                "baseline_avg_rebalances": baseline_run["avg_rebalances"],
                "candidate_avg_rebalances": candidate_run["avg_rebalances"],
                "delta_avg_rebalances": candidate_run["avg_rebalances"] - baseline_run["avg_rebalances"],
                "baseline_avg_hit_rate_pct": baseline_run["avg_hit_rate"],
                "candidate_avg_hit_rate_pct": candidate_run["avg_hit_rate"],
                "baseline_worst_return_pct": baseline_run["worst_window_return"],
                "candidate_worst_return_pct": candidate_run["worst_window_return"],
                "baseline_worst_max_drawdown_pct": baseline_run["worst_window_max_drawdown"],
                "candidate_worst_max_drawdown_pct": candidate_run["worst_window_max_drawdown"],
                "candidate_profitable_windows": candidate_run["profitable_windows"],
                "baseline_profitable_windows": baseline_run["profitable_windows"],
            }
        )
    return rows


def write_summary_csv(rows: list[dict]) -> None:
    output_path = ARTIFACTS_DIR / "walk_forward_validation_summary.csv"
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows: list[dict]) -> None:
    output_path = ARTIFACTS_DIR / "walk_forward_validation_summary.md"
    lines = [
        "# Walk-Forward Validation Summary",
        "",
        "| Scenario | Baseline Avg Return | Candidate Avg Return | Delta Return | Baseline Avg Sortino | Candidate Avg Sortino | Delta Sortino | Baseline Completed | Candidate Completed | Delta Completed | Baseline Rebalances | Candidate Rebalances | Delta Rebalances |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {baseline_avg_return_pct:.2f}% | {candidate_avg_return_pct:.2f}% | {delta_avg_return_pct:+.2f}% | "
            "{baseline_avg_sortino:.2f} | {candidate_avg_sortino:.2f} | {delta_avg_sortino:+.2f} | "
            "{baseline_avg_completed_trades:.1f} | {candidate_avg_completed_trades:.1f} | {delta_avg_completed_trades:+.1f} | "
            "{baseline_avg_rebalances:.1f} | {candidate_avg_rebalances:.1f} | {delta_avg_rebalances:+.1f} |".format(**row)
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    original_aug = copy.deepcopy(config.get("augmentation", {}).get("synthetic_bears", {}))
    scenarios = load_scenarios()
    full_data = load_tradingview_data(config.get("data", {}).get("csv_path"))

    smoke_dataset = filter_for_scenario(full_data, None, scenarios)
    smoke_run = run_single(
        label="smoke_default_modified",
        scenario_key="default_full_cycle",
        scenario_name=None,
        oversample_ratio=0.08,
        dataset=smoke_dataset,
        budget=SMOKE_BUDGET,
    )

    baseline_runs = {}
    candidate_runs = {}
    for scenario_key, scenario_name in SCENARIO_ORDER:
        dataset = filter_for_scenario(full_data, scenario_name, scenarios)
        baseline_runs[scenario_key] = run_single(
            label=f"baseline_{scenario_key}",
            scenario_key=scenario_key,
            scenario_name=scenario_name,
            oversample_ratio=0.18,
            dataset=dataset,
            budget=VALIDATION_BUDGET,
        )
        candidate_runs[scenario_key] = run_single(
            label=f"candidate_{scenario_key}",
            scenario_key=scenario_key,
            scenario_name=scenario_name,
            oversample_ratio=0.08,
            dataset=dataset,
            budget=VALIDATION_BUDGET,
        )

    config.setdefault("augmentation", {})["synthetic_bears"] = original_aug

    scenario_comparison = build_scenario_comparison(baseline_runs, candidate_runs)

    payload = {
        "generated_at": pd.Timestamp.utcnow(),
        "smoke_run": smoke_run,
        "validation_budget": VALIDATION_BUDGET,
        "baseline_oversample_ratio": 0.18,
        "candidate_oversample_ratio": 0.08,
        "scenario_order": [scenario for scenario, _ in SCENARIO_ORDER],
        "runs": {
            "baseline": baseline_runs,
            "candidate": candidate_runs,
        },
        "scenario_comparison": scenario_comparison,
    }

    raw_path = ARTIFACTS_DIR / "codex_validation_raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    write_summary_csv(scenario_comparison)
    write_summary_md(scenario_comparison)


if __name__ == "__main__":
    main()
