"""
Separate Risk Management Model for AlgoTrader2.

Architecture:
- The TRADING model decides long/short (trained separately)
- The RISK model decides whether to allow/block that trade (this module)
- Separation of concerns: trading model optimizes returns, risk model limits drawdowns

The risk model is trained on backtest data from the trading model. It learns
to identify situations where the trading model's predictions lead to large losses
and blocks those trades.

Training flow:
1. Run trading model through backtest → collect trade history with outcomes
2. Build RiskManagementEnv from that history
3. Train PPO risk model to maximize: allowed winning trades - allowed losing trades
4. Save risk model separately

Inference flow:
1. Trading model predicts long/short
2. Risk model observes prediction + portfolio state + market conditions
3. Risk model outputs: allow (0) or block (1)
4. If blocked, maintain current position (no trade)
"""

import logging
import numpy as np
import pandas as pd
import os
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
from enum import IntEnum

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False

from environment import TradingEnv
from config import config
import money
import constants

logger = logging.getLogger(__name__)


class RiskAction(IntEnum):
    ALLOW = 0
    BLOCK = 1


def collect_trade_history(model, data: pd.DataFrame) -> Dict:
    """
    Run a trained trading model through data and collect detailed step-by-step
    history for risk model training.

    Returns dict with:
        - steps: list of per-step dicts with state, action, outcome info
        - trades: list of completed trades with entry/exit/pnl
        - portfolio_history: list of portfolio values
    """
    env = TradingEnv(
        data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1)
    )

    obs, _ = env.reset()
    is_recurrent = hasattr(model, 'policy') and hasattr(model.policy, 'lstm')
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    steps = []
    trades = []
    portfolio_history = [float(env.net_worth)]

    current_trade = None
    done = False

    while not done:
        # Get model prediction
        if is_recurrent:
            action, lstm_states = model.predict(obs, state=lstm_states,
                                                 episode_start=episode_starts, deterministic=True)
            episode_starts = np.array([done])
        else:
            action, _ = model.predict(obs, deterministic=True)

        action_int = int(action)
        old_position = env.position
        old_net_worth = float(env.net_worth)
        old_max_net_worth = float(env.max_net_worth)

        # Calculate drawdown before step
        if old_max_net_worth > 0:
            drawdown_pct = (old_max_net_worth - old_net_worth) / old_max_net_worth
        else:
            drawdown_pct = 0.0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_net_worth = float(env.net_worth)
        portfolio_history.append(new_net_worth)

        # Calculate step P&L
        step_pnl = new_net_worth - old_net_worth

        # Record step data for risk model training
        step_data = {
            "step": env.current_step,
            "action": action_int,
            "old_position": old_position,
            "new_position": env.position,
            "position_changed": info.get("position_changed", False),
            "net_worth": new_net_worth,
            "step_pnl": step_pnl,
            "drawdown_pct": drawdown_pct,
            "time_in_position": info.get("time_in_position", 0),
            "trade_count": info.get("trade_count", 0),
            "reward": float(reward),
        }
        steps.append(step_data)

        # Track individual trades
        if info.get("position_changed", False):
            # Close previous trade if there was one
            if current_trade is not None and old_position != 0:
                current_trade["exit_step"] = env.current_step
                current_trade["exit_net_worth"] = new_net_worth
                current_trade["pnl"] = new_net_worth - current_trade["entry_net_worth"]
                current_trade["bars_held"] = current_trade["exit_step"] - current_trade["entry_step"]
                trades.append(current_trade)
                current_trade = None

            # Open new trade if entering a position
            if env.position != 0:
                current_trade = {
                    "entry_step": env.current_step,
                    "entry_net_worth": new_net_worth,
                    "direction": env.position,  # 1=long, -1=short
                    "drawdown_at_entry": drawdown_pct,
                }

    # Close any open trade at end
    if current_trade is not None:
        current_trade["exit_step"] = env.current_step
        current_trade["exit_net_worth"] = float(env.net_worth)
        current_trade["pnl"] = float(env.net_worth) - current_trade["entry_net_worth"]
        current_trade["bars_held"] = current_trade["exit_step"] - current_trade["entry_step"]
        trades.append(current_trade)

    return {
        "steps": steps,
        "trades": trades,
        "portfolio_history": portfolio_history,
    }


class RiskManagementEnv(gym.Env):
    """
    Gymnasium environment for training a risk management model.

    The risk model replays the trading model's backtest and decides at each
    position-change step whether to ALLOW or BLOCK the trade.

    Observation space:
        - proposed_action: the trading model's action (0=long, 1=short), normalized
        - current_position: current position (-1, 0, 1), normalized to [-1, 1]
        - drawdown_pct: current drawdown from peak [0, 1]
        - unrealized_pnl_norm: normalized unrealized P&L [-1, 1]
        - time_in_position_norm: normalized time in position [0, 1]
        - recent_win_rate: win rate of last N trades [0, 1]
        - recent_avg_pnl: average P&L of last N trades, normalized [-1, 1]
        - consecutive_losses: number of consecutive losing trades, normalized [0, 1]
        - portfolio_return_pct: total return so far, normalized [-1, 1]
        - trade_count_norm: number of trades so far, normalized [0, 1]

    Action space:
        0 = ALLOW the trade
        1 = BLOCK the trade (maintain current position)

    Reward:
        - Allow a winning trade: +1.0
        - Allow a losing trade: penalty proportional to loss magnitude
        - Block a losing trade: +0.5 (saved money)
        - Block a winning trade: -0.2 (missed opportunity, light penalty)
        - Preventing drawdown below threshold: bonus
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, trade_history: Dict, lookback_trades: int = 10):
        super().__init__()

        self.steps = trade_history["steps"]
        self.trades = trade_history["trades"]
        self.portfolio_history = trade_history["portfolio_history"]
        self.lookback_trades = lookback_trades

        # Find steps where position changes (trade decisions)
        self.decision_points = []
        for i, step in enumerate(self.steps):
            if step["position_changed"]:
                self.decision_points.append(i)

        if len(self.decision_points) == 0:
            raise ValueError("No trades in history — cannot train risk model")

        # Observation: 10 features
        self.obs_size = 10
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        # Action: allow (0) or block (1)
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.current_decision_idx = 0
        self.allowed_trades = []
        self.blocked_trades = []
        self.simulated_net_worth = config["environment"]["initial_balance"]
        self.simulated_max_net_worth = self.simulated_net_worth
        self.recent_trade_results = []  # Track last N trade outcomes

    def reset(self, *, seed=None, options=None):
        self.current_decision_idx = 0
        self.allowed_trades = []
        self.blocked_trades = []
        self.simulated_net_worth = config["environment"]["initial_balance"]
        self.simulated_max_net_worth = self.simulated_net_worth
        self.recent_trade_results = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        if self.current_decision_idx >= len(self.decision_points):
            return np.zeros(self.obs_size, dtype=np.float32)

        step_idx = self.decision_points[self.current_decision_idx]
        step = self.steps[step_idx]

        # Proposed action (normalized: 0→-1, 1→1)
        proposed_action = step["action"] * 2.0 - 1.0

        # Current position before this trade
        current_position = float(step["old_position"])

        # Drawdown
        drawdown = float(step["drawdown_pct"])

        # Time in position (normalized)
        time_in_pos = float(np.tanh(step["time_in_position"] / 50.0))

        # Recent trade statistics
        if len(self.recent_trade_results) > 0:
            recent_wins = sum(1 for r in self.recent_trade_results if r > 0)
            recent_win_rate = recent_wins / len(self.recent_trade_results)
            recent_avg_pnl = np.tanh(np.mean(self.recent_trade_results) / 1000.0)

            # Consecutive losses from end
            consecutive_losses = 0
            for r in reversed(self.recent_trade_results):
                if r <= 0:
                    consecutive_losses += 1
                else:
                    break
            consecutive_losses_norm = min(consecutive_losses / 5.0, 1.0)
        else:
            recent_win_rate = 0.5
            recent_avg_pnl = 0.0
            consecutive_losses_norm = 0.0

        # Portfolio return (normalized)
        initial = config["environment"]["initial_balance"]
        portfolio_return = np.tanh((self.simulated_net_worth - initial) / initial)

        # Trade count (normalized — 0 at start, approaches 1 after many trades)
        total_trades = len(self.allowed_trades) + len(self.blocked_trades)
        trade_count_norm = np.tanh(total_trades / 100.0)

        # Unrealized PnL proxy (use step_pnl as approximation)
        unrealized_pnl_norm = np.tanh(step.get("step_pnl", 0) / 500.0)

        obs = np.array([
            proposed_action,
            current_position,
            drawdown,
            unrealized_pnl_norm,
            time_in_pos,
            recent_win_rate,
            recent_avg_pnl,
            consecutive_losses_norm,
            portfolio_return,
            trade_count_norm,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _find_trade_outcome(self, step_idx: int) -> Optional[float]:
        """Find the P&L outcome of a trade that starts near this step."""
        step = self.steps[step_idx]
        trade_count_at_step = step["trade_count"]

        # Find the matching trade in the trade list
        # Trades are in order, so find by approximate trade count
        for trade in self.trades:
            if abs(trade["entry_step"] - step["step"]) <= 2:
                return trade["pnl"]

        # Fallback: use net_worth change over next N steps
        lookahead = min(50, len(self.steps) - step_idx - 1)
        if lookahead > 0:
            future_nw = self.steps[min(step_idx + lookahead, len(self.steps) - 1)]["net_worth"]
            return future_nw - step["net_worth"]
        return 0.0

    def step(self, action):
        if self.current_decision_idx >= len(self.decision_points):
            return self._get_obs(), 0.0, True, False, {}

        step_idx = self.decision_points[self.current_decision_idx]
        trade_pnl = self._find_trade_outcome(step_idx)
        if trade_pnl is None:
            trade_pnl = 0.0

        reward = 0.0
        info = {"trade_pnl": trade_pnl, "action": int(action)}

        if action == RiskAction.ALLOW:
            # Trade is allowed — reward based on outcome
            self.simulated_net_worth += trade_pnl
            self.allowed_trades.append(trade_pnl)
            self.recent_trade_results.append(trade_pnl)

            if trade_pnl > 0:
                reward = 1.0 + np.tanh(trade_pnl / 1000.0)  # Bonus for big wins
            else:
                # Penalty proportional to loss size
                reward = -1.0 - np.tanh(abs(trade_pnl) / 1000.0)

        elif action == RiskAction.BLOCK:
            # Trade is blocked — no P&L change
            self.blocked_trades.append(trade_pnl)

            if trade_pnl < 0:
                # Correctly blocked a losing trade
                reward = 0.5 + np.tanh(abs(trade_pnl) / 1000.0) * 0.5
            else:
                # Incorrectly blocked a winning trade — strong penalty to prevent over-blocking
                reward = -1.0 - np.tanh(trade_pnl / 1000.0) * 0.5

        # Update peak for drawdown tracking
        if self.simulated_net_worth > self.simulated_max_net_worth:
            self.simulated_max_net_worth = self.simulated_net_worth

        # Bonus for keeping drawdown low
        if self.simulated_max_net_worth > config["environment"]["initial_balance"]:
            dd = (self.simulated_max_net_worth - self.simulated_net_worth) / self.simulated_max_net_worth
            if dd < 0.10:
                reward += 0.1  # Small bonus for low drawdown

        # Keep recent trades bounded
        if len(self.recent_trade_results) > self.lookback_trades:
            self.recent_trade_results = self.recent_trade_results[-self.lookback_trades:]

        # Advance to next decision point
        self.current_decision_idx += 1
        terminated = self.current_decision_idx >= len(self.decision_points)

        obs = self._get_obs()
        return obs, reward, terminated, False, info


class RiskModelWrapper:
    """
    Wraps a trained risk model for use during evaluation and live trading.

    Supports two modes:
    - "ppo": Uses a trained PPO risk model
    - "rules": Uses optimized rule-based thresholds (more reliable)

    Sits between the trading model's prediction and execution:
        prediction = trading_model.predict(obs)
        should_trade = risk_wrapper.should_allow(prediction, state)
        if should_trade:
            execute(prediction)
    """

    def __init__(self, risk_model_path: Optional[str] = None,
                 mode: str = "rules", rules_config: Optional[Dict] = None):
        self.model = None
        self.mode = mode
        self.enabled = False

        if mode == "ppo" and risk_model_path and os.path.exists(risk_model_path):
            self.model = PPO.load(risk_model_path)
            self.enabled = True
            logger.info(f"Risk model (PPO) loaded from {risk_model_path}")
        elif mode == "rules":
            self.enabled = True
            # Default rule thresholds — can be optimized from backtest data
            defaults = {
                "max_drawdown_pct": 0.30,       # Block trades when DD > 30%
                "max_consecutive_losses": 4,     # Pause after 4 consecutive losses
                "cooldown_bars_after_losses": 20, # Stay flat 20 bars after loss streak
                "max_daily_loss_pct": 0.15,      # Stop trading after 15% daily loss
                "min_bars_between_trades": 2,     # Minimum 2 bars between trades
            }
            self.rules = {**defaults, **(rules_config or {})}
            self._cooldown_remaining = 0
            self._bars_since_last_trade = 999
            self._daily_start_nw = None
            logger.info(f"Risk model (rules) enabled: {self.rules}")

        # Tracking stats
        self.total_decisions = 0
        self.allowed_count = 0
        self.blocked_count = 0
        self.block_reasons = {}
        self.recent_trade_results = []

    def should_allow(self, proposed_action: int, portfolio_state: Dict) -> bool:
        """
        Decide whether to allow a proposed trade.

        Args:
            proposed_action: 0 (long) or 1 (short) from trading model
            portfolio_state: dict with keys:
                - position: current position (-1, 0, 1)
                - drawdown_pct: current drawdown from peak
                - unrealized_pnl: current unrealized P&L
                - time_in_position: bars in current position
                - net_worth: current portfolio value
                - trade_count: total trades so far

        Returns:
            True if trade should be allowed, False if blocked
        """
        if not self.enabled:
            return True

        self.total_decisions += 1

        if self.mode == "ppo":
            return self._ppo_decide(proposed_action, portfolio_state)
        elif self.mode == "rules":
            return self._rules_decide(proposed_action, portfolio_state)

        return True

    def _ppo_decide(self, proposed_action: int, state: Dict) -> bool:
        """PPO-based decision."""
        obs = self._build_obs(proposed_action, state)
        action, _ = self.model.predict(obs, deterministic=True)
        if int(action) == RiskAction.ALLOW:
            self.allowed_count += 1
            return True
        self.blocked_count += 1
        return False

    def _rules_decide(self, proposed_action: int, state: Dict) -> bool:
        """Rule-based decision with optimized thresholds."""
        r = self.rules

        # Initialize daily tracking
        if self._daily_start_nw is None:
            self._daily_start_nw = state.get("net_worth", config["environment"]["initial_balance"])

        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._block("cooldown")
            return False

        # Increment bars since last trade
        self._bars_since_last_trade += 1

        # Rule 1: Max drawdown — don't enter new trades when deep in drawdown
        dd = state.get("drawdown_pct", 0)
        if dd > r["max_drawdown_pct"]:
            self._block("max_drawdown")
            return False

        # Rule 2: Consecutive losses — pause trading after losing streak
        consecutive_losses = 0
        for result in reversed(self.recent_trade_results):
            if result <= 0:
                consecutive_losses += 1
            else:
                break
        if consecutive_losses >= r["max_consecutive_losses"]:
            self._cooldown_remaining = r["cooldown_bars_after_losses"]
            self._block("consecutive_losses")
            return False

        # Rule 3: Daily loss limit
        nw = state.get("net_worth", self._daily_start_nw)
        daily_loss_pct = (self._daily_start_nw - nw) / self._daily_start_nw if self._daily_start_nw > 0 else 0
        if daily_loss_pct > r["max_daily_loss_pct"]:
            self._block("daily_loss_limit")
            return False

        # Rule 4: Minimum bars between trades
        if self._bars_since_last_trade < r["min_bars_between_trades"]:
            self._block("min_bars_between")
            return False

        # All checks passed
        self.allowed_count += 1
        self._bars_since_last_trade = 0
        return True

    def _block(self, reason: str):
        """Record a block with reason."""
        self.blocked_count += 1
        self.block_reasons[reason] = self.block_reasons.get(reason, 0) + 1

    def _build_obs(self, proposed_action: int, state: Dict) -> np.ndarray:
        """Build observation vector matching RiskManagementEnv format (for PPO mode)."""
        initial = config["environment"]["initial_balance"]

        if len(self.recent_trade_results) > 0:
            recent_wins = sum(1 for r in self.recent_trade_results if r > 0)
            recent_win_rate = recent_wins / len(self.recent_trade_results)
            recent_avg_pnl = float(np.tanh(np.mean(self.recent_trade_results) / 1000.0))
            consecutive_losses = 0
            for r in reversed(self.recent_trade_results):
                if r <= 0:
                    consecutive_losses += 1
                else:
                    break
            consecutive_losses_norm = min(consecutive_losses / 5.0, 1.0)
        else:
            recent_win_rate = 0.5
            recent_avg_pnl = 0.0
            consecutive_losses_norm = 0.0

        obs = np.array([
            proposed_action * 2.0 - 1.0,
            float(state.get("position", 0)),
            float(state.get("drawdown_pct", 0)),
            float(np.tanh(state.get("unrealized_pnl", 0) / 500.0)),
            float(np.tanh(state.get("time_in_position", 0) / 50.0)),
            recent_win_rate,
            recent_avg_pnl,
            consecutive_losses_norm,
            float(np.tanh((state.get("net_worth", initial) - initial) / initial)),
            float(np.tanh(state.get("trade_count", 0) / 100.0)),
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def record_trade_result(self, pnl: float):
        """Record a trade result for recent history tracking."""
        self.recent_trade_results.append(pnl)
        if len(self.recent_trade_results) > 10:
            self.recent_trade_results = self.recent_trade_results[-10:]

    def tick(self):
        """Called each bar to update internal counters (for bars_since_last_trade)."""
        if self.mode == "rules":
            self._bars_since_last_trade += 1
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1

    def get_stats(self) -> Dict:
        """Return risk model statistics."""
        stats = {
            "total_decisions": self.total_decisions,
            "allowed": self.allowed_count,
            "blocked": self.blocked_count,
            "block_rate": self.blocked_count / max(self.total_decisions, 1) * 100,
        }
        if self.mode == "rules" and self.block_reasons:
            stats["block_reasons"] = dict(self.block_reasons)
        return stats
