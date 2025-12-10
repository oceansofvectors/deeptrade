"""
Performance metrics for trading evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TradeMetrics:
    """Container for trade performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_return: float
    total_profit: float
    total_loss: float
    profit_factor: float


def calculate_returns(net_worths: np.ndarray) -> np.ndarray:
    """
    Calculate returns from net worth series.

    Args:
        net_worths: Array of portfolio values

    Returns:
        Array of returns
    """
    returns = np.diff(net_worths) / net_worths[:-1]
    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 78  # 5-min bars
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(net_worths: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        net_worths: Array of portfolio values

    Returns:
        Maximum drawdown as a positive percentage
    """
    if len(net_worths) < 2:
        return 0.0

    peak = net_worths[0]
    max_dd = 0.0

    for value in net_worths:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def calculate_win_rate(trades: List[float]) -> float:
    """
    Calculate win rate from trade returns.

    Args:
        trades: List of trade returns

    Returns:
        Win rate (0-1)
    """
    if len(trades) == 0:
        return 0.0

    wins = sum(1 for t in trades if t > 0)
    return wins / len(trades)


def calculate_profit_factor(trades: List[float]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of trade returns

    Returns:
        Profit factor
    """
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def extract_trades(
    positions: np.ndarray,
    net_worths: np.ndarray
) -> List[float]:
    """
    Extract individual trade returns from position and net worth series.

    Args:
        positions: Array of position states (1, -1, 0)
        net_worths: Array of portfolio values

    Returns:
        List of trade returns
    """
    trades = []
    in_trade = False
    trade_start_value = 0.0
    prev_position = 0

    for i in range(len(positions)):
        position = positions[i]
        value = net_worths[i]

        # Position change
        if position != prev_position:
            if in_trade:
                # Close previous trade
                trade_return = (value - trade_start_value) / trade_start_value
                trades.append(trade_return)

            if position != 0:
                # Open new trade
                in_trade = True
                trade_start_value = value
            else:
                in_trade = False

        prev_position = position

    # Close final trade if still open
    if in_trade and len(net_worths) > 0:
        trade_return = (net_worths[-1] - trade_start_value) / trade_start_value
        trades.append(trade_return)

    return trades


def calculate_metrics(
    net_worths: np.ndarray,
    positions: np.ndarray
) -> TradeMetrics:
    """
    Calculate comprehensive trading metrics.

    Args:
        net_worths: Array of portfolio values
        positions: Array of position states

    Returns:
        TradeMetrics object
    """
    # Basic returns
    returns = calculate_returns(net_worths)
    total_return = (net_worths[-1] - net_worths[0]) / net_worths[0] if len(net_worths) > 0 else 0.0

    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns)

    # Max drawdown
    max_dd = calculate_max_drawdown(net_worths)

    # Extract trades
    trades = extract_trades(positions, net_worths)

    # Trade metrics
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)

    total_profit = sum(t for t in trades if t > 0)
    total_loss = abs(sum(t for t in trades if t < 0))
    avg_trade = np.mean(trades) if trades else 0.0

    return TradeMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=len(trades),
        avg_trade_return=avg_trade,
        total_profit=total_profit,
        total_loss=total_loss,
        profit_factor=profit_factor
    )


def print_metrics(metrics: TradeMetrics):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 50)
    print("TRADING PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Total Return:      {metrics.total_return * 100:.2f}%")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:      {metrics.max_drawdown * 100:.2f}%")
    print(f"Win Rate:          {metrics.win_rate * 100:.1f}%")
    print(f"Number of Trades:  {metrics.num_trades}")
    print(f"Avg Trade Return:  {metrics.avg_trade_return * 100:.3f}%")
    print(f"Profit Factor:     {metrics.profit_factor:.2f}")
    print("=" * 50 + "\n")
