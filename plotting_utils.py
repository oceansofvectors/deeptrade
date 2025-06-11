import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import os

def plot_training_progress(training_history: List[Dict], save_path: str = None) -> None:
    """Plot training progress over iterations."""
    if not training_history:
        return
    
    iterations = list(range(len(training_history)))
    returns = [result.get("total_return_pct", 0) for result in training_history]
    hit_rates = [result.get("hit_rate", 0) for result in training_history]
    sharpe_ratios = [result.get("sharpe_ratio", 0) for result in training_history]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot returns
    ax1.plot(iterations, returns, 'b-o', label='Return %')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Training Progress: Return')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Sharpe ratios
    ax2.plot(iterations, sharpe_ratios, 'r-^', label='Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Training Progress: Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot hit rates
    ax3.plot(iterations, hit_rates, 'g-s', label='Hit Rate %')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Hit Rate (%)')
    ax3.set_title('Training Progress: Hit Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/training_progress.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_portfolio_performance(portfolio_history: List[float], 
                             action_history: List[int] = None,
                             price_data: pd.DataFrame = None,
                             save_path: str = None) -> None:
    """Plot portfolio performance over time."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot portfolio value
    axes[0].plot(portfolio_history, 'purple', linewidth=2, label='Portfolio Value')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title('Portfolio Performance')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot price and signals if available
    if price_data is not None and action_history is not None:
        # Get price column
        price_col = 'close'
        
        axes[1].plot(price_data[price_col].values, 'gray', label='Price', alpha=0.7)
        
        # Add buy/sell signals
        if len(action_history) <= len(price_data):
            buy_signals = [i for i, action in enumerate(action_history) if action == 0]
            sell_signals = [i for i, action in enumerate(action_history) if action == 1]
            
            if buy_signals:
                buy_prices = [price_data[price_col].iloc[i] for i in buy_signals if i < len(price_data)]
                axes[1].scatter(buy_signals[:len(buy_prices)], buy_prices, 
                              color='green', marker='^', s=50, label='Buy', alpha=0.8)
            
            if sell_signals:
                sell_prices = [price_data[price_col].iloc[i] for i in sell_signals if i < len(price_data)]
                axes[1].scatter(sell_signals[:len(sell_prices)], sell_prices, 
                              color='red', marker='v', s=50, label='Sell', alpha=0.8)
    
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Price')
    axes[1].set_title('Price and Trading Signals')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/portfolio_performance.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_walk_forward_results(results: List[Dict], save_path: str = None) -> None:
    """Plot walk-forward testing results."""
    if not results:
        return
    
    windows = [r.get('window', i+1) for i, r in enumerate(results)]
    returns = [r.get('return', 0) for r in results]
    sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
    portfolio_values = [r.get('portfolio_value', 0) for r in results]
    trade_counts = [r.get('trade_count', 0) for r in results]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # Plot returns
    axes[0].bar(windows, returns, color='blue', alpha=0.7)
    axes[0].set_ylabel('Return (%)')
    axes[0].set_title('Returns by Walk-Forward Window')
    axes[0].grid(True, alpha=0.3)
    
    # Add return values as text
    for i, (w, r) in enumerate(zip(windows, returns)):
        axes[0].text(w, r, f'{r:.1f}%', ha='center', va='bottom' if r >= 0 else 'top')
    
    # Plot Sharpe ratios
    axes[1].bar(windows, sharpe_ratios, color='red', alpha=0.7)
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].set_title('Sharpe Ratios by Walk-Forward Window')
    axes[1].grid(True, alpha=0.3)
    
    # Add Sharpe ratio values as text
    for i, (w, s) in enumerate(zip(windows, sharpe_ratios)):
        axes[1].text(w, s, f'{s:.2f}', ha='center', va='bottom' if s >= 0 else 'top')
    
    # Plot portfolio values
    axes[2].bar(windows, portfolio_values, color='green', alpha=0.7)
    axes[2].set_ylabel('Final Portfolio Value ($)')
    axes[2].set_title('Final Portfolio Values by Window')
    axes[2].grid(True, alpha=0.3)
    
    # Plot trade counts
    axes[3].bar(windows, trade_counts, color='orange', alpha=0.7)
    axes[3].set_xlabel('Walk-Forward Window')
    axes[3].set_ylabel('Trade Count')
    axes[3].set_title('Trade Counts by Window')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/walk_forward_results.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_cumulative_performance(results: List[Dict], save_path: str = None) -> None:
    """Plot cumulative performance across walk-forward windows."""
    if not results:
        return
        
    windows = [r.get('window', i+1) for i, r in enumerate(results)]
    returns = [r.get('return', 0) for r in results]
    
    # Calculate cumulative returns
    cumulative_returns = np.cumsum(returns)
    
    plt.figure(figsize=(10, 6))
    plt.plot(windows, cumulative_returns, 'o-', linewidth=2, markersize=6, color='blue')
    plt.xlabel('Walk-Forward Window')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Performance Across Windows')
    plt.grid(True, alpha=0.3)
    
    # Add final cumulative return as text
    if len(cumulative_returns) > 0:
        final_return = cumulative_returns[-1]
        plt.text(windows[-1], final_return, f'Final: {final_return:.1f}%', 
                ha='right', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/cumulative_performance.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_sharpe_comparison(results: Dict, save_path: str = None) -> None:
    """Plot comparison between individual window Sharpe ratios and pooled Sharpe ratio."""
    if not results or 'all_window_results' not in results:
        return
    
    window_results = results['all_window_results']
    if not window_results:
        return
        
    windows = [r.get('window', i+1) for i, r in enumerate(window_results)]
    sharpe_ratios = [r.get('sharpe_ratio', 0) for r in window_results]
    avg_sharpe = results.get('avg_sharpe_ratio', 0)
    pooled_sharpe = results.get('pooled_sharpe_ratio', 0)
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual window Sharpe ratios
    plt.bar(windows, sharpe_ratios, alpha=0.7, color='lightblue', 
            label=f'Individual Windows (avg: {avg_sharpe:.2f})')
    
    # Add horizontal lines for average and pooled Sharpe ratios
    plt.axhline(y=avg_sharpe, color='blue', linestyle='--', linewidth=2, 
                label=f'Average Sharpe: {avg_sharpe:.2f}')
    plt.axhline(y=pooled_sharpe, color='red', linestyle='-', linewidth=2, 
                label=f'Pooled Sharpe: {pooled_sharpe:.2f}')
    
    plt.xlabel('Walk-Forward Window')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison: Individual vs Pooled')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation explaining the difference
    y_range = max(sharpe_ratios) - min(sharpe_ratios) if sharpe_ratios else 1
    plt.text(0.02, 0.98, 
             f'Pooled Sharpe uses all {sum(len(r.get("portfolio_history", [])) for r in window_results)} returns\n'
             f'from {len(window_results)} windows combined into one calculation',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/sharpe_comparison.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_hyperparameter_tuning_summary(results: Dict, save_path: str = None) -> None:
    """Plot summary of hyperparameter tuning results across windows."""
    if not results or 'all_window_results' not in results:
        return
    
    window_results = results['all_window_results']
    tuned_windows = [r for r in window_results if r.get('best_hyperparameters') is not None]
    
    if not tuned_windows:
        return
    
    # Extract hyperparameter values
    windows = [r.get('window', i+1) for i, r in enumerate(tuned_windows)]
    learning_rates = [r.get('best_hyperparameters', {}).get('learning_rate', 0) for r in tuned_windows]
    n_steps = [r.get('best_hyperparameters', {}).get('n_steps', 0) for r in tuned_windows]
    ent_coefs = [r.get('best_hyperparameters', {}).get('ent_coef', 0) for r in tuned_windows]
    batch_sizes = [r.get('best_hyperparameters', {}).get('batch_size', 0) for r in tuned_windows]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # Plot learning rates
    axes[0].plot(windows, learning_rates, 'o-', color='blue', linewidth=2, markersize=6)
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('Optimal Learning Rates by Window')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot n_steps
    axes[1].plot(windows, n_steps, 'o-', color='green', linewidth=2, markersize=6)
    axes[1].set_ylabel('N Steps')
    axes[1].set_title('Optimal N Steps by Window')
    axes[1].grid(True, alpha=0.3)
    
    # Plot entropy coefficients
    axes[2].plot(windows, ent_coefs, 'o-', color='red', linewidth=2, markersize=6)
    axes[2].set_ylabel('Entropy Coefficient')
    axes[2].set_title('Optimal Entropy Coefficients by Window')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    # Plot batch sizes
    axes[3].plot(windows, batch_sizes, 'o-', color='purple', linewidth=2, markersize=6)
    axes[3].set_xlabel('Walk-Forward Window')
    axes[3].set_ylabel('Batch Size')
    axes[3].set_title('Optimal Batch Sizes by Window')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/hyperparameter_tuning_summary.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def create_summary_report(results: Dict, save_path: str = None) -> str:
    """Create a text summary report of results."""
    
    report = []
    report.append("="*60)
    report.append("WALK-FORWARD TESTING SUMMARY REPORT")
    report.append("="*60)
    report.append("")
    
    # Basic statistics
    report.append("OVERALL PERFORMANCE:")
    report.append(f"  Number of Windows: {results.get('num_windows', 0)}")
    report.append(f"  Average Return: {results.get('avg_return', 0):.2f}%")
    report.append(f"  Standard Deviation: {results.get('std_return', 0):.2f}%")
    
    # Add Sharpe ratio statistics if available
    if 'avg_sharpe_ratio' in results:
        report.append(f"  Average Sharpe Ratio: {results.get('avg_sharpe_ratio', 0):.2f}")
        report.append(f"  Sharpe Ratio Std Dev: {results.get('std_sharpe_ratio', 0):.2f}")
        
    # Add pooled Sharpe ratio if available
    if 'pooled_sharpe_ratio' in results:
        report.append(f"  Pooled Sharpe Ratio: {results.get('pooled_sharpe_ratio', 0):.2f} (from all pooled returns)")
    
    report.append(f"  Average Portfolio Value: ${results.get('avg_portfolio', 0):.2f}")
    report.append(f"  Average Trades per Window: {results.get('avg_trades', 0):.1f}")
    report.append("")
    
    # Additional metrics if available
    if 'avg_hit_rate' in results:
        report.append(f"  Average Hit Rate: {results.get('avg_hit_rate', 0):.2f}%")
    
    if 'avg_prediction_accuracy' in results:
        report.append(f"  Average Prediction Accuracy: {results.get('avg_prediction_accuracy', 0):.2f}%")
    
    # Bankruptcy statistics if available
    if 'bankrupted_windows' in results:
        report.append(f"  Bankrupted Windows: {results.get('bankrupted_windows', 0)}")
        report.append(f"  Bankruptcy Rate: {results.get('bankruptcy_rate', 0):.1f}%")
    
    # Hyperparameter tuning statistics
    if 'all_window_results' in results and results['all_window_results']:
        tuned_windows = [r for r in results['all_window_results'] if r.get('best_hyperparameters') is not None]
        if tuned_windows:
            report.append("")
            report.append("HYPERPARAMETER TUNING SUMMARY:")
            report.append(f"  Windows with Hyperparameter Tuning: {len(tuned_windows)}/{results.get('num_windows', 0)}")
            
            # Calculate average hyperparameters
            if tuned_windows:
                avg_lr = np.mean([r.get('best_hyperparameters', {}).get('learning_rate', 0) for r in tuned_windows])
                avg_n_steps = np.mean([r.get('best_hyperparameters', {}).get('n_steps', 0) for r in tuned_windows])
                avg_ent_coef = np.mean([r.get('best_hyperparameters', {}).get('ent_coef', 0) for r in tuned_windows])
                avg_batch_size = np.mean([r.get('best_hyperparameters', {}).get('batch_size', 0) for r in tuned_windows])
                
                report.append(f"  Average Optimal Learning Rate: {avg_lr:.6f}")
                report.append(f"  Average Optimal N Steps: {avg_n_steps:.0f}")
                report.append(f"  Average Optimal Entropy Coef: {avg_ent_coef:.6f}")
                report.append(f"  Average Optimal Batch Size: {avg_batch_size:.0f}")
    
    report.append("")
    
    # Window-by-window breakdown
    if 'all_window_results' in results and results['all_window_results']:
        report.append("WINDOW-BY-WINDOW BREAKDOWN:")
        report.append("-" * 80)
        
        for window_result in results['all_window_results']:
            window = window_result.get('window', 0)
            ret = window_result.get('return', 0)
            sharpe = window_result.get('sharpe_ratio', 0)
            portfolio = window_result.get('portfolio_value', 0)
            trades = window_result.get('trade_count', 0)
            has_hp_tuning = window_result.get('best_hyperparameters') is not None
            
            hp_indicator = " [HP]" if has_hp_tuning else ""
            report.append(f"  Window {window:2d}: {ret:6.2f}% return, {sharpe:6.2f} Sharpe, ${portfolio:8.2f} portfolio, {trades:3d} trades{hp_indicator}")
    
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(f'{save_path}/summary_report.txt', 'w') as f:
            f.write(report_text)
    
    return report_text 