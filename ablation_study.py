import os
import json
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment import TradingEnv
from config import config
from train import evaluate_agent, train_agent_iteratively
import matplotlib.pyplot as plt

# Make sure config has necessary structure
if 'data' not in config:
    config['data'] = {}
if 'symbol' not in config['data']:
    config['data']['symbol'] = 'NQ'  # Default to NQ
if 'period' not in config['data']:
    config['data']['period'] = '60d'  # Default to 60 days
if 'interval' not in config['data']:
    config['data']['interval'] = '5m'  # Default to 5 minutes
if 'environment' not in config:
    config['environment'] = {'initial_balance': 10000.0, 'position_size': 1}
if 'training' not in config:
    config['training'] = {}

def run_ablation_study(data_train, data_val, all_indicators, results_dir="ablation_results"):
    """
    Run ablation study to measure the effect of removing each indicator.
    
    Args:
        data_train: Training data DataFrame
        data_val: Validation data DataFrame
        all_indicators: List of all available indicators
        results_dir: Directory to save results
    
    Returns:
        DataFrame with ablation study results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Define training parameters
    initial_timesteps = config["training"].get("total_timesteps", 20000) // 2  # Reduced for efficiency
    max_iterations = config["training"].get("max_iterations", 10) // 2  # Reduced for efficiency
    
    # First, train a baseline model with all indicators
    print("Training baseline model with all indicators...")
    baseline_model, baseline_results, _ = train_agent_iteratively(
        data_train, 
        data_val,
        initial_timesteps=initial_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=2,
        evaluation_metric="return"
    )
    
    # Get the expected observation dimension from the baseline model
    expected_obs_dim = baseline_model.observation_space.shape[0]
    print(f"Model expects observation space with dimension: {expected_obs_dim}")
    
    # Expected indicator count (obs_dim - close_norm - position)
    expected_indicator_count = expected_obs_dim - 2
    print(f"Expected indicator count: {expected_indicator_count}")
    
    # Verify all_indicators matches expected dimension
    if len(all_indicators) != expected_indicator_count:
        print(f"Warning: Model expects {expected_indicator_count} indicators but {len(all_indicators)} were provided")
        # If we have too many indicators, trim the list
        if len(all_indicators) > expected_indicator_count:
            print(f"Trimming indicator list to match model's expected dimensions")
            all_indicators = all_indicators[:expected_indicator_count]
        # If we have too few indicators, we might need more
        else:
            print(f"Not enough indicators provided. Results may be unreliable.")
    
    baseline_performance = baseline_results["total_return_pct"]
    print(f"Baseline model performance: {baseline_performance:.2f}%")
    
    # Save baseline results
    with open(os.path.join(results_dir, "baseline_results.json"), "w") as f:
        json.dump({
            "performance": baseline_performance,
            "portfolio_value": baseline_results["final_portfolio_value"],
            "trade_count": baseline_results["trade_count"]
        }, f, indent=4)
    
    # Now run ablation for each indicator
    ablation_results = []
    
    for i, indicator_to_remove in enumerate(all_indicators):
        print(f"Running ablation {i+1}/{len(all_indicators)}: Removing {indicator_to_remove}")
        
        # Create subset of indicators without the current one
        reduced_indicators = [ind for ind in all_indicators if ind != indicator_to_remove]
        
        # Check if removing the indicator makes the observation space too small
        if len(reduced_indicators) < expected_indicator_count - 1:
            print(f"Warning: Removing {indicator_to_remove} would create an observation space smaller than expected.")
            print(f"Adding a placeholder indicator to maintain observation space size")
            # Add a placeholder (use the first indicator duplicated)
            reduced_indicators.append(reduced_indicators[0])
        
        # Train model without this indicator
        try:
            model, results, _ = train_agent_iteratively(
                data_train, 
                data_val,
                initial_timesteps=initial_timesteps,
                max_iterations=max_iterations,
                n_stagnant_loops=2,
                evaluation_metric="return",
                enabled_indicators=reduced_indicators
            )
            
            # Calculate performance difference
            performance = results["total_return_pct"]
            performance_change = performance - baseline_performance
            
            # Record results
            ablation_results.append({
                "Indicator": indicator_to_remove,
                "Performance": performance,
                "Performance_Change": performance_change,
                "Portfolio_Value": results["final_portfolio_value"],
                "Trade_Count": results["trade_count"]
            })
            
            print(f"  Performance without {indicator_to_remove}: {performance:.2f}% (Change: {performance_change:+.2f}%)")
        except Exception as e:
            print(f"Error in ablation for {indicator_to_remove}: {e}")
            # Add to results with null values to maintain record
            ablation_results.append({
                "Indicator": indicator_to_remove,
                "Performance": None,
                "Performance_Change": None,
                "Portfolio_Value": None,
                "Trade_Count": None,
                "Error": str(e)
            })
        
        # Save interim results
        ablation_df = pd.DataFrame(ablation_results)
        ablation_df.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False)
    
    # Sort by performance change, handling None values
    ablation_df = pd.DataFrame(ablation_results)
    valid_results = ablation_df.dropna(subset=["Performance_Change"])
    if not valid_results.empty:
        valid_results = valid_results.sort_values("Performance_Change", ascending=False)
    
    # Plot results if we have valid data
    if not valid_results.empty:
        plot_ablation_results(valid_results, baseline_performance, results_dir)
    
    return ablation_df

def plot_ablation_results(ablation_df, baseline_performance, results_dir):
    """
    Plot ablation study results.
    """
    # Sort by performance change
    df_sorted = ablation_df.sort_values("Performance_Change")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot performance changes
    bars = plt.barh(df_sorted["Indicator"], df_sorted["Performance_Change"])
    
    # Color bars based on impact (negative=bad, positive=good)
    colors = ['green' if x > 0 else 'red' for x in df_sorted["Performance_Change"]]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel("Performance Change (%)")
    plt.ylabel("Indicator Removed")
    plt.title("Impact of Removing Each Indicator (Higher is Better to Remove)")
    
    # Add annotation explaining the chart
    plt.figtext(0.5, 0.01, 
                "Green bars: Removing this indicator improved performance (candidate for pruning)\n"
                "Red bars: Removing this indicator hurt performance (keep this indicator)", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ablation_results.png"))
    plt.close()
    
    # Create horizontal bar chart of indicator importance
    plt.figure(figsize=(12, 10))
    
    # Calculate importance as negative of performance change (more negative change = more important)
    df_sorted["Importance"] = -df_sorted["Performance_Change"]
    df_by_importance = df_sorted.sort_values("Importance", ascending=False)
    
    # Plot importance
    bars = plt.barh(df_by_importance["Indicator"], df_by_importance["Importance"])
    
    # Color bars based on importance
    colors = ['blue' if x > 0 else 'lightblue' for x in df_by_importance["Importance"]]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add labels and title
    plt.xlabel("Indicator Importance (Impact when Removed)")
    plt.ylabel("Indicator")
    plt.title("Indicator Importance Based on Ablation Study")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "indicator_importance.png"))
    plt.close()

def run_forward_selection(data_train, data_val, all_indicators, results_dir="forward_selection_results"):
    """
    Run forward selection to find the optimal subset of indicators.
    
    Args:
        data_train: Training data DataFrame
        data_val: Validation data DataFrame
        all_indicators: List of all available indicators
        results_dir: Directory to save results
    
    Returns:
        List of optimal indicators
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Define training parameters (use shorter training for efficiency)
    initial_timesteps = config["training"].get("total_timesteps", 20000) // 3
    max_iterations = config["training"].get("max_iterations", 10) // 2
    
    # First, train a baseline model with all indicators to get expected dimensions
    print("Training temporary baseline model to determine expected dimensions...")
    temp_model, _, _ = train_agent_iteratively(
        data_train, 
        data_val,
        initial_timesteps=initial_timesteps//2,  # Use even fewer timesteps for this test
        max_iterations=2,  # Minimal iterations
        n_stagnant_loops=1,
        evaluation_metric="return"
    )
    
    # Get the expected observation dimension from the baseline model
    expected_obs_dim = temp_model.observation_space.shape[0]
    print(f"Model expects observation space with dimension: {expected_obs_dim}")
    
    # Expected indicator count (obs_dim - close_norm - position)
    expected_indicator_count = expected_obs_dim - 2
    print(f"Expected indicator count: {expected_indicator_count}")
    
    # Sort indicators by potential importance (can be refined later)
    # This helps with the initial steps of forward selection
    all_indicators_copy = all_indicators.copy()
    
    # Start with empty set of indicators
    selected_indicators = []
    remaining_indicators = all_indicators_copy.copy()
    
    # Keep track of performance at each step
    selection_results = []
    
    best_performance = -float('inf')
    
    while remaining_indicators and len(selected_indicators) < expected_indicator_count:
        step_results = []
        
        print(f"Forward selection step {len(selected_indicators) + 1}/{expected_indicator_count}")
        print(f"Currently selected: {selected_indicators}")
        
        # Try adding each remaining indicator
        for indicator in remaining_indicators:
            current_indicators = selected_indicators + [indicator]
            print(f"  Testing with {indicator} added...")
            
            # If we have fewer indicators than expected, pad with duplicates
            if len(current_indicators) < expected_indicator_count:
                # Add padding placeholders to match expected dimension
                padding_needed = expected_indicator_count - len(current_indicators)
                if padding_needed > 0:
                    print(f"  Adding {padding_needed} placeholder indicators to match dimension requirements")
                    # Use first indicator repeatedly as padding
                    padding = [current_indicators[0]] * padding_needed
                    padded_indicators = current_indicators + padding
                else:
                    padded_indicators = current_indicators
            else:
                padded_indicators = current_indicators
            
            try:
                # Train model with current set of indicators
                model, results, _ = train_agent_iteratively(
                    data_train, 
                    data_val,
                    initial_timesteps=initial_timesteps,
                    max_iterations=max_iterations,
                    n_stagnant_loops=2,
                    evaluation_metric="return",
                    enabled_indicators=padded_indicators
                )
                
                performance = results["total_return_pct"]
                
                step_results.append({
                    "Indicator": indicator,
                    "Performance": performance,
                    "Portfolio_Value": results["final_portfolio_value"],
                    "Trade_Count": results["trade_count"]
                })
                
                print(f"    Performance with {indicator}: {performance:.2f}%")
            except Exception as e:
                print(f"Error when testing with {indicator}: {e}")
                # Skip this indicator
                continue
        
        # Find the best indicator to add
        if step_results:
            step_df = pd.DataFrame(step_results)
            best_idx = step_df["Performance"].idxmax()
            best_indicator = step_df.loc[best_idx, "Indicator"]
            best_step_performance = step_df.loc[best_idx, "Performance"]
            
            # Record results
            selection_results.append({
                "Step": len(selected_indicators) + 1,
                "Added_Indicator": best_indicator,
                "Current_Indicators": selected_indicators + [best_indicator],
                "Performance": best_step_performance
            })
            
            # Check if we improved performance
            if best_step_performance > best_performance:
                best_performance = best_step_performance
                selected_indicators.append(best_indicator)
                remaining_indicators.remove(best_indicator)
                
                print(f"  Added {best_indicator} (Performance: {best_step_performance:.2f}%)")
            else:
                print(f"  No improvement, stopping")
                break
        else:
            print("  No valid results in this step, stopping")
            break
        
        # Save interim results
        selection_df = pd.DataFrame(selection_results)
        selection_df.to_csv(os.path.join(results_dir, "forward_selection_results.csv"), index=False)
    
    # Plot results
    if selection_results:
        plot_selection_results(selection_results, results_dir)
    
    # Return optimal set
    return selected_indicators

def plot_selection_results(selection_results, results_dir):
    """
    Plot forward selection results.
    """
    selection_df = pd.DataFrame(selection_results)
    
    plt.figure(figsize=(12, 6))
    plt.plot(selection_df["Step"], selection_df["Performance"], 'o-', linewidth=2)
    plt.xlabel("Number of Indicators")
    plt.ylabel("Performance (%)")
    plt.title("Performance vs. Number of Indicators (Forward Selection)")
    plt.grid(True, alpha=0.3)
    
    # Add indicator labels
    for i, row in selection_df.iterrows():
        plt.annotate(row["Added_Indicator"], 
                   (row["Step"], row["Performance"]),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "forward_selection_performance.png"))
    plt.close()

if __name__ == "__main__":
    from get_data import get_data
    
    # Load data with default parameter values if needed
    train_data, validation_data, test_data = get_data(
        symbol=config["data"].get("symbol", "NQ"),
        period=config["data"].get("period", "60d"),
        interval=config["data"].get("interval", "5m"),
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        use_yfinance=True
    )
    
    # Get all available indicators
    all_indicators = []
    for col in train_data.columns:
        # Check if column is a technical indicator
        if col in ['supertrend', 'RSI', 'CCI', 'ADX', 'ADX_POS', 'ADX_NEG', 'STOCH_K', 'STOCH_D',
                 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC', 'WILLIAMS_R', 'SMA_NORM', 'EMA_NORM',
                 'DISPARITY', 'ATR', 'OBV_NORM', 'CMF', 'PSAR_NORM', 'PSAR_DIR', 'VOLUME_MA',
                 'VWAP_NORM', 'DOW_SIN', 'DOW_COS', 'MSO_SIN', 'MSO_COS']:
            all_indicators.append(col)
    
    print(f"Found {len(all_indicators)} indicators in the data")
    
    # Choose which analysis to run
    ablation_study = True
    forward_selection = False
    
    if ablation_study:
        print("\n=== Running Ablation Study ===")
        ablation_results = run_ablation_study(train_data, validation_data, all_indicators)
        
        # Print recommendations
        print("\nIndicator Pruning Recommendations:")
        # Only use valid results (not None)
        valid_results = ablation_results.dropna(subset=["Performance_Change"])
        
        if not valid_results.empty:
            benefit_indicators = valid_results[valid_results["Performance_Change"] > 0]["Indicator"].tolist()
            if benefit_indicators:
                print(f"These indicators can be safely removed (improved performance when removed):")
                for ind in benefit_indicators:
                    change = valid_results[valid_results["Indicator"] == ind]["Performance_Change"].values[0]
                    print(f"  - {ind} (+{change:.2f}% when removed)")
            
            neutral_indicators = valid_results[
                (valid_results["Performance_Change"] <= 0) & 
                (valid_results["Performance_Change"] >= -0.5)
            ]["Indicator"].tolist()
            
            if neutral_indicators:
                print(f"\nThese indicators have minimal impact (within 0.5% performance):")
                for ind in neutral_indicators:
                    change = valid_results[valid_results["Indicator"] == ind]["Performance_Change"].values[0]
                    print(f"  - {ind} ({change:.2f}% when removed)")
            
            important_indicators = valid_results[valid_results["Performance_Change"] < -0.5]["Indicator"].tolist()
            if important_indicators:
                print(f"\nThese indicators are important and should be kept:")
                for ind in important_indicators:
                    change = valid_results[valid_results["Indicator"] == ind]["Performance_Change"].values[0]
                    print(f"  - {ind} ({change:.2f}% when removed)")
        else:
            print("No valid results were obtained in the ablation study.")
    
    if forward_selection:
        print("\n=== Running Forward Selection ===")
        optimal_indicators = run_forward_selection(train_data, validation_data, all_indicators)
        
        print("\nOptimal set of indicators:")
        for i, ind in enumerate(optimal_indicators):
            print(f"{i+1}. {ind}")
        
        # Save optimal set
        with open("optimal_indicators.json", "w") as f:
            json.dump({"optimal_indicators": optimal_indicators}, f, indent=4) 