import logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import TradingEnv
from get_data import get_data
from config import config
import money  # Import the new money module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEE_RATE = 0.0  # No trading fees

def train_agent(train_data, total_timesteps: int):
    """
    Train a PPO model based on training data.

    Args:
        train_data (pd.DataFrame): Training dataset.
        total_timesteps (int): Number of training timesteps.

    Returns:
        model: Trained PPO model.
    """
    env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        position_size=config["environment"].get("position_size", 1)
    )
    check_env(env, skip_render_check=True)
    model = PPO("MlpPolicy", env, verbose=1)
    logger.info("Starting training for %d timesteps", total_timesteps)
    model.learn(total_timesteps=total_timesteps)
    logger.info("Training completed")
    return model

def train_agent_iteratively(train_data, validation_data, initial_timesteps: int, max_iterations: int = 20, 
                           n_stagnant_loops: int = 3, improvement_threshold: float = 0.1, additional_timesteps: int = 10000):
    """
    Train a PPO model iteratively based on validation performance.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        validation_data (pd.DataFrame): Validation dataset for model selection.
        initial_timesteps (int): Initial number of training timesteps.
        max_iterations (int): Maximum number of training iterations.
        n_stagnant_loops (int): Number of consecutive iterations without improvement before stopping.
        improvement_threshold (float): Minimum percentage improvement considered significant.
        additional_timesteps (int): Number of additional timesteps for each iteration.
        
    Returns:
        tuple: (best_model, best_results, all_results)
    """
    # Initialize training environment
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,  # No transaction costs
        position_size=config["environment"].get("position_size", 1)
    )
    
    check_env(train_env, skip_render_check=True)
    
    # Get verbosity level from config
    verbose_level = config["training"].get("verbose", 1)
    
    # Initialize the PPO model
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=verbose_level,
        ent_coef=config["model"].get("ent_coef", 0.01),
        learning_rate=config["model"].get("learning_rate", 0.0003),
        n_steps=config["model"].get("n_steps", 2048),
        batch_size=config["model"].get("batch_size", 64),
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps)
    
    # Evaluate initial model on validation data (deterministic action selection)
    if verbose_level > 0:
        logger.info("Evaluating model on validation data with deterministic action selection")
    results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
    best_return = results["total_return_pct"]
    best_model = model
    best_results = results
    
    # Save the initial model as the best model so far
    best_model.save("best_model")
    
    logger.info(f"Initial training completed. Validation Return: {best_return:.2f}%, Validation Portfolio: ${results['final_portfolio_value']:.2f}")
    
    # Store all results for comparison
    all_results = [results]
    
    # Counter for consecutive iterations without significant improvement
    stagnant_counter = 0
    
    # Continue training until max_iterations or n_stagnant_loops consecutive iterations without improvement
    for iteration in range(1, max_iterations + 1):
        # Train for additional timesteps
        if verbose_level > 0:
            logger.info(f"Starting iteration {iteration} training for {additional_timesteps} timesteps")
        model.learn(total_timesteps=additional_timesteps)
        
        # Evaluate the model on validation data (deterministic)
        results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
        current_return = results["total_return_pct"]
        all_results.append(results)
        
        # Calculate improvement
        improvement = current_return - best_return
        logger.info(f"Iteration {iteration} - Validation Return: {current_return:.2f}%, " 
                   f"Validation Portfolio: ${results['final_portfolio_value']:.2f}, "
                   f"Improvement: {improvement:.2f}%")
        
        # Check if this is the best model so far based on validation performance
        if current_return > best_return + improvement_threshold:
            best_return = current_return
            best_model = model
            best_results = results
            logger.info(f"New best model found! Validation Return: {best_return:.2f}%, " 
                       f"Validation Portfolio: ${best_results['final_portfolio_value']:.2f}")
            
            # Save the best model
            best_model.save("best_model")
            
            # Reset stagnant counter since we found improvement
            stagnant_counter = 0
        else:
            # Increment stagnant counter if no significant improvement
            stagnant_counter += 1
            if verbose_level > 0:
                logger.info(f"No significant improvement. Stagnant iterations: {stagnant_counter}/{n_stagnant_loops}")
        
        # Stop if we've had n_stagnant_loops consecutive iterations without improvement
        if stagnant_counter >= n_stagnant_loops:
            logger.info(f"Stopping training after {n_stagnant_loops} consecutive iterations without significant improvement")
            break
    
    logger.info(f"Iterative training completed. Best validation return: {best_return:.2f}%, " 
               f"Validation Portfolio: ${best_results['final_portfolio_value']:.2f}")
    return best_model, best_results, all_results

def evaluate_agent(model, test_data, fee_rate: float = 0.0, verbose: int = 1, deterministic: bool = True):
    """
    Evaluate the trained model on test data and record trade history.

    Args:
        model: Trained PPO model.
        test_data (pd.DataFrame): Testing dataset.
        fee_rate (float): Trading fee rate (default 0.0 - no fees).
        verbose (int): Verbosity level for logging (default 1).
        deterministic (bool): Whether to use deterministic action selection (default: True).

    Returns:
        dict: A dictionary containing performance metrics and trade history.
    """
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,  # No transaction costs
        position_size=config["environment"].get("position_size", 1)
    )
    
    obs, _ = env.reset()
    initial_net_worth = env.net_worth
    
    trade_history = []
    trade_count = 0
    profitable_trades = 0  # Track number of profitable trades
    action_counts = {0: 0, 1: 0}  # Track action distribution (0: long/buy, 1: short/sell)

    # Records for plotting
    # Get the actual dates from the index
    dates = [test_data.index[env.current_step]]
    step_counter = 0
    current_price_initial = round(env.data.loc[env.current_step, "Close"], 2)
    price_history = [current_price_initial]
    portfolio_history = [float(env.net_worth)]  # Convert Decimal to float for plotting
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    prev_position = env.position
    prev_net_worth = env.net_worth  # Track previous net worth to determine if a trade was profitable
    entry_net_worth = env.net_worth  # Track net worth at entry for calculating trade profitability
    entry_positions = {-1: None, 1: None}  # Track entry net worth for both short and long positions
    
    # Only log initial state if verbose > 1
    if verbose > 1:
        logger.info(f"Initial position: {prev_position}, Initial observation: {obs}")

    while True:
        current_index = env.current_step
        current_price = round(env.data.loc[current_index, "Close"], 2)
        current_date = test_data.index[current_index]  # Get current date from the index
        action, _ = model.predict(obs, deterministic=deterministic)  # Use deterministic parameter
        action_counts[int(action)] += 1  # Count this action
        
        # Debug log only if verbose > 1
        if verbose > 1:
            logger.debug(f"Step {step_counter}: Action={action}, Position={env.position}, Price={current_price}")
        
        prev_position = env.position
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        # Record trade if a position change occurred
        if env.position != prev_position:
            # Only log position changes if verbose > 1
            if verbose > 1:
                logger.info(f"Position changed from {prev_position} to {env.position} at step {step_counter}, price {current_price}")
            
            # Count trades more accurately:
            # 1. Closing a position (going from non-zero to zero) counts as 1 trade
            # 2. Opening a position (going from zero to non-zero) doesn't add to trade count (it's part of entry/exit pair)
            # 3. Switching positions directly (e.g., long to short) counts as 2 trades (close previous + open new)
            
            if prev_position != 0:
                # We're closing or changing a position
                trade_count += 1
                
                # Check if the trade was profitable based on the position type
                position_type = "long" if prev_position > 0 else "short"
                position_entry_worth = entry_positions.get(prev_position)
                
                if position_entry_worth is not None:
                    # For long positions, profit when exit value > entry value
                    # For short positions, profit when exit value < entry value
                    is_profitable = False
                    if prev_position > 0:  # Long position
                        is_profitable = env.net_worth > position_entry_worth
                    else:  # Short position
                        is_profitable = env.net_worth > position_entry_worth
                    
                    if is_profitable:
                        profitable_trades += 1
                        if verbose > 1:
                            logger.info(f"Profitable {position_type} trade completed. Profit: {float(env.net_worth - position_entry_worth):.2f}")
                    elif verbose > 1:
                        logger.info(f"Unprofitable {position_type} trade completed. Loss: {float(position_entry_worth - env.net_worth):.2f}")
            
            # If we're opening a new position, record the entry value
            if env.position != 0:
                entry_positions[env.position] = env.net_worth
            
            # Record the trade
            if (prev_position, env.position) in [(0, 1), (-1, 1), (-1, 0)]:
                buy_dates.append(current_date)
                buy_prices.append(current_price)
                trade_type = "Buy"
            elif (prev_position, env.position) in [(0, -1), (1, -1), (1, 0)]:
                sell_dates.append(current_date)
                sell_prices.append(current_price)
                trade_type = "Sell"
                
            # Ensure portfolio value is never negative in the trade history
            portfolio_value = max(Decimal('0.01'), env.net_worth)
            
            trade_history.append({
                "date": current_date,
                "trade_type": trade_type,
                "price": current_price,
                "portfolio_value": float(portfolio_value),  # Convert Decimal to float for serialization
                "profitable": float(portfolio_value) > float(prev_net_worth),  # Track if this specific action was profitable
                "position_from": prev_position,
                "position_to": env.position
            })
            
            prev_net_worth = env.net_worth

        step_counter += 1
        dates.append(current_date)
        price_history.append(current_price)
        portfolio_history.append(float(money.format_money(env.net_worth, 2)))  # Format and convert to float for plotting

        if done:
            break

    # Calculate percentage return using the money module
    total_return_pct = money.calculate_return_pct(env.net_worth, initial_net_worth)
    
    # Calculate hit rate (percentage of profitable trades)
    hit_rate = 0
    if trade_count > 0:
        hit_rate = (profitable_trades / trade_count) * 100
    
    # Only log detailed evaluation if verbose > 0
    if verbose > 0:
        logger.info("Evaluation completed: Final portfolio value: $%s (%.2f%% return), Total trades: %d",
                    money.format_money_str(env.net_worth), float(total_return_pct), trade_count)
        
        if trade_count > 0:
            logger.info("Hit rate: %.2f%% (%d/%d profitable trades)", hit_rate, profitable_trades, trade_count)
        
        # Log action distribution only if verbose > 1
        if verbose > 1:
            logger.info(f"Action distribution - Long: {action_counts[0]}, Short: {action_counts[1]}")

    return {
        "final_portfolio_value": float(money.format_money(env.net_worth, 2)),
        "total_return_pct": float(money.format_money(total_return_pct, 2)),
        "trade_count": trade_count,
        "profitable_trades": profitable_trades,
        "hit_rate": hit_rate,
        "final_position": env.position,
        "dates": dates,
        "price_history": price_history,
        "portfolio_history": portfolio_history,
        "trade_history": trade_history,
        "buy_dates": buy_dates,
        "buy_prices": buy_prices,
        "sell_dates": sell_dates,
        "sell_prices": sell_prices,
    }

def plot_results(results):
    """
    Plot BTC price, trade signals, and portfolio value over time using Plotly.
    """
    dates = results["dates"]
    price_history = results["price_history"]
    portfolio_history = results["portfolio_history"]

    # Create subplots with 2 rows in one column and shared X-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.15,  # Increased spacing between subplots
        subplot_titles=(
            "Asset Price and Trade Signals",
            "Portfolio Value"  # Simplified title - we'll add the details in annotations
        )
    )

    # Plot BTC Price line in the first row
    fig.add_trace(
        go.Scatter(x=dates, y=price_history, name="Price", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Plot Buy signals with triangle-up markers
    fig.add_trace(
        go.Scatter(
            x=results["buy_dates"],
            y=results["buy_prices"],
            name="Buy",
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=12)
        ),
        row=1, col=1
    )
    
    # Plot Sell signals with triangle-down markers
    fig.add_trace(
        go.Scatter(
            x=results["sell_dates"],
            y=results["sell_prices"],
            name="Sell",
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12)
        ),
        row=1, col=1
    )
    
    # Plot Portfolio Value in the second row
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=portfolio_history, 
            name="Portfolio Value", 
            line=dict(color="purple"),
            showlegend=False  # Remove duplicate legend entry
        ),
        row=2, col=1
    )
    
    # Add portfolio performance annotation
    fig.add_annotation(
        text=f"Initial: ${config['environment']['initial_balance']:,.2f}<br>Final: ${results['final_portfolio_value']:,.2f}<br>Return: {results['total_return_pct']}%",
        xref="paper", yref="paper",
        x=1.0, y=0.4,  # Position annotation on right side of bottom plot
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    
    # Update layout for titles and axes
    fig.update_layout(
        height=800,
        width=1200,  # Increased width
        showlegend=True,
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title="Asset Price ($)", tickprefix="$"),
        xaxis2=dict(title="Date"),
        yaxis2=dict(title="Portfolio Value ($)", tickprefix="$"),
        legend=dict(
            x=0.5,  # Center the legend
            y=1.15,  # Move legend above the plot
            xanchor="center",
            orientation="h"
        ),
        margin=dict(l=60, r=60, t=100, b=50)  # Increased margins
    )
    
    fig.show()

def plot_training_progress(all_results):
    """
    Plot the training progress across iterations.
    
    Args:
        all_results (list): List of evaluation results from each training iteration.
    """
    iterations = list(range(len(all_results)))
    returns = [result["total_return_pct"] for result in all_results]
    trade_counts = [result["trade_count"] for result in all_results]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=iterations, y=returns, name="Return %", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=iterations, y=trade_counts, name="Trade Count", line=dict(color="red", width=2, dash="dot")),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text="Training Progress by Iteration",
        xaxis=dict(title="Iteration"),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        height=600,
        width=1000,
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Return %", secondary_y=False)
    fig.update_yaxes(title_text="Trade Count", secondary_y=True)
    
    fig.show()

def save_trade_history(trade_history, filename="trade_history.csv"):
    """
    Export the trade history to a CSV file.

    Args:
        trade_history (list): List of trade event dictionaries.
        filename (str): Output filename (default "trade_history.csv").
    """
    trade_history_df = pd.DataFrame(trade_history)
    trade_history_df.to_csv(filename, index=False)
    logger.info("Trade history saved to %s", filename)

def main():
    # Load data using settings from YAML with three-way split and direct Yahoo Finance download
    train_data, validation_data, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        use_yfinance=True  # Use Yahoo Finance directly
    )

    # Use the iterative training approach with validation data for model selection
    initial_timesteps = config["training"].get("total_timesteps", 50000)
    max_iterations = config["training"].get("max_iterations", 20)
    improvement_threshold = config["training"].get("improvement_threshold", 0.1)
    additional_timesteps = config["training"].get("additional_timesteps", 10000)
    n_stagnant_loops = config["training"].get("n_stagnant_loops", 3)
    
    logger.info(f"Starting iterative training with initial_timesteps={initial_timesteps}, "
               f"max_iterations={max_iterations}, n_stagnant_loops={n_stagnant_loops}, "
               f"improvement_threshold={improvement_threshold}%, "
               f"additional_timesteps={additional_timesteps}")
    
    # Train using train data and validate using validation data
    best_model, best_validation_results, all_validation_results = train_agent_iteratively(
        train_data, 
        validation_data,  # Use validation data for model selection
        initial_timesteps=initial_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=n_stagnant_loops,
        improvement_threshold=improvement_threshold,
        additional_timesteps=additional_timesteps
    )
    
    # Final evaluation on the test data (previously unseen)
    logger.info("Performing final evaluation on test data (previously unseen)")
    test_results = evaluate_agent(best_model, test_data, verbose=config["training"].get("verbose", 1), deterministic=True)
    
    # Log detailed evaluation results for the best model on test data
    logger.info("Final Test Results:")
    logger.info("Final Portfolio Value: $%.2f", test_results["final_portfolio_value"])
    logger.info("Total Return: %.2f%%", test_results["total_return_pct"])
    logger.info("Total Trades Executed: %d", test_results["trade_count"])
    logger.info("Final Position: %d", test_results["final_position"])
    
    # Save trade history to CSV
    save_trade_history(test_results["trade_history"], "best_model_test_trade_history.csv")
    
    # Plot results for the best model on test data
    logger.info("Plotting test evaluation results...")
    plot_results(test_results)
    
    # Plot training progress using validation results
    logger.info("Plotting training progress across iterations...")
    plot_training_progress(all_validation_results)
    
    # Also save validation trade history for comparison
    save_trade_history(best_validation_results["trade_history"], "best_model_validation_trade_history.csv")

if __name__ == "__main__":
    main()

