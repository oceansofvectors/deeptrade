import logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import TradingEnv
from get_data import get_data

from config import config

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

def train_agent_iteratively(train_data, test_data, initial_timesteps: int, max_iterations: int = 20, 
                           n_stagnant_loops: int = 3, improvement_threshold: float = 0.1, additional_timesteps: int = 10000):
    """
    Train a PPO model iteratively based on evaluation results.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
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
    
    # Evaluate initial model (now using non-deterministic evaluation)
    if verbose_level > 0:
        logger.info("Evaluating model with non-deterministic action selection")
    results = evaluate_agent(model, test_data, verbose=verbose_level)
    best_return = results["total_return_pct"]
    best_model = model
    best_results = results
    
    # Save the initial model as the best model so far
    best_model.save("best_model")
    
    logger.info(f"Initial training completed. Return: {best_return:.2f}%, Portfolio: ${results['final_portfolio_value']:.2f}")
    
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
        
        # Evaluate the model (non-deterministic)
        results = evaluate_agent(model, test_data, verbose=verbose_level)
        current_return = results["total_return_pct"]
        all_results.append(results)
        
        # Calculate improvement
        improvement = current_return - best_return
        logger.info(f"Iteration {iteration} - Return: {current_return:.2f}%, Portfolio: ${results['final_portfolio_value']:.2f}, Improvement: {improvement:.2f}%")
        
        # Check if this is the best model so far
        if current_return > best_return + improvement_threshold:
            best_return = current_return
            best_model = model
            best_results = results
            logger.info(f"New best model found! Return: {best_return:.2f}%, Portfolio: ${best_results['final_portfolio_value']:.2f}")
            
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
    
    logger.info(f"Iterative training completed. Best return: {best_return:.2f}%, Portfolio: ${best_results['final_portfolio_value']:.2f}")
    return best_model, best_results, all_results

def evaluate_agent(model, test_data, fee_rate: float = 0.0, verbose: int = 1):
    """
    Evaluate the trained model on test data and record trade history.

    Args:
        model: Trained PPO model.
        test_data (pd.DataFrame): Testing dataset.
        fee_rate (float): Trading fee rate (default 0.0 - no fees).
        verbose (int): Verbosity level for logging (default 1).

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
    action_counts = {0: 0, 1: 0}  # Track action distribution (0: long/buy, 1: short/sell)

    # Records for plotting
    # Get the actual dates from the index
    dates = [test_data.index[env.current_step]]
    step_counter = 0
    current_price_initial = round(env.data.loc[env.current_step, "Close"], 2)
    price_history = [current_price_initial]
    portfolio_history = [env.net_worth]
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    prev_position = env.position
    
    # Only log initial state if verbose > 1
    if verbose > 1:
        logger.info(f"Initial position: {prev_position}, Initial observation: {obs}")

    while True:
        current_index = env.current_step
        current_price = round(env.data.loc[current_index, "Close"], 2)
        current_date = test_data.index[current_index]  # Get current date from the index
        action, _ = model.predict(obs, deterministic=False)  # Changed from True to False for non-deterministic evaluation
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
            
            trade_count += 1

            if (prev_position, env.position) in [(0, 1), (-1, 1)]:
                buy_dates.append(current_date)
                buy_prices.append(current_price)
                trade_type = "Buy"
            elif (prev_position, env.position) in [(0, -1), (1, -1)]:
                sell_dates.append(current_date)
                sell_prices.append(current_price)
                trade_type = "Sell"
            elif prev_position == 1 and env.position == 0:
                sell_dates.append(current_date)
                sell_prices.append(current_price)
                trade_type = "Sell"
            elif prev_position == -1 and env.position == 0:
                buy_dates.append(current_date)
                buy_prices.append(current_price)
                trade_type = "Buy"
                
            # Ensure portfolio value is never negative in the trade history
            portfolio_value = max(0.01, env.net_worth)
            
            trade_history.append({
                "date": current_date,
                "trade_type": trade_type,
                "price": current_price,
                "portfolio_value": portfolio_value
            })

        step_counter += 1
        dates.append(current_date)
        price_history.append(current_price)
        portfolio_history.append(round(env.net_worth, 2))

        if done:
            break

    # Calculate percentage return
    total_return_pct = ((env.net_worth - initial_net_worth) / initial_net_worth) * 100
    
    # Only log detailed evaluation if verbose > 0
    if verbose > 0:
        logger.info("Evaluation completed: Final portfolio value: $%.2f (%.2f%% return), Total trades: %d",
                    env.net_worth, total_return_pct, trade_count)
        
        # Log action distribution only if verbose > 1
        if verbose > 1:
            logger.info(f"Action distribution - Long: {action_counts[0]}, Short: {action_counts[1]}")

    return {
        "final_portfolio_value": round(env.net_worth, 2),
        "total_return_pct": round(total_return_pct, 2),
        "trade_count": trade_count,
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
    # Load data using settings from YAML
    train_data, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=config["data"]["train_ratio"]
    )

    # Use the iterative training approach
    initial_timesteps = config["training"].get("total_timesteps", 50000)
    max_iterations = config["training"].get("max_iterations", 20)
    improvement_threshold = config["training"].get("improvement_threshold", 0.1)
    additional_timesteps = config["training"].get("additional_timesteps", 10000)
    n_stagnant_loops = config["training"].get("n_stagnant_loops", 3)
    
    logger.info(f"Starting iterative training with initial_timesteps={initial_timesteps}, "
               f"max_iterations={max_iterations}, n_stagnant_loops={n_stagnant_loops}, "
               f"improvement_threshold={improvement_threshold}%, "
               f"additional_timesteps={additional_timesteps}")
    
    best_model, best_results, all_results = train_agent_iteratively(
        train_data, 
        test_data, 
        initial_timesteps=initial_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=n_stagnant_loops,
        improvement_threshold=improvement_threshold,
        additional_timesteps=additional_timesteps
    )
    
    # Log detailed evaluation results for the best model
    logger.info("Best Model Evaluation Results:")
    logger.info("Final Portfolio Value: $%.2f", best_results["final_portfolio_value"])
    logger.info("Total Return: %.2f%%", best_results["total_return_pct"])
    logger.info("Total Trades Executed: %d", best_results["trade_count"])
    logger.info("Final Position: %d", best_results["final_position"])
    
    # Save trade history to CSV
    save_trade_history(best_results["trade_history"], "best_model_trade_history.csv")
    
    # Plot results for the best model
    logger.info("Plotting best model evaluation results...")
    plot_results(best_results)
    
    # Plot training progress
    logger.info("Plotting training progress across iterations...")
    plot_training_progress(all_results)

if __name__ == "__main__":
    main()

