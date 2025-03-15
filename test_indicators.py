import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from get_data import get_data
from environment import TradingEnv
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_indicators():
    """
    Test that all technical indicators are being calculated correctly.
    """
    # Get data with all indicators
    logger.info("Fetching data with all technical indicators...")
    train_data, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=config["data"]["train_ratio"]
    )
    
    # Print the columns to see which indicators were calculated
    logger.info("Available columns in the dataset:")
    for col in train_data.columns:
        logger.info(f"- {col}")
    
    # Create the environment with the data
    logger.info("Creating trading environment...")
    env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["training"]["fee_rate"]
    )
    
    # Print the technical indicators used in the observation space
    logger.info("Technical indicators in observation space:")
    for indicator in env.technical_indicators:
        logger.info(f"- {indicator}")
    
    # Print observation space shape
    logger.info(f"Observation space shape: {env.observation_space.shape}")
    
    # Reset the environment and get the initial observation
    obs, _ = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial observation: {obs}")
    
    # Take a random action and check the next observation
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    logger.info(f"Action: {action}, Reward: {reward}")
    logger.info(f"Next observation shape: {next_obs.shape}")
    
    # Plot some of the indicators for visualization
    plot_indicators(train_data)
    
    return train_data, env

def plot_indicators(data, n_samples=500):
    """
    Plot some of the technical indicators for visualization.
    
    Args:
        data (pd.DataFrame): DataFrame containing price and indicator data.
        n_samples (int): Number of samples to plot.
    """
    # Use the last n_samples for plotting
    if len(data) > n_samples:
        plot_data = data.iloc[-n_samples:].copy()
    else:
        plot_data = data.copy()
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    
    # Plot 1: Price and moving averages
    axes[0].set_title('Price and Moving Averages')
    axes[0].plot(plot_data.index, plot_data['Close'], label='Close Price', color='black')
    if 'SMA' in plot_data.columns:
        axes[0].plot(plot_data.index, plot_data['SMA'], label='SMA', color='blue')
    if 'EMA' in plot_data.columns:
        axes[0].plot(plot_data.index, plot_data['EMA'], label='EMA', color='red')
    if 'PSAR' in plot_data.columns:
        axes[0].scatter(plot_data.index, plot_data['PSAR'], label='PSAR', color='green', s=10)
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Oscillators (RSI, Stochastic)
    axes[1].set_title('Oscillators')
    if 'RSI' in plot_data.columns:
        axes[1].plot(plot_data.index, plot_data['RSI'] * 100, label='RSI', color='purple')
    if 'STOCH_K' in plot_data.columns:
        axes[1].plot(plot_data.index, plot_data['STOCH_K'] * 100, label='Stoch K', color='blue')
    if 'STOCH_D' in plot_data.columns:
        axes[1].plot(plot_data.index, plot_data['STOCH_D'] * 100, label='Stoch D', color='red')
    # Add horizontal lines at 30 and 70 for reference
    axes[1].axhline(y=30, color='gray', linestyle='--')
    axes[1].axhline(y=70, color='gray', linestyle='--')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: MACD
    axes[2].set_title('MACD')
    if 'MACD' in plot_data.columns:
        axes[2].plot(plot_data.index, plot_data['MACD'], label='MACD', color='blue')
    if 'MACD_SIGNAL' in plot_data.columns:
        axes[2].plot(plot_data.index, plot_data['MACD_SIGNAL'], label='Signal', color='red')
    if 'MACD_HIST' in plot_data.columns:
        axes[2].bar(plot_data.index, plot_data['MACD_HIST'], label='Histogram', color='green', alpha=0.5)
    axes[2].axhline(y=0, color='gray', linestyle='--')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Trend Indicators (ADX, CCI)
    axes[3].set_title('Trend Indicators')
    if 'ADX' in plot_data.columns:
        axes[3].plot(plot_data.index, plot_data['ADX'] * 100, label='ADX', color='blue')
    if 'CCI' in plot_data.columns:
        axes[3].plot(plot_data.index, plot_data['CCI'] * 100, label='CCI', color='red')
    if 'trend_direction' in plot_data.columns:
        # Plot trend direction as a step function
        axes[3].step(plot_data.index, plot_data['trend_direction'] * 50 + 50, label='Trend Direction', color='green')
    axes[3].axhline(y=25, color='gray', linestyle='--')  # ADX reference line
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('technical_indicators.png')
    plt.close()
    
    logger.info("Technical indicators plot saved as 'technical_indicators.png'")

if __name__ == "__main__":
    train_data, env = test_indicators() 