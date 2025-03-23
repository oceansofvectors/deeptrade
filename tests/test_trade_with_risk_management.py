import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import pytz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade import trade_with_risk_management
import money

class TestTradeWithRiskManagement(unittest.TestCase):
    """
    Test suite for the trade_with_risk_management function in trade.py.
    
    These tests focus on testing the integration of stop loss, take profit, 
    and trailing stop functionality with the trading logic.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a small test dataset
        self.eastern = pytz.timezone('US/Eastern')
        dates = []
        current_date = datetime(2023, 1, 1, 9, 30, 0)
        
        # Create 100 5-minute candles
        for i in range(100):
            dates.append(pd.Timestamp(current_date, tz=self.eastern))
            current_date += timedelta(minutes=5)
            
        # Create a simple price series that goes up then down
        close_prices = []
        high_prices = []
        low_prices = []
        open_prices = []
        
        # Start at 15000
        base_price = 15000
        
        # First 40 candles: uptrend
        for i in range(40):
            # Add some random walk with upward bias
            step = np.random.normal(2, 5)  # Mean positive step
            base_price += step
            
            # Create OHLC data for this candle
            open_price = base_price - np.random.normal(0, 3)
            close_price = base_price 
            high_price = max(open_price, close_price) + np.random.normal(5, 3)
            low_price = min(open_price, close_price) - np.random.normal(5, 3)
            
            open_prices.append(open_price)
            close_prices.append(close_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            
        # Next 60 candles: downtrend
        for i in range(60):
            # Add some random walk with downward bias
            step = np.random.normal(-2, 5)  # Mean negative step
            base_price += step
            
            # Create OHLC data for this candle
            open_price = base_price - np.random.normal(0, 3)
            close_price = base_price 
            high_price = max(open_price, close_price) + np.random.normal(5, 3)
            low_price = min(open_price, close_price) - np.random.normal(5, 3)
            
            open_prices.append(open_price)
            close_prices.append(close_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            
        # Create the test DataFrame
        self.test_data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices
        }, index=dates)
        
        # Add some simple features for the environment
        self.test_data['SMA_10'] = self.test_data['Close'].rolling(10).mean()
        self.test_data['RSI'] = 50 + np.random.normal(0, 10, len(self.test_data))  # Random RSI values
        
        # Drop NaN values
        self.test_data = self.test_data.dropna()
        
        # Initial balance for tests
        self.initial_balance = 10000.0
        
    @patch('trade.PPO')
    @patch('trade.TradingEnv')
    def test_stop_loss_integration(self, mock_env_class, mock_ppo_class):
        """
        Test that stop loss is correctly applied in the trading function.
        """
        # Configure the mock PPO model to always predict "buy" (long)
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)  # 0 means "buy" in our env
        mock_ppo_class.load.return_value = mock_model
        
        # Configure the mock environment
        mock_env = MagicMock()
        mock_env.current_step = 0
        mock_env.reset.return_value = (np.zeros(10), {})  # Dummy observation
        
        # Environment step counter
        step_counter = 0
        
        # Make the env.step method advance the step counter and return the appropriate state
        def mock_step(action):
            nonlocal step_counter
            step_counter += 1
            # Return dummy values with done=True at the end
            mock_env.current_step = step_counter
            done = step_counter >= len(self.test_data) - 1
            return (np.zeros(10), 0, done, False, {})
            
        mock_env.step.side_effect = mock_step
        mock_env_class.return_value = mock_env
        
        # Call the function with stop loss only
        results = trade_with_risk_management(
            model_path="dummy_model",
            test_data=self.test_data,
            stop_loss_pct=1.0,  # 1% stop loss
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            verbose=0,
            close_at_end_of_day=True  # Close positions at the end of the trading day
        )
        
        # Check the results
        # We expect the stop loss to have been triggered at least once
        exit_reasons = results["exit_reasons"]
        self.assertGreater(exit_reasons["stop_loss"], 0)
        
    @patch('trade.PPO')
    @patch('trade.TradingEnv')
    def test_take_profit_integration(self, mock_env_class, mock_ppo_class):
        """
        Test that take profit is correctly applied in the trading function.
        """
        # Configure the mock PPO model to always predict "buy" (long)
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)  # 0 means "buy" in our env
        mock_ppo_class.load.return_value = mock_model
        
        # Configure the mock environment
        mock_env = MagicMock()
        mock_env.current_step = 0
        mock_env.reset.return_value = (np.zeros(10), {})  # Dummy observation
        
        # Environment step counter
        step_counter = 0
        
        # Make the env.step method advance the step counter and return the appropriate state
        def mock_step(action):
            nonlocal step_counter
            step_counter += 1
            # Return dummy values with done=True at the end
            mock_env.current_step = step_counter
            done = step_counter >= len(self.test_data) - 1
            return (np.zeros(10), 0, done, False, {})
            
        mock_env.step.side_effect = mock_step
        mock_env_class.return_value = mock_env
        
        # Call the function with take profit only
        results = trade_with_risk_management(
            model_path="dummy_model",
            test_data=self.test_data,
            stop_loss_pct=None,
            take_profit_pct=2.0,  # 2% take profit
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            verbose=0,
            close_at_end_of_day=True  # Close positions at the end of the trading day
        )
        
        # Check the results
        # We expect the take profit to have been triggered at least once
        exit_reasons = results["exit_reasons"]
        self.assertGreater(exit_reasons["take_profit"], 0)
        
    @patch('trade.PPO')
    @patch('trade.TradingEnv')
    def test_trailing_stop_integration(self, mock_env_class, mock_ppo_class):
        """
        Test that trailing stop is correctly applied in the trading function.
        """
        # Configure the mock PPO model to always predict "buy" (long)
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)  # 0 means "buy" in our env
        mock_ppo_class.load.return_value = mock_model
        
        # Configure the mock environment
        mock_env = MagicMock()
        mock_env.current_step = 0
        mock_env.reset.return_value = (np.zeros(10), {})  # Dummy observation
        
        # Environment step counter
        step_counter = 0
        
        # Make the env.step method advance the step counter and return the appropriate state
        def mock_step(action):
            nonlocal step_counter
            step_counter += 1
            # Return dummy values with done=True at the end
            mock_env.current_step = step_counter
            done = step_counter >= len(self.test_data) - 1
            return (np.zeros(10), 0, done, False, {})
            
        mock_env.step.side_effect = mock_step
        mock_env_class.return_value = mock_env
        
        # Call the function with trailing stop only
        results = trade_with_risk_management(
            model_path="dummy_model",
            test_data=self.test_data,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.5,  # 0.5% trailing stop
            initial_balance=self.initial_balance,
            verbose=0,
            close_at_end_of_day=True  # Close positions at the end of the trading day
        )
        
        # Check the results
        # We expect the trailing stop to have been triggered at least once
        exit_reasons = results["exit_reasons"]
        self.assertGreater(exit_reasons["trailing_stop"], 0)
        
    @patch('trade.PPO')
    @patch('trade.TradingEnv')
    def test_combined_risk_management_integration(self, mock_env_class, mock_ppo_class):
        """
        Test that all risk management strategies work together in the trading function.
        """
        # Configure the mock PPO model to alternate between buy and sell
        predict_calls = 0
        
        def alternating_predict(obs, deterministic=False):
            nonlocal predict_calls
            action = predict_calls % 2  # Alternate between 0 (buy) and 1 (sell)
            predict_calls += 1
            return (action, None)
            
        mock_model = MagicMock()
        mock_model.predict.side_effect = alternating_predict
        mock_ppo_class.load.return_value = mock_model
        
        # Configure the mock environment
        mock_env = MagicMock()
        mock_env.current_step = 0
        mock_env.reset.return_value = (np.zeros(10), {})  # Dummy observation
        
        # Environment step counter
        step_counter = 0
        
        # Make the env.step method advance the step counter and return the appropriate state
        def mock_step(action):
            nonlocal step_counter
            step_counter += 1
            # Return dummy values with done=True at the end
            mock_env.current_step = step_counter
            done = step_counter >= len(self.test_data) - 1
            return (np.zeros(10), 0, done, False, {})
            
        mock_env.step.side_effect = mock_step
        mock_env_class.return_value = mock_env
        
        # Call the function with all risk management strategies
        results = trade_with_risk_management(
            model_path="dummy_model",
            test_data=self.test_data,
            stop_loss_pct=2.0,       # 2% stop loss
            take_profit_pct=3.0,     # 3% take profit
            trailing_stop_pct=1.0,   # 1% trailing stop
            initial_balance=self.initial_balance,
            verbose=0,
            close_at_end_of_day=True  # Close positions at the end of the trading day
        )
        
        # Check the results
        exit_reasons = results["exit_reasons"]
        
        # We expect at least one exit for each reason (except end_of_period which might not happen)
        self.assertGreaterEqual(exit_reasons["stop_loss"] + 
                               exit_reasons["take_profit"] + 
                               exit_reasons["trailing_stop"] + 
                               exit_reasons["model_signal"], 1)
        
        # Portfolio value should be different from initial value
        self.assertNotEqual(results["final_portfolio_value"], self.initial_balance)
        
        # There should be trades recorded
        self.assertGreater(len(results["trade_history"]), 0)

    @patch('trade.PPO')
    @patch('trade.TradingEnv')
    def test_end_of_day_closing(self, mock_env_class, mock_ppo_class):
        """
        Test that positions are properly closed at the end of the trading day.
        """
        # Configure the mock PPO model to always predict "buy" (long)
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)  # 0 means "buy" in our env
        mock_ppo_class.load.return_value = mock_model
        
        # Configure the mock environment
        mock_env = MagicMock()
        mock_env.current_step = 0
        mock_env.reset.return_value = (np.zeros(10), {})  # Dummy observation
        
        # Create test data with different days
        # We'll use a smaller dataset with two trading days
        eastern = pytz.timezone('US/Eastern')
        dates = []
        current_date = datetime(2023, 1, 1, 9, 30, 0)
        
        # Day 1: 10 data points
        for i in range(10):
            dates.append(pd.Timestamp(current_date, tz=eastern))
            current_date += timedelta(minutes=5)
            
        # Day 2: 10 data points (next day)
        current_date = datetime(2023, 1, 2, 9, 30, 0)
        for i in range(10):
            dates.append(pd.Timestamp(current_date, tz=eastern))
            current_date += timedelta(minutes=5)
            
        # Create price data
        prices = [15000 + i for i in range(20)]
        test_data_eod = pd.DataFrame({
            'Open': prices,
            'High': [p + 5 for p in prices],
            'Low': [p - 5 for p in prices],
            'Close': prices
        }, index=dates)
        
        # Environment step counter with day transition detection
        step_counter = 0
        
        # Make the env.step method advance the step counter and handle day transitions
        def mock_step(action):
            nonlocal step_counter
            mock_env.current_step = step_counter
            
            # Check if we're at the end of day 1 (index 9 to 10)
            is_day_transition = step_counter == 9
            
            step_counter += 1
            
            # Return done only at the very end
            done = step_counter >= 19
            return (np.zeros(10), 0, done, False, {})
            
        mock_env.step.side_effect = mock_step
        mock_env_class.return_value = mock_env
        
        # Call the function with close_at_end_of_day=True
        results = trade_with_risk_management(
            model_path="dummy_model",
            test_data=test_data_eod,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            verbose=0,
            close_at_end_of_day=True  # Close positions at the end of the trading day
        )
        
        # Check the results
        exit_reasons = results["exit_reasons"]
        
        # We expect at least one "end_of_day" exit
        self.assertGreater(exit_reasons.get("end_of_day", 0), 0, 
                          "Should have at least one end_of_day exit reason")
        
        # There should be no positions held at the end
        self.assertEqual(results["final_position"], 0, "Should have no positions at the end")
        
if __name__ == "__main__":
    unittest.main() 