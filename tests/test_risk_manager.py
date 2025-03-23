import unittest
from decimal import Decimal
import pandas as pd
from datetime import datetime
import sys
import os
import pytz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade import RiskManager
import money

class TestRiskManager(unittest.TestCase):
    """
    Test suite for the RiskManager class in trade.py.
    
    These tests focus on the stop loss, take profit, and trailing stop functionality.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Common test parameters
        self.initial_balance = 10000.0
        self.transaction_cost = 0.0
        self.eastern = pytz.timezone('US/Eastern')
        
    def test_stop_loss_long_position(self):
        """
        Test that stop loss exits a long position when the price drops enough.
        """
        # Create a risk manager with stop loss only (1% of portfolio)
        risk_manager = RiskManager(
            stop_loss_pct=1.0,  # 1% stop loss
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # The contract price is 15000 and we have 1 contract
        # Each NQ point is worth $20
        # For a 1% stop loss on a $10,000 portfolio, we need to lose $100 
        # With a 15000 entry price, we need to drop to 14995 to lose $100
        current_price = 14996.0  # Not enough to trigger stop loss
        high_price = 15000.0
        low_price = 14996.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Now test price below stop loss threshold
        current_price = 14994.0  # Should trigger stop loss
        low_price = 14994.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify exit was triggered with stop loss reason
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "stop_loss")
    
    def test_stop_loss_short_position(self):
        """
        Test that stop loss exits a short position when the price rises enough.
        """
        # Create a risk manager with stop loss only (1% of portfolio)
        risk_manager = RiskManager(
            stop_loss_pct=1.0,  # 1% stop loss
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a short position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(-1, entry_price, entry_date, 1)
        
        # The contract price is 15000 and we have 1 contract
        # Each NQ point is worth $20
        # For a 1% stop loss on a $10,000 portfolio, we need to lose $100
        # With a 15000 entry price, we need to rise to 15005 to lose $100
        current_price = 15004.0  # Not enough to trigger stop loss
        high_price = 15004.0
        low_price = 15000.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Now test price above stop loss threshold
        current_price = 15006.0  # Should trigger stop loss
        high_price = 15006.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify exit was triggered with stop loss reason
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "stop_loss")
    
    def test_take_profit_long_position(self):
        """
        Test that take profit exits a long position when the price rises enough.
        """
        # Create a risk manager with take profit only (2% of portfolio)
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=2.0,  # 2% take profit
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # The contract price is 15000 and we have 1 contract
        # Each NQ point is worth $20
        # For a 2% take profit on a $10,000 portfolio, we need to gain $200
        # With a 15000 entry price, we need to rise to 15010 to gain $200
        current_price = 15009.0  # Not enough to trigger take profit
        high_price = 15009.0
        low_price = 15000.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Now test price above take profit threshold
        current_price = 15011.0  # Should trigger take profit
        high_price = 15011.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify exit was triggered with take profit reason
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "take_profit")
    
    def test_take_profit_short_position(self):
        """
        Test that take profit exits a short position when the price falls enough.
        """
        # Create a risk manager with take profit only (2% of portfolio)
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=2.0,  # 2% take profit
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a short position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(-1, entry_price, entry_date, 1)
        
        # The contract price is 15000 and we have 1 contract
        # Each NQ point is worth $20
        # For a 2% take profit on a $10,000 portfolio, we need to gain $200
        # With a 15000 entry price, we need to fall to 14990 to gain $200
        current_price = 14991.0  # Not enough to trigger take profit
        high_price = 15000.0
        low_price = 14991.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Now test price below take profit threshold
        current_price = 14989.0  # Should trigger take profit
        low_price = 14989.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify exit was triggered with take profit reason
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "take_profit")
    
    def test_trailing_stop_long_position(self):
        """
        Test that trailing stop exits a long position when the price falls enough from the peak.
        """
        # Create a risk manager with trailing stop only (0.5% of portfolio)
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.5,  # 0.5% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # First update to set a peak - need to have some gain first to calculate trailing stop
        high_price = 15100.0  # Up 100 points (2% gain)
        current_price = 15100.0  
        low_price = 15050.0
        
        # Update stops and check exits - this should set the highest_unrealized_profit_pct
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered, just setting peak
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Small retreat - not enough to trigger trailing stop
        # Our trailing stop is 0.5% of portfolio = $50
        # With $20/point, that's 2.5 points
        # So a drop of 2 points from peak shouldn't trigger
        current_price = 15098.0  # Down 2 points from peak
        high_price = 15100.0  # Same peak
        low_price = 15098.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Larger retreat - should trigger trailing stop
        # A drop of 3 points should be enough to trigger trailing stop
        current_price = 15097.0  # Down 3 points from peak
        high_price = 15100.0  # Same peak
        low_price = 15097.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify exit was triggered with trailing stop reason
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "trailing_stop")
    
    def test_trailing_stop_short_position(self):
        """
        Test that trailing stop exits a short position when the price rises enough from the trough.
        """
        # Create a risk manager with trailing stop only (0.5% of portfolio)
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.5,  # 0.5% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a short position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(-1, entry_price, entry_date, 1)
        
        # Price falls - set a new trough - need to have some gain first for trailing stop
        current_price = 14900.0  # Down 100 points (2% gain)
        high_price = 14950.0
        low_price = 14900.0
        
        # Update stops and check exits - this should set the highest_unrealized_profit_pct
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered, just setting trough
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Small rise - not enough to trigger trailing stop
        # Our trailing stop is 0.5% of portfolio = $50
        # With $20/point, that's 2.5 points
        # So a rise of 2 points from trough shouldn't trigger
        current_price = 14902.0  # Up 2 points from trough
        high_price = 14902.0
        low_price = 14900.0  # Same trough
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify no exit was triggered
        self.assertFalse(exit_triggered)
        self.assertEqual(reason, "")
        
        # Larger rise - should trigger trailing stop
        # A rise of 3 points should be enough to trigger trailing stop
        current_price = 14903.0  # Up 3 points from trough
        high_price = 14903.0
        low_price = 14900.0  # Same trough
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Verify exit was triggered with trailing stop reason
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "trailing_stop")
    
    def test_combined_risk_management(self):
        """
        Test that all risk management strategies work together appropriately.
        """
        # Create a risk manager with all strategies
        risk_manager = RiskManager(
            stop_loss_pct=2.0,           # 2% stop loss
            take_profit_pct=3.0,         # 3% take profit
            trailing_stop_pct=1.0,       # 1% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # First try a small price rise that won't trigger any exits
        # The contract price is 15000 and we have 1 contract
        # Each NQ point is worth $20
        # With 3% take profit on $10,000, we need $300 gain (15 points)
        # With 2% stop loss on $10,000, we need $200 loss (10 points)
        # With 1% trailing stop on $10,000, we need a $100 drop from peak (5 points)
        current_price = 15005.0  # Up 5 points (only 1% gain)
        high_price = 15005.0
        low_price = 15000.0
        
        # Update stops and check exits
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Should not trigger any exit yet
        self.assertFalse(exit_triggered)
        
        # Now test take profit
        current_price = 15020.0  # Up 20 points (4% gain)
        high_price = 15020.0
        low_price = 15000.0
        
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Should trigger take profit
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "take_profit")
        
        # Reset for next test
        risk_manager = RiskManager(
            stop_loss_pct=2.0,           # 2% stop loss
            take_profit_pct=3.0,         # 3% take profit
            trailing_stop_pct=1.0,       # 1% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # Test trailing stop - first establish a peak
        current_price = 15015.0  # Up 15 points (3% gain)
        high_price = 15015.0
        low_price = 15000.0
        
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # This should trigger take profit first
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "take_profit")
        
        # Reset for the next test
        risk_manager = RiskManager(
            stop_loss_pct=2.0,           # 2% stop loss
            take_profit_pct=6.0,         # Higher take profit to test trailing stop
            trailing_stop_pct=1.0,       # 1% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # Test trailing stop - first establish a peak
        current_price = 15015.0  # Up 15 points (3% gain)
        high_price = 15015.0
        low_price = 15000.0
        
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Should not trigger any exit yet
        self.assertFalse(exit_triggered)
        
        # Now fall from peak to trigger trailing stop
        current_price = 15009.0  # Down 6 points from peak
        high_price = 15015.0  # Peak stays the same
        low_price = 15009.0
        
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Should trigger trailing stop
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "trailing_stop")
        
        # Reset for stop loss test
        risk_manager = RiskManager(
            stop_loss_pct=2.0,           # 2% stop loss
            take_profit_pct=3.0,         # 3% take profit
            trailing_stop_pct=1.0,       # 1% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # Test stop loss
        current_price = 14989.0  # Down 11 points (just over 2% loss)
        high_price = 15000.0
        low_price = 14989.0
        
        risk_manager.update_stops(current_price, high_price, low_price)
        exit_triggered, reason = risk_manager.check_exits(current_price, high_price, low_price)
        
        # Should trigger stop loss
        self.assertTrue(exit_triggered)
        self.assertEqual(reason, "stop_loss")
    
    def test_exit_position_profit_calculation(self):
        """
        Test that exit_position correctly calculates profits.
        """
        # Create a risk manager without any risk management strategies
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # Exit at higher price
        exit_price = 15100.0  # 100 points profit
        exit_date = pd.Timestamp(datetime(2023, 1, 1, 11, 0, 0), tz=self.eastern)
        
        # Get initial portfolio value
        initial_portfolio = risk_manager.net_worth
        
        # Exit the position
        risk_manager.exit_position(exit_price, exit_date, "test")
        
        # Calculate expected profit
        # 100 points * $20/point * 1 contract = $2,000
        expected_profit = 100 * 20
        
        # Verify the profit calculation
        self.assertEqual(risk_manager.net_worth, initial_portfolio + expected_profit)
        
        # Check trade history
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["action"], "sell")
        self.assertEqual(last_trade["price"], exit_price)
        self.assertEqual(last_trade["profit"], expected_profit)
        self.assertEqual(last_trade["exit_reason"], "test")
        
        # Test short position
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a short position at price 15000
        entry_price = 15000.0
        risk_manager.enter_position(-1, entry_price, entry_date, 1)
        
        # Exit at lower price
        exit_price = 14900.0  # 100 points profit
        
        # Get initial portfolio value
        initial_portfolio = risk_manager.net_worth
        
        # Exit the position
        risk_manager.exit_position(exit_price, exit_date, "test")
        
        # Calculate expected profit
        # 100 points * $20/point * 1 contract = $2,000
        expected_profit = 100 * 20
        
        # Verify the profit calculation
        self.assertEqual(risk_manager.net_worth, initial_portfolio + expected_profit)
        
        # Check trade history
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["action"], "buy")
        self.assertEqual(last_trade["price"], exit_price)
        self.assertEqual(last_trade["profit"], expected_profit)
        self.assertEqual(last_trade["exit_reason"], "test")
        
    def test_transaction_costs(self):
        """
        Test that transaction costs are correctly applied to profits.
        """
        # Create a risk manager with transaction costs
        transaction_cost = 0.1  # 0.1% transaction cost
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=transaction_cost
        )
        
        # Enter a long position at price 15000
        entry_price = 15000.0
        entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        risk_manager.enter_position(1, entry_price, entry_date, 1)
        
        # Exit at higher price
        exit_price = 15100.0  # 100 points profit
        exit_date = pd.Timestamp(datetime(2023, 1, 1, 11, 0, 0), tz=self.eastern)
        
        # Get initial portfolio value
        initial_portfolio = risk_manager.net_worth
        
        # Exit the position
        risk_manager.exit_position(exit_price, exit_date, "test")
        
        # Calculate expected profit
        # 100 points * $20/point * 1 contract = $2,000 (gross profit)
        # Transaction cost = 15100 * 1 * 20 * 0.1% = $30.2
        gross_profit = money.to_decimal(100) * money.to_decimal(20)
        transaction_cost_amount = money.to_decimal(exit_price) * money.to_decimal(20) * (money.to_decimal(transaction_cost) / money.to_decimal(100))
        expected_net_profit = gross_profit - transaction_cost_amount
        
        # Verify the profit calculation (allowing for small rounding differences)
        profit_difference = abs(risk_manager.net_worth - (initial_portfolio + expected_net_profit))
        self.assertLess(float(profit_difference), 0.01)
        
        # Check trade history
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["action"], "sell")
        self.assertEqual(last_trade["price"], exit_price)
        
        # Allow for small rounding differences in profit
        profit_difference = abs(money.to_decimal(last_trade["profit"]) - expected_net_profit)
        self.assertLess(float(profit_difference), 0.01)
        
if __name__ == "__main__":
    unittest.main() 