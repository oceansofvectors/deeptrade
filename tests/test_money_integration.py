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

class TestMoneyIntegration(unittest.TestCase):
    """
    Test the integration between the money module and risk management features.
    
    These tests focus on verifying that monetary calculations are performed 
    correctly when using the risk management features.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Common test parameters
        self.initial_balance = Decimal('10000.0')
        self.transaction_cost = Decimal('0.0')
        self.eastern = pytz.timezone('US/Eastern')
        self.entry_date = pd.Timestamp(datetime(2023, 1, 1, 10, 0, 0), tz=self.eastern)
        self.exit_date = pd.Timestamp(datetime(2023, 1, 1, 11, 0, 0), tz=self.eastern)
        
    def test_money_conversion_in_risk_manager(self):
        """
        Test that values are correctly converted to Decimal in RiskManager.
        """
        # Create a risk manager with float values
        risk_manager = RiskManager(
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            trailing_stop_pct=0.5,
            initial_balance=10000.0,
            transaction_cost=0.1
        )
        
        # Verify converted to Decimal internally
        self.assertIsInstance(risk_manager.stop_loss_pct, Decimal)
        self.assertIsInstance(risk_manager.take_profit_pct, Decimal)
        self.assertIsInstance(risk_manager.trailing_stop_pct, Decimal)
        self.assertIsInstance(risk_manager.initial_balance, Decimal)
        self.assertIsInstance(risk_manager.transaction_cost, Decimal)
        self.assertIsInstance(risk_manager.net_worth, Decimal)
        
    def test_stop_loss_profit_calculation(self):
        """
        Test that profits are correctly calculated when a stop loss is triggered.
        """
        # Create a risk manager with stop loss
        risk_manager = RiskManager(
            stop_loss_pct=1.0,  # 1% stop loss
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position
        entry_price = 15000.0
        risk_manager.enter_position(1, entry_price, self.entry_date, 1)
        
        # Verify entry price is converted to Decimal
        self.assertIsInstance(risk_manager.entry_price, Decimal)
        self.assertEqual(risk_manager.entry_price, Decimal('15000.0'))
        
        # Simulate a price drop that triggers stop loss
        exit_price = 14800.0  # Down 200 points
        
        # Get portfolio value before exit
        initial_portfolio = risk_manager.net_worth
        
        # Exit position due to stop loss
        risk_manager.exit_position(exit_price, self.exit_date, "stop_loss")
        
        # Calculate expected loss
        # 200 points * $20/point * 1 contract = $4,000 loss
        expected_loss = money.to_decimal(200) * money.to_decimal(20)
        expected_portfolio = initial_portfolio - expected_loss
        
        # Check portfolio value
        self.assertEqual(risk_manager.net_worth, expected_portfolio)
        
        # Check trade record
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["exit_reason"], "stop_loss")
        self.assertEqual(last_trade["profit"], float(-expected_loss))
        
    def test_take_profit_calculation(self):
        """
        Test that profits are correctly calculated when take profit is triggered.
        """
        # Create a risk manager with take profit
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=2.0,  # 2% take profit
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position
        entry_price = 15000.0
        risk_manager.enter_position(1, entry_price, self.entry_date, 1)
        
        # Simulate a price increase that triggers take profit
        exit_price = 15200.0  # Up 200 points
        
        # Get portfolio value before exit
        initial_portfolio = risk_manager.net_worth
        
        # Exit position due to take profit
        risk_manager.exit_position(exit_price, self.exit_date, "take_profit")
        
        # Calculate expected profit
        # 200 points * $20/point * 1 contract = $4,000 profit
        expected_profit = money.to_decimal(200) * money.to_decimal(20)
        expected_portfolio = initial_portfolio + expected_profit
        
        # Check portfolio value
        self.assertEqual(risk_manager.net_worth, expected_portfolio)
        
        # Check trade record
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["exit_reason"], "take_profit")
        self.assertEqual(last_trade["profit"], float(expected_profit))
        
    def test_trailing_stop_calculation(self):
        """
        Test that profits are correctly calculated when trailing stop is triggered.
        """
        # Create a risk manager with trailing stop
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.5,  # 0.5% trailing stop
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position
        entry_price = 15000.0
        risk_manager.enter_position(1, entry_price, self.entry_date, 1)
        
        # Simulate price going up and then falling
        # First, update to establish a peak
        high_price = 15300.0  # Up 300 points
        risk_manager.update_stops(high_price, high_price, high_price)
        
        # Now simulate a fall that triggers trailing stop
        exit_price = 15150.0  # Down 150 points from peak, but still up from entry
        
        # Get portfolio value before exit
        initial_portfolio = risk_manager.net_worth
        
        # Exit position due to trailing stop
        risk_manager.exit_position(exit_price, self.exit_date, "trailing_stop")
        
        # Calculate expected profit
        # 150 points * $20/point * 1 contract = $3,000 profit
        expected_profit = money.to_decimal(150) * money.to_decimal(20)
        expected_portfolio = initial_portfolio + expected_profit
        
        # Check portfolio value
        self.assertEqual(risk_manager.net_worth, expected_portfolio)
        
        # Check trade record
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["exit_reason"], "trailing_stop")
        self.assertEqual(last_trade["profit"], float(expected_profit))
        
    def test_transaction_cost_with_risk_features(self):
        """
        Test that transaction costs are correctly applied with risk management features.
        """
        # Create a risk manager with transaction costs and stop loss
        transaction_cost = Decimal('0.1')  # 0.1% transaction cost
        risk_manager = RiskManager(
            stop_loss_pct=1.0,
            take_profit_pct=None,
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=transaction_cost
        )
        
        # Enter a long position
        entry_price = 15000.0
        risk_manager.enter_position(1, entry_price, self.entry_date, 1)
        
        # Simulate a price drop that triggers stop loss
        exit_price = 14800.0  # Down 200 points
        
        # Get portfolio value before exit
        initial_portfolio = risk_manager.net_worth
        
        # Exit position due to stop loss
        risk_manager.exit_position(exit_price, self.exit_date, "stop_loss")
        
        # Calculate expected loss with transaction costs
        # 200 points * $20/point * 1 contract = $4,000 loss
        price_loss = money.to_decimal(200) * money.to_decimal(20)
        
        # Transaction cost = 14800 * 1 * 20 * 0.1% = $29.6
        trans_cost = money.to_decimal(exit_price) * money.to_decimal(20) * (transaction_cost / Decimal('100'))
        
        expected_total_loss = price_loss + trans_cost
        expected_portfolio = initial_portfolio - expected_total_loss
        
        # Check portfolio value (allowing for small rounding differences)
        self.assertAlmostEqual(float(risk_manager.net_worth), float(expected_portfolio), places=2)
        
        # Check trade record
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["exit_reason"], "stop_loss")
        self.assertAlmostEqual(last_trade["profit"], float(-expected_total_loss), places=2)
        
    def test_multiple_contracts_with_risk_features(self):
        """
        Test that multiple contracts are correctly handled with risk management features.
        """
        # Create a risk manager with take profit
        risk_manager = RiskManager(
            stop_loss_pct=None,
            take_profit_pct=2.0,  # 2% take profit
            trailing_stop_pct=None,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Enter a long position with 2 contracts
        entry_price = 15000.0
        contracts = 2
        risk_manager.enter_position(1, entry_price, self.entry_date, contracts)
        
        # Verify contracts stored correctly
        self.assertEqual(risk_manager.current_contracts, contracts)
        
        # Simulate a price increase that triggers take profit
        exit_price = 15100.0  # Up 100 points
        
        # Get portfolio value before exit
        initial_portfolio = risk_manager.net_worth
        
        # Exit position due to take profit
        risk_manager.exit_position(exit_price, self.exit_date, "take_profit")
        
        # Calculate expected profit with multiple contracts
        # 100 points * $20/point * 2 contracts = $4,000 profit
        expected_profit = money.to_decimal(100) * money.to_decimal(20) * money.to_decimal(contracts)
        expected_portfolio = initial_portfolio + expected_profit
        
        # Check portfolio value
        self.assertEqual(risk_manager.net_worth, expected_portfolio)
        
        # Check trade record
        last_trade = risk_manager.trade_history[-1]
        self.assertEqual(last_trade["exit_reason"], "take_profit")
        self.assertEqual(last_trade["profit"], float(expected_profit))
        self.assertEqual(last_trade["contracts"], contracts)

if __name__ == "__main__":
    unittest.main() 