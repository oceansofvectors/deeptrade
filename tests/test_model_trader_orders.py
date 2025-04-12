import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib import ModelTrader, IB, Order

class TestModelTraderOrders(unittest.TestCase):
    """
    Test suite for the ModelTrader class in ib.py, focusing on order submission functionality.
    
    These tests focus on testing the proper creation and submission of orders,
    particularly with take profit functionality.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock IB instance
        self.mock_ib = MagicMock(spec=IB)
        
        # Create a mock contract
        self.mock_contract = MagicMock()
        self.mock_contract.symbol = "NQ"
        self.mock_contract.exchange = "CME"
        
        # Create a ModelTrader instance with mocked dependencies
        self.model_trader = ModelTrader(
            ib_instance=self.mock_ib,
            model_path="dummy_model",
            use_risk_management=True
        )
        
        # Set active contract
        self.model_trader.set_active_contract(self.mock_contract)
        
        # Mock the model
        self.model_trader.model = MagicMock()
        self.model_trader.model.predict.return_value = (0, None)  # 0 means "long" in our system
        
        # Set risk parameters
        self.model_trader.stop_loss_pct = 1.0  # 1% stop loss
        self.model_trader.take_profit_pct = 2.0  # 2% take profit
        self.model_trader.position_size = 1  # Default position size

    def test_enter_long_position_with_take_profit(self):
        """Test entering a long position with take profit."""
        # Set up the mock IB to track placed orders
        placed_orders = []
        self.mock_ib.placeOrder.side_effect = lambda contract, order: placed_orders.append(order)
        
        # Current price for the test
        current_price = 15000.0
        
        # Call the method to enter a long position
        self.model_trader._enter_position(direction=1, price=current_price)
        
        # Verify the order creation
        self.assertEqual(len(placed_orders), 3)  # Parent order, take profit order, stop loss order
        
        # Check that the parent order is a BUY order
        parent_order = placed_orders[0]
        self.assertEqual(parent_order.action, "BUY")
        self.assertEqual(parent_order.orderType, "MKT")
        self.assertEqual(parent_order.totalQuantity, 1)
        self.assertFalse(parent_order.transmit)
        
        # Check that the take profit order is a SELL limit order
        take_profit_order = placed_orders[1]
        self.assertEqual(take_profit_order.action, "SELL")
        self.assertEqual(take_profit_order.orderType, "LMT")
        
        # Calculate expected take profit price (2% above entry)
        expected_tp_price = current_price + (current_price * 0.02) / 20.0  # $20 per point for NQ
        self.assertAlmostEqual(take_profit_order.lmtPrice, round(expected_tp_price, 2))
        
        # Ensure the take profit order references the parent order
        self.assertEqual(take_profit_order.parentId, parent_order.orderId)
        
        # Check that position is updated
        self.assertEqual(self.model_trader.current_position, 1)

    def test_enter_short_position_with_take_profit(self):
        """Test entering a short position with take profit."""
        # Set up the mock IB to track placed orders
        placed_orders = []
        self.mock_ib.placeOrder.side_effect = lambda contract, order: placed_orders.append(order)
        
        # Current price for the test
        current_price = 15000.0
        
        # Call the method to enter a short position
        self.model_trader._enter_position(direction=-1, price=current_price)
        
        # Verify the order creation
        self.assertEqual(len(placed_orders), 3)  # Parent order, take profit order, stop loss order
        
        # Check that the parent order is a SELL order
        parent_order = placed_orders[0]
        self.assertEqual(parent_order.action, "SELL")
        self.assertEqual(parent_order.orderType, "MKT")
        self.assertEqual(parent_order.totalQuantity, 1)
        self.assertFalse(parent_order.transmit)
        
        # Check that the take profit order is a BUY limit order
        take_profit_order = placed_orders[1]
        self.assertEqual(take_profit_order.action, "BUY")
        self.assertEqual(take_profit_order.orderType, "LMT")
        
        # Calculate expected take profit price (2% below entry for short)
        expected_tp_price = current_price - (current_price * 0.02) / 20.0  # $20 per point for NQ
        self.assertAlmostEqual(take_profit_order.lmtPrice, round(expected_tp_price, 2))
        
        # Ensure the take profit order references the parent order
        self.assertEqual(take_profit_order.parentId, parent_order.orderId)
        
        # Check that position is updated
        self.assertEqual(self.model_trader.current_position, -1)

    def test_bracket_order_creation(self):
        """Test the specific bracket order creation method directly."""
        # Set up the mock IB to track placed orders
        placed_orders = []
        self.mock_ib.placeOrder.side_effect = lambda contract, order: placed_orders.append(order)
        
        # Test creating a bracket order for a long position
        price = 15000.0
        direction = 1
        quantity = 2
        
        # Call the bracket order creation method directly
        self.model_trader._create_bracket_order(direction, price, quantity)
        
        # Verify all three orders were created
        self.assertEqual(len(placed_orders), 3)
        
        # Check parent order details
        parent_order = placed_orders[0]
        self.assertEqual(parent_order.action, "BUY")
        self.assertEqual(parent_order.orderType, "MKT")
        self.assertEqual(parent_order.totalQuantity, quantity)
        
        # Check take profit order details
        take_profit_order = placed_orders[1]
        self.assertEqual(take_profit_order.action, "SELL")
        self.assertEqual(take_profit_order.orderType, "LMT")
        self.assertEqual(take_profit_order.totalQuantity, quantity)
        
        # Check calculated take profit price
        tp_points = (price * (self.model_trader.take_profit_pct / 100)) / 20.0
        expected_tp_price = price + tp_points
        self.assertAlmostEqual(take_profit_order.lmtPrice, round(expected_tp_price, 2))
        
        # Check stop loss order details
        stop_loss_order = placed_orders[2]
        self.assertEqual(stop_loss_order.action, "SELL")
        self.assertEqual(stop_loss_order.orderType, "STP")
        self.assertEqual(stop_loss_order.totalQuantity, quantity)
        
        # Check calculated stop loss price
        sl_points = (price * (self.model_trader.stop_loss_pct / 100)) / 20.0
        expected_sl_price = price - sl_points
        self.assertAlmostEqual(stop_loss_order.auxPrice, round(expected_sl_price, 2))

    def test_take_profit_only(self):
        """Test creating an order with take profit but no stop loss."""
        # Set up trader with only take profit
        self.model_trader.stop_loss_pct = None
        self.model_trader.take_profit_pct = 2.0
        
        # Set up the mock IB to track placed orders
        placed_orders = []
        self.mock_ib.placeOrder.side_effect = lambda contract, order: placed_orders.append(order)
        
        # Current price for the test
        current_price = 15000.0
        
        # Call the method to enter a position
        self.model_trader._enter_position(direction=1, price=current_price)
        
        # Verify the order creation
        self.assertEqual(len(placed_orders), 2)  # Parent order and take profit order only
        
        # Check that the parent order is a BUY order
        parent_order = placed_orders[0]
        self.assertEqual(parent_order.action, "BUY")
        
        # Check that the take profit order is a SELL limit order
        take_profit_order = placed_orders[1]
        self.assertEqual(take_profit_order.action, "SELL")
        self.assertEqual(take_profit_order.orderType, "LMT")
        
        # Check that the take profit order is set to transmit since it's the last one
        self.assertTrue(take_profit_order.transmit)

    def test_execute_trade_with_model_prediction(self):
        """Test the full trade execution flow based on model prediction."""
        # Set up the mock IB to track placed orders
        placed_orders = []
        self.mock_ib.placeOrder.side_effect = lambda contract, order: placed_orders.append(order)
        
        # Mock a bar
        bar = {
            'open': 15000.0,
            'high': 15010.0,
            'low': 14990.0,
            'close': 15005.0,
            'volume': 100,
            'start': '2023-01-01 09:30:00',
            'end': '2023-01-01 09:35:00'
        }
        
        # Create observation
        observation = np.array([0.5, 0.0], dtype=np.float32)  # normalized price and no position
        
        # Mock preprocess_bar to return our observation
        self.model_trader.preprocess_bar = MagicMock(return_value=observation)
        
        # Mock model prediction to go long
        self.model_trader.get_prediction = MagicMock(return_value=0)  # 0 = long
        
        # Execute the trade
        self.model_trader.execute_trade(0, bar)
        
        # Verify that orders were placed
        self.assertGreater(len(placed_orders), 0)
        
        # Verify that we have a long position
        self.assertEqual(self.model_trader.current_position, 1)

if __name__ == '__main__':
    unittest.main() 