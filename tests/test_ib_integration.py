import unittest
import sys
import os
import time
import yaml
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib_insync import IB, Future, MarketOrder, util, Order, LimitOrder
from ib import ModelTrader

class TestIBIntegration(unittest.TestCase):
    """
    Integration tests that connect to IB paper account and submit real orders.
    
    These tests require:
    1. IB Gateway or TWS running with paper account connected
    2. Market data subscription for the contracts being tested
    3. Paper account with sufficient margin
    
    WARNING: These tests will place actual orders in your paper account!
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up resources needed for all tests."""
        # Load config for paper account settings
        cls.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(cls.config_path):
            with open(cls.config_path, "r") as f:
                cls.config = yaml.safe_load(f)
        else:
            # Use default settings if config file not found
            cls.config = {
                "paper_account": {
                    "host": "127.0.0.1",
                    "port": 7497,  # Default TWS/Gateway paper trading port
                    "client_id": 1
                }
            }
        
        # Connect to IB
        cls.ib = IB()
        try:
            paper_account = cls.config.get("paper_account", {})
            host = paper_account.get("host", "127.0.0.1")
            port = paper_account.get("port", 7497)
            client_id = paper_account.get("client_id", 1)
            
            cls.ib.connect(host, port, clientId=client_id)
            print(f"Connected to IB at {host}:{port} with client ID {client_id}")
            
            # Wait for connection to establish
            time.sleep(1)
            
            # Get NQ contract info (E-mini Nasdaq futures)
            cls.nq_contract_generic = Future('NQ', exchange='CME')
            cls.contracts = cls.ib.reqContractDetails(cls.nq_contract_generic)
            if not cls.contracts:
                raise RuntimeError("No contract details found for NQ")
            
            # Select the most recent (front-month) contract
            cls.active_contract = cls.get_most_recent_contract(cls.contracts)
            print(f"Using contract: {cls.active_contract}")
            
            # Wait for any pending orders to process
            time.sleep(1)
            
        except Exception as e:
            print(f"Error connecting to IB: {e}")
            print("Make sure TWS or IB Gateway is running with paper account connected")
            cls.tearDownClass()
            raise
            
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        if hasattr(cls, 'ib') and cls.ib.isConnected():
            # Cancel all open orders
            open_trades = cls.ib.openTrades()
            for trade in open_trades:
                cls.ib.cancelOrder(trade.order)
                print(f"Cancelled order: {trade.order}")
            
            # Close any open positions
            portfolio = cls.ib.portfolio()
            for position in portfolio:
                if position.position != 0 and position.contract.symbol == 'NQ':
                    action = "SELL" if position.position > 0 else "BUY"
                    quantity = abs(position.position)
                    order = MarketOrder(action, quantity)
                    cls.ib.placeOrder(position.contract, order)
                    print(f"Closed position: {action} {quantity} {position.contract.symbol}")
            
            # Disconnect from IB
            cls.ib.disconnect()
            print("Disconnected from IB")
    
    @classmethod
    def get_most_recent_contract(cls, contracts):
        """
        From a list of contract details for NQ futures, select the contract with the earliest expiration date
        that is still in the future (the active or front-month contract).
        """
        from datetime import datetime

        # Helper function to parse the expiration string.
        def parse_date(date_str):
            try:
                # Handle format 'YYYYMMDD' or sometimes 'YYYYMM'
                if len(date_str) == 8:
                    return datetime.strptime(date_str, '%Y%m%d')
                elif len(date_str) == 6:
                    return datetime.strptime(date_str, '%Y%m')
            except Exception:
                return datetime.max

        today = datetime.today()
        # Filter contracts that have an expiration date in the future.
        valid_contracts = [cd for cd in contracts if parse_date(cd.contract.lastTradeDateOrContractMonth) > today]
        if valid_contracts:
            # Sort filtered contracts by expiration date (earliest first)
            sorted_contracts = sorted(valid_contracts, key=lambda cd: parse_date(cd.contract.lastTradeDateOrContractMonth))
            return sorted_contracts[0].contract
        else:
            # Fallback to the first contract if for some reason none are in the future.
            return contracts[0].contract
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Cancel any previous orders before each test
        open_trades = self.ib.openTrades()
        for trade in open_trades:
            self.ib.cancelOrder(trade.order)
            print(f"Cancelled order: {trade.order}")
        
        # Allow time for orders to be cancelled
        time.sleep(1)
    
    def get_current_price(self, contract, max_attempts=5):
        """
        Request market data with retries and extended wait times to reliably get prices.
        
        Args:
            contract: The contract to get market data for
            max_attempts: Maximum number of retries
            
        Returns:
            float: The current price or None if not available after retries
        """
        for attempt in range(max_attempts):
            # Request market data
            market_data = self.ib.reqMktData(contract)
            
            # Wait longer for market data with increasing backoff
            wait_time = 3 + (attempt * 2)  # 3, 5, 7, 9, 11 seconds
            print(f"Waiting {wait_time}s for market data (attempt {attempt+1}/{max_attempts})...")
            self.ib.sleep(wait_time)
            
            # Try different price sources
            price = None
            if hasattr(market_data, 'last') and market_data.last and market_data.last > 0:
                price = market_data.last
                print(f"Using last price: {price}")
            elif hasattr(market_data, 'close') and market_data.close and market_data.close > 0:
                price = market_data.close
                print(f"Using close price: {price}")
            elif hasattr(market_data, 'bid') and hasattr(market_data, 'ask') and market_data.bid and market_data.ask:
                # Use mid price if bid/ask available
                price = (market_data.bid + market_data.ask) / 2
                print(f"Using mid price: {price} (bid: {market_data.bid}, ask: {market_data.ask})")
            
            if price and price > 0:
                return price
                
            # Cancel the market data subscription before retrying
            self.ib.cancelMktData(contract)
        
        return None
    
    def test_long_position_with_take_profit(self):
        """Test submitting a long order with take profit to paper account."""
        # Skip test if not connected
        if not self.ib.isConnected():
            self.skipTest("Not connected to IB")
        
        # Get current price with improved retry mechanism
        current_price = self.get_current_price(self.active_contract)
        
        # Ensure we have a valid price
        self.assertIsNotNone(current_price, "Could not get valid market price after multiple attempts")
        self.assertGreater(current_price, 0, "Could not get valid market price")
        print(f"Current price: {current_price}")
        
        # Initialize the model trader with risk management enabled
        model_trader = ModelTrader(
            ib_instance=self.ib,
            model_path="dummy_model",  # Not actually using the model for this test
            use_risk_management=True
        )
        
        # Set active contract and risk parameters
        model_trader.set_active_contract(self.active_contract)
        model_trader.stop_loss_pct = 1.0  # 1% stop loss
        model_trader.take_profit_pct = 0.5  # Small 0.5% take profit for testing
        model_trader.position_size = 1  # Trade 1 contract
        
        try:
            # Enter a long position using the model trader
            model_trader._enter_position(direction=1, price=current_price)
            
            # Wait for orders to be transmitted
            self.ib.sleep(2)
            
            # Get open orders
            open_orders = self.ib.openOrders()
            
            # Verify we have orders
            self.assertGreaterEqual(len(open_orders), 2, "Expected at least 2 orders (parent and take profit)")
            
            # Find the take profit order
            take_profit_order = None
            for order in open_orders:
                if order.orderType == 'LMT' and order.action == 'SELL':
                    take_profit_order = order
                    break
            
            # Verify the take profit order
            self.assertIsNotNone(take_profit_order, "Take profit order not found")
            
            # Calculate expected take profit price
            point_value = 20.0  # $20 per point for NQ futures
            tp_points = (current_price * (model_trader.take_profit_pct / 100)) / point_value
            expected_tp_price = current_price + tp_points
            
            # Check that the limit price is close to what we expect (within a few ticks)
            self.assertAlmostEqual(
                take_profit_order.lmtPrice, 
                round(expected_tp_price, 2),
                delta=1.0,  # Allow for some difference due to price movement
                msg="Take profit price not set correctly"
            )
            
            print(f"Take profit order placed at {take_profit_order.lmtPrice} (expected ~{expected_tp_price:.2f})")
            
        finally:
            # Cancel all orders
            open_trades = self.ib.openTrades()
            for trade in open_trades:
                self.ib.cancelOrder(trade.order)
                print(f"Cancelled order: {trade.order}")
            
            # Close any position we might have opened
            if model_trader.current_position != 0:
                model_trader._exit_position()
                print("Closed test position")
    
    def test_take_profit_only_order(self):
        """Test submitting an order with take profit but without stop loss."""
        # Skip test if not connected
        if not self.ib.isConnected():
            self.skipTest("Not connected to IB")
        
        # Get current price with improved retry mechanism
        current_price = self.get_current_price(self.active_contract)
        
        # Ensure we have a valid price
        self.assertIsNotNone(current_price, "Could not get valid market price after multiple attempts")
        self.assertGreater(current_price, 0, "Could not get valid market price")
        print(f"Current price: {current_price}")
        
        # Initialize the model trader with only take profit
        model_trader = ModelTrader(
            ib_instance=self.ib,
            model_path="dummy_model",  # Not actually using the model for this test
            use_risk_management=True
        )
        
        # Set active contract and risk parameters
        model_trader.set_active_contract(self.active_contract)
        model_trader.stop_loss_pct = None  # No stop loss
        model_trader.take_profit_pct = 0.5  # Small 0.5% take profit for testing
        model_trader.position_size = 1  # Trade 1 contract
        
        try:
            # Enter a long position using the model trader
            model_trader._enter_position(direction=1, price=current_price)
            
            # Wait for orders to be transmitted
            self.ib.sleep(2)
            
            # Get open orders
            open_orders = self.ib.openOrders()
            
            # Verify we have orders
            self.assertGreaterEqual(len(open_orders), 2, "Expected at least 2 orders (parent and take profit)")
            
            # Find the take profit order
            take_profit_order = None
            for order in open_orders:
                if order.orderType == 'LMT' and order.action == 'SELL':
                    take_profit_order = order
                    break
            
            # Verify the take profit order
            self.assertIsNotNone(take_profit_order, "Take profit order not found")
            
            # Verify no stop loss order
            stop_loss_orders = [o for o in open_orders if o.orderType == 'STP']
            self.assertEqual(len(stop_loss_orders), 0, "Found unexpected stop loss orders")
            
            print(f"Take profit only order placed successfully at {take_profit_order.lmtPrice}")
            
        finally:
            # Cancel all orders
            open_trades = self.ib.openTrades()
            for trade in open_trades:
                self.ib.cancelOrder(trade.order)
                print(f"Cancelled order: {trade.order}")
            
            # Close any position we might have opened
            if model_trader.current_position != 0:
                model_trader._exit_position()
                print("Closed test position")

if __name__ == '__main__':
    unittest.main() 