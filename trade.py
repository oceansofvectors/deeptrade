import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
from decimal import Decimal
import pytz

from stable_baselines3 import PPO
from environment import TradingEnv
from get_data import get_data
from config import config
import money  # Import the new money module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management wrapper for trading strategies.
    Implements stop loss, take profit, trailing stop loss, and position sizing.
    """
    
    def __init__(
        self,
        stop_loss_pct: Optional[float] = None,  # Stop loss as percentage of portfolio value
        take_profit_pct: Optional[float] = None,  # Take profit as percentage of portfolio value
        trailing_stop_pct: Optional[float] = None,  # Trailing stop as percentage of highest/lowest price
        position_size: float = 1.0,  # Position size as a multiplier (1.0 = 100% of available capital)
        max_risk_per_trade_pct: float = 2.0,  # Maximum risk per trade as percentage of portfolio
        initial_balance: float = 10000.0,  # Initial portfolio balance
        transaction_cost: float = 0.0,  # Transaction cost as percentage
        daily_risk_limit: Optional[float] = None  # Maximum dollar loss allowed per trading day
    ):
        """
        Initialize the risk manager.
        
        Args:
            stop_loss_pct: Stop loss percentage of portfolio value (None to disable)
            take_profit_pct: Take profit percentage of portfolio value (None to disable)
            trailing_stop_pct: Trailing stop percentage (None to disable)
            position_size: Position size multiplier (1.0 = 100% of available capital)
            max_risk_per_trade_pct: Maximum risk per trade as percentage of portfolio
            initial_balance: Initial portfolio balance
            transaction_cost: Transaction cost as percentage
            daily_risk_limit: Maximum dollar loss allowed per trading day (None to disable)
        """
        # Convert all monetary values to Decimal for precision
        self.stop_loss_pct = money.to_decimal(stop_loss_pct) if stop_loss_pct is not None else None
        self.take_profit_pct = money.to_decimal(take_profit_pct) if take_profit_pct is not None else None
        self.trailing_stop_pct = money.to_decimal(trailing_stop_pct) if trailing_stop_pct is not None else None
        self.position_size = money.to_decimal(position_size)
        self.max_risk_per_trade_pct = money.to_decimal(max_risk_per_trade_pct)
        self.initial_balance = money.to_decimal(initial_balance)
        self.transaction_cost = money.to_decimal(transaction_cost)
        self.daily_risk_limit = money.to_decimal(daily_risk_limit) if daily_risk_limit is not None else None
        
        # Internal state
        self.position = 0  # Current position: 0 (neutral), 1 (long), -1 (short)
        self.entry_price = Decimal('0.0')  # Price at which position was entered
        self.entry_portfolio_value = Decimal('0.0')  # Portfolio value at entry
        self.highest_price = Decimal('0.0')  # Highest price since entry (for trailing stop on long)
        self.lowest_price = Decimal('Infinity')  # Lowest price since entry (for trailing stop on short)
        self.highest_unrealized_profit_pct = Decimal('0.0')  # Highest unrealized profit % (for trailing stop)
        self.stop_loss_price = Decimal('0.0')  # Current stop loss price
        self.take_profit_price = Decimal('0.0')  # Current take profit price
        self.trailing_stop_price = Decimal('0.0')  # Current trailing stop price
        self.net_worth = self.initial_balance  # Current portfolio value
        self.trade_history = []  # History of trades
        self.current_contracts = 0  # Number of contracts in current position
        
        # Daily risk tracking
        self.daily_start_balance = self.initial_balance  # Portfolio value at start of day
        self.daily_lowest_balance = self.initial_balance  # Lowest portfolio value during the day
        self.daily_trading_enabled = True  # Flag to track if trading is allowed for the day
        self.current_trading_day = None  # Current trading day being tracked
        
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate the number of contracts to trade based on portfolio value.
        
        Args:
            price: Current price of the asset
            
        Returns:
            int: Number of contracts to trade
        """
        # Convert price to Decimal for precise calculations
        price = money.to_decimal(price)
        point_value = money.to_decimal(20.0)  # NQ futures point value
        
        # Calculate max contracts based on position sizing parameter
        # Use the full position_size percentage of the portfolio
        max_position_value = self.net_worth * self.position_size
        
        # For NQ futures, we need enough to cover the notional exposure
        # Each contract is worth price * point_value
        contract_value = price * point_value
        max_contracts = int(max_position_value / contract_value)
        
        # Always trade at least 1 contract if we have any funds
        if max_contracts < 1 and self.net_worth > money.to_decimal(0):
            max_contracts = 1
        
        # Calculate actual exposure (notional value)
        notional_value = price * point_value * money.to_decimal(max_contracts)
        
        # Log position sizing information
        logger.info(f"Position sizing: price=${float(price):.2f}, portfolio=${float(self.net_worth):.2f}, "
                   f"position size={float(self.position_size)}, contracts={max_contracts}, "
                   f"exposure=${float(notional_value):.2f}")
        
        return max_contracts
    
    def update_stops(self, close_price: float, high_price: float = None, low_price: float = None) -> None:
        """
        Update stop loss, take profit, and trailing stop prices based on current candle prices.
        
        Args:
            close_price: Current closing price
            high_price: Current candle high price (optional)
            low_price: Current candle low price (optional)
        """
        if self.position == 0:
            return
            
        # Convert prices to Decimal
        close_price = money.to_decimal(close_price)
        high_price = money.to_decimal(high_price) if high_price is not None else close_price
        low_price = money.to_decimal(low_price) if low_price is not None else close_price
        
        # Calculate current unrealized P&L in points
        if self.position == 1:  # Long position
            unrealized_pnl_points = close_price - self.entry_price
        else:  # Short position
            unrealized_pnl_points = self.entry_price - close_price
            
        # Convert to dollar value (each NQ point worth $20)
        point_value = money.to_decimal(20.0)
        unrealized_pnl_dollars = unrealized_pnl_points * point_value * self.current_contracts
        
        # Calculate as percentage of entry portfolio value
        unrealized_pnl_pct = (unrealized_pnl_dollars / self.entry_portfolio_value) * 100 if self.entry_portfolio_value and self.entry_portfolio_value != 0 else Decimal('0.0')
        
        # Update highest unrealized profit % for trailing stop
        if unrealized_pnl_pct > self.highest_unrealized_profit_pct:
            self.highest_unrealized_profit_pct = unrealized_pnl_pct
            
            # Update trailing stop if enabled
            if self.trailing_stop_pct is not None:
                # Trail at specified percentage below highest profit
                self.trailing_stop_price = self.highest_unrealized_profit_pct - self.trailing_stop_pct
        
        # Also update highest/lowest prices for price-based trailing stop if needed
        if self.position == 1:  # Long position
            # Use high price to update highest price seen (best case)
            if high_price > self.highest_price:
                self.highest_price = high_price
        elif self.position == -1:  # Short position
            # Use low price to update lowest price seen (best case)
            if low_price < self.lowest_price:
                self.lowest_price = low_price
    
    def check_exits(self, close_price: float, high_price: float = None, low_price: float = None) -> Tuple[bool, str]:
        """
        Check if any exit conditions are met based on unrealized P&L.
        
        Args:
            close_price: Current closing price
            high_price: Current candle high price (optional)
            low_price: Current candle low price (optional)
            
        Returns:
            Tuple[bool, str]: (exit_triggered, reason)
        """
        if self.position == 0:
            return False, ""
        
        # Convert prices to Decimal
        close_price = money.to_decimal(close_price)
        high_price = money.to_decimal(high_price) if high_price is not None else close_price
        low_price = money.to_decimal(low_price) if low_price is not None else close_price
        
        # Point value for NQ futures ($20 per point)
        point_value = money.to_decimal(20.0)
        
        # Calculate unrealized P&L for different scenarios
        if self.position == 1:  # Long position
            # For stop loss check, use low price (worst case)
            pnl_points_stop = low_price - self.entry_price
            # For take profit check, use high price (best case)
            pnl_points_profit = high_price - self.entry_price
        else:  # Short position
            # For stop loss check, use high price (worst case)
            pnl_points_stop = self.entry_price - high_price
            # For take profit check, use low price (best case)
            pnl_points_profit = self.entry_price - low_price
            
        # Convert to dollar value
        pnl_dollars_stop = pnl_points_stop * point_value * self.current_contracts
        pnl_dollars_profit = pnl_points_profit * point_value * self.current_contracts
        
        # Calculate as percentage of entry portfolio value
        pnl_pct_stop = (pnl_dollars_stop / self.entry_portfolio_value) * 100 if self.entry_portfolio_value and self.entry_portfolio_value != 0 else Decimal('0.0')
        pnl_pct_profit = (pnl_dollars_profit / self.entry_portfolio_value) * 100 if self.entry_portfolio_value and self.entry_portfolio_value != 0 else Decimal('0.0')
        
        # Add debug logging
        logging.debug(f"Position: {self.position}, Entry price: {self.entry_price}, Close: {close_price}")
        logging.debug(f"Stop loss check (using {'low' if self.position == 1 else 'high'} price): P&L = {pnl_pct_stop:.2f}%, Threshold: -{self.stop_loss_pct}%")
        logging.debug(f"Take profit check (using {'high' if self.position == 1 else 'low'} price): P&L = {pnl_pct_profit:.2f}%, Threshold: +{self.take_profit_pct}%")
        
        # Check if any exit conditions are met
        exit_triggered = False
        reason = ""
        
        # Check stop loss - exit if unrealized P&L percentage is below stop loss threshold
        if self.stop_loss_pct is not None and pnl_pct_stop <= -self.stop_loss_pct:
            exit_triggered = True
            reason = "stop_loss"
            logging.info(f"Stop loss triggered! Unrealized P&L: {pnl_pct_stop:.2f}%, Threshold: -{self.stop_loss_pct}%")
            
        # Check take profit - exit if unrealized P&L percentage is above take profit threshold
        elif self.take_profit_pct is not None and pnl_pct_profit >= self.take_profit_pct:
            exit_triggered = True
            reason = "take_profit"
            logging.info(f"Take profit triggered! Unrealized P&L: {pnl_pct_profit:.2f}%, Threshold: +{self.take_profit_pct}%")
            
        # Check trailing stop - exit if profit has fallen below trailing stop
        elif self.trailing_stop_pct is not None and self.highest_unrealized_profit_pct > 0:
            # Calculate current unrealized P&L percentage using close price
            if self.position == 1:  # Long position
                current_pnl_points = close_price - self.entry_price
            else:  # Short position
                current_pnl_points = self.entry_price - close_price
                
            current_pnl_dollars = current_pnl_points * point_value * self.current_contracts
            current_pnl_pct = (current_pnl_dollars / self.entry_portfolio_value) * 100
            
            # Add debug logging for trailing stop
            if self.highest_unrealized_profit_pct > self.trailing_stop_pct:
                logging.debug(f"Trailing stop check: Current P&L: {current_pnl_pct:.2f}%, Peak: {self.highest_unrealized_profit_pct:.2f}%, Threshold: {self.highest_unrealized_profit_pct - self.trailing_stop_pct:.2f}%")
            
            # Check if profit has fallen below trailing stop threshold
            if current_pnl_pct < (self.highest_unrealized_profit_pct - self.trailing_stop_pct):
                exit_triggered = True
                reason = "trailing_stop"
                logging.info(f"Trailing stop triggered! Current P&L: {current_pnl_pct:.2f}%, Peak: {self.highest_unrealized_profit_pct:.2f}%, Threshold: {self.highest_unrealized_profit_pct - self.trailing_stop_pct:.2f}%")
                
        return exit_triggered, reason
    
    def enter_position(self, position: int, price: float, date: pd.Timestamp, contracts: int) -> None:
        """
        Enter a new position.
        
        Args:
            position: Position to enter (1 for long, -1 for short)
            price: Entry price
            date: Entry date
            contracts: Number of contracts
        """
        self.position = position
        self.entry_price = money.to_decimal(price)
        self.current_contracts = contracts  # Store the number of contracts
        
        # Ensure we never have a zero portfolio value at entry to prevent division by zero
        if self.net_worth == 0:
            # If net_worth is zero, use the initial_balance
            self.net_worth = self.initial_balance
            logging.warning(f"Portfolio value was zero at position entry. Using initial balance: {self.initial_balance}")
            
        self.entry_portfolio_value = money.to_decimal(self.net_worth)  # Store portfolio value at entry
        self.highest_unrealized_profit_pct = Decimal('0.0')  # Reset highest profit
        
        # Set initial values
        if self.position == 1:  # Long position
            self.highest_price = self.entry_price
        elif self.position == -1:  # Short position
            self.lowest_price = self.entry_price
            
        # Record trade
        self.trade_history.append({
            "date": date,
            "action": "buy" if position == 1 else "sell",
            "price": float(self.entry_price),  # Convert Decimal to float for serialization
            "contracts": contracts,
            "portfolio_value": float(self.net_worth)  # Convert Decimal to float for serialization
        })
        
    def exit_position(self, price: float, date: pd.Timestamp, reason: str) -> None:
        """
        Exit the current position.
        
        Args:
            price: Exit price
            date: Exit date
            reason: Reason for exit
        """
        # Convert price to Decimal
        price_decimal = money.to_decimal(price)
        
        # Calculate price change - this is the source of the issue
        # We need to ensure this is calculated correctly for NQ futures
        price_change = money.calculate_price_change(self.entry_price, price_decimal, self.position)
        
        # Debug the price change calculation
        logging.debug(f"Exit calculation: Entry price: {self.entry_price}, Exit price: {price_decimal}")
        logging.debug(f"Position: {self.position}, Raw price change: {price_change}")
        
        # For NQ futures:
        # - Each full point is worth $20
        # - The minimum tick size is 0.25 points, worth $5
        # - Calculate contracts × points × $20/point
        point_value = money.to_decimal(20.0)
        
        # Calculate dollar change - this is where we apply the point value
        # We use the raw price difference multiplied by the point value
        dollar_change = price_change * point_value * money.to_decimal(self.current_contracts)
        logging.debug(f"Dollar change before fees: ${dollar_change}")
            
        # Apply transaction costs
        transaction_cost = money.calculate_transaction_cost(
            price_decimal, self.current_contracts, self.transaction_cost
        )
        net_profit = money.calculate_net_profit(dollar_change, transaction_cost)
        logging.debug(f"Net profit after fees: ${net_profit}")
        
        # FAILSAFE: Ensure portfolio never goes negative
        if self.net_worth + net_profit < money.to_decimal(0):
            old_net_profit = net_profit
            # Calculate the maximum loss we can take (leave $100 minimum)
            net_profit = money.to_decimal(-1) * (self.net_worth - money.to_decimal(100))
            logger.warning(f"FAILSAFE: Limiting loss to ${float(net_profit):.2f} instead of ${float(old_net_profit):.2f} to prevent negative portfolio")
        
        # Update portfolio value
        self.net_worth += net_profit
        
        # Record trade
        self.trade_history.append({
            "date": date,
            "action": "sell" if self.position == 1 else "buy",
            "price": float(price_decimal),  # Convert Decimal to float for serialization
            "contracts": self.current_contracts,
            "portfolio_value": float(self.net_worth),  # Convert Decimal to float for serialization
            "profit": float(net_profit),  # Convert Decimal to float for serialization
            "exit_reason": reason,
            "entry_price": float(self.entry_price),  # Store entry price in trade record
            "entry_portfolio": float(self.entry_portfolio_value),  # Store entry portfolio value
        })
        
        # Reset position
        self.position = 0
        self.entry_price = Decimal('0.0')
        self.entry_portfolio_value = Decimal('0.0')
        self.highest_price = Decimal('0.0')
        self.lowest_price = Decimal('Infinity')
        self.highest_unrealized_profit_pct = Decimal('0.0')
        self.stop_loss_price = Decimal('0.0')
        self.take_profit_price = Decimal('0.0')
        self.trailing_stop_price = Decimal('0.0')
        self.current_contracts = 0  # Reset the number of contracts

    def check_daily_risk_limit(self, current_date: pd.Timestamp) -> Tuple[bool, str]:
        """
        Check if daily risk limit has been exceeded.
        
        Args:
            current_date: Current timestamp
            
        Returns:
            Tuple[bool, str]: (limit_exceeded, reason)
        """
        if self.daily_risk_limit is None:
            return False, ""
            
        # Convert current date to eastern time for market hours check
        eastern = pytz.timezone('US/Eastern')
        current_date_eastern = current_date.tz_convert(eastern)
        
        # If this is a new trading day, reset daily tracking
        if self.current_trading_day is None or self.current_trading_day.date() != current_date_eastern.date():
            self.current_trading_day = current_date_eastern
            self.daily_start_balance = self.net_worth
            self.daily_lowest_balance = self.net_worth
            self.daily_trading_enabled = True
            logger.info(f"New trading day started: {current_date_eastern.date()}")
        
        # Update daily lowest balance
        if self.net_worth < self.daily_lowest_balance:
            self.daily_lowest_balance = self.net_worth
            
        # Calculate daily loss
        daily_loss = self.daily_start_balance - self.daily_lowest_balance
        
        # Check if daily loss exceeds limit
        if daily_loss >= self.daily_risk_limit:
            logger.warning(f"Daily risk limit exceeded! Daily loss: ${float(daily_loss):.2f}, Limit: ${float(self.daily_risk_limit):.2f}")
            return True, "daily_risk_limit"
            
        return False, ""

def trade_with_risk_management(
    model_path: str,
    test_data: pd.DataFrame,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_pct: Optional[float] = None,
    position_size: float = 1.0,
    max_risk_per_trade_pct: float = 2.0,
    initial_balance: float = 10000.0,
    transaction_cost: float = 0.0,
    verbose: int = 1,
    deterministic: bool = True,  # Added parameter with default True for trading
    close_at_end_of_day: bool = False,  # Add parameter to close positions at end of trading day
    daily_risk_limit: Optional[float] = None  # Add daily risk limit parameter
) -> Dict:
    """
    Trade with risk management using a trained model.
    
    This implementation uses portfolio-based risk management:
    - Stop loss: Exits when unrealized loss reaches stop_loss_pct% of portfolio value
    - Take profit: Exits when unrealized gain reaches take_profit_pct% of portfolio value
    - Trailing stop: Tracks highest unrealized profit and exits if profit falls by trailing_stop_pct%
    - Daily risk limit: Closes all positions and stops trading for the day if daily loss exceeds limit
    
    Args:
        model_path: Path to the trained model file
        test_data: Testing dataset
        stop_loss_pct: Stop loss percentage of portfolio value (None to disable)
        take_profit_pct: Take profit percentage of portfolio value (None to disable)
        trailing_stop_pct: Trailing stop percentage (None to disable)
        position_size: Position size as a multiplier (1.0 = 100% of available capital)
        max_risk_per_trade_pct: Maximum risk per trade as percentage of portfolio
        initial_balance: Initial portfolio balance
        transaction_cost: Transaction cost as percentage
        verbose: Verbosity level for logging (default 1)
        deterministic: Whether to use deterministic action selection (default: True)
        close_at_end_of_day: Whether to close positions at the end of the trading day (default: False)
        daily_risk_limit: Maximum dollar loss allowed per trading day (None to disable)
        
    Returns:
        Dict: A dictionary containing performance metrics and trade history
    """
    # Load the model
    model = PPO.load(model_path)
    
    # Create trading environment
    env = TradingEnv(
        test_data,
        initial_balance=initial_balance,
        transaction_cost=0.0,  # We'll handle transaction costs in the risk manager
        position_size=1  # We'll handle position sizing in the risk manager
    )
    
    # Initialize risk manager
    risk_manager = RiskManager(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        position_size=position_size,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        daily_risk_limit=daily_risk_limit  # Add daily risk limit parameter
    )
    
    # Reset environment
    obs, _ = env.reset()
    
    # Records for plotting
    dates = [test_data.index[env.current_step]]
    price_history = [test_data.loc[test_data.index[env.current_step], "Close"]]
    portfolio_history = [float(risk_manager.net_worth)]  # Convert Decimal to float for plotting
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    # Track action counts
    action_counts = {0: 0, 1: 0, 2: 0}  # 0: long/buy, 1: short/sell, 2: hold
    
    # Track exit reasons
    exit_reasons = {
        "stop_loss": 0,
        "take_profit": 0,
        "trailing_stop": 0,
        "model_signal": 0,
        "end_of_day": 0,
        "end_of_period": 0,
        "daily_risk_limit": 0,  # Add daily risk limit to exit reasons
        "second_to_last_candle": 0  # Add reason for second-to-last candle exit
    }
    
    # Add flags for monitoring unrealistic profits
    unrealistic_profit_threshold = money.to_decimal(500)  # $500 per point is unrealistic
    unrealistic_profit_count = 0
    last_valid_profit = None
    multiplier_threshold = money.to_decimal(1.5)  # Convert threshold multiplier to Decimal
    
    # Dictionary to track last candles for each trading day
    eastern = pytz.timezone('US/Eastern')
    trading_day_candles = {}
    
    # Pre-process data to identify the last and second-to-last candles for each trading day
    for i in range(len(test_data)):
        # Get the date in Eastern Time
        candle_date = test_data.index[i]
        if candle_date.tz is None:
            candle_date = candle_date.tz_localize('UTC')
        candle_date_et = candle_date.tz_convert(eastern)
        
        # Use date as key for identifying trading days
        trading_day = candle_date_et.date()
        
        # Track candle indices for each trading day
        if trading_day not in trading_day_candles:
            trading_day_candles[trading_day] = []
        trading_day_candles[trading_day].append(i)
    
    # Identify last and second-to-last candles for each trading day
    last_candles = {}  # Index of last candle for each trading day
    second_to_last_candles = {}  # Index of second-to-last candle for each trading day
    
    for day, candles in trading_day_candles.items():
        if len(candles) >= 2:
            last_candles[day] = candles[-1]
            second_to_last_candles[day] = candles[-2]
        elif len(candles) == 1:
            last_candles[day] = candles[0]
            # No second-to-last candle for this day
    
    if verbose > 0:
        logger.info(f"Identified {len(trading_day_candles)} trading days in the data")
        logger.info(f"Will prevent trading on last candle and close positions on second-to-last candle")
    
    # Main trading loop
    while True:
        # Get current price and date
        current_index = env.current_step
        current_price = test_data.loc[test_data.index[current_index], "Close"]
        current_high = test_data.loc[test_data.index[current_index], "High"]
        current_low = test_data.loc[test_data.index[current_index], "Low"]
        current_date = test_data.index[current_index]
        
        # Convert to Eastern Time for day detection
        if current_date.tz is None:
            current_date_et = current_date.tz_localize('UTC').tz_convert(eastern)
        else:
            current_date_et = current_date.tz_convert(eastern)
        current_day = current_date_et.date()
        
        # Check if this is the second-to-last candle of the day
        is_second_to_last_candle = second_to_last_candles.get(current_day) == current_index
        
        # Check if this is the last candle of the day
        is_last_candle = last_candles.get(current_day) == current_index
        
        # For debugging
        if verbose > 1 and (is_second_to_last_candle or is_last_candle):
            if is_second_to_last_candle:
                logger.info(f"Second-to-last candle detected at {current_date_et}")
            if is_last_candle:
                logger.info(f"Last candle detected at {current_date_et}")
        
        # Check daily risk limit first
        daily_limit_exceeded, daily_limit_reason = risk_manager.check_daily_risk_limit(current_date)
        if daily_limit_exceeded:
            # Close any open position
            if risk_manager.position != 0:
                # For daily risk limit, use close price
                exit_price = current_price
                
                # Record portfolio value before exit for profit checking
                portfolio_before_exit = risk_manager.net_worth
                
                # Exit position
                risk_manager.exit_position(exit_price, current_date, daily_limit_reason)
                
                # Calculate profit from this trade
                trade_profit = risk_manager.net_worth - portfolio_before_exit
                
                # Check for unrealistic profits
                price_diff = abs(money.to_decimal(exit_price) - risk_manager.entry_price)
                expected_max_profit = price_diff * money.to_decimal(20)  # $20 per point for NQ
                
                if trade_profit > unrealistic_profit_threshold and trade_profit > expected_max_profit * multiplier_threshold:
                    unrealistic_profit_count += 1
                    logger.warning(f"Unrealistic profit detected: ${float(trade_profit):.2f} "
                                  f"from price change of {price_diff:.2f} points. "
                                  f"Expected max: ${expected_max_profit:.2f}")
                
                # Record exit
                if risk_manager.position == 1:  # Was long, now exiting
                    sell_dates.append(current_date)
                    sell_prices.append(exit_price)
                else:  # Was short, now exiting
                    buy_dates.append(current_date)
                    buy_prices.append(exit_price)
                    
                # Count exit reason
                exit_reasons[daily_limit_reason] += 1
            
            # Skip trading for the rest of the day
            risk_manager.daily_trading_enabled = False
            logger.info(f"Daily risk limit exceeded. Stopping trading for the day at {current_date}")
            
            # Take a step in the environment to advance to next day
            obs, _, terminated, truncated, _ = env.step(2)  # Use hold action
            done = terminated or truncated
            
            # Record data for plotting
            dates.append(current_date)
            price_history.append(current_price)
            portfolio_history.append(float(risk_manager.net_worth))
            
            if done:
                break
            continue
        
        # Check if we need to exit based on risk management rules
        exit_triggered, exit_reason = risk_manager.check_exits(
            close_price=current_price,
            high_price=current_high,
            low_price=current_low
        )
        
        if exit_triggered:
            # For exit price, use the appropriate price based on exit reason and position
            exit_price = current_price  # Default to close price
            
            if exit_reason == "stop_loss":
                # For futures, we need to work backwards from desired P&L to price
                point_value = money.to_decimal(20.0)  # $20 per point for NQ
                
                # Calculate how many points we need to move to achieve exact stop loss
                # stop_loss_pct is % of portfolio, so we convert to dollars first
                target_loss_dollars = risk_manager.entry_portfolio_value * (risk_manager.stop_loss_pct / 100)
                
                # Calculate how many points that equals based on contracts
                target_loss_points = target_loss_dollars / (point_value * risk_manager.current_contracts)
                
                # Calculate exact exit price
                if risk_manager.position == 1:  # Long position
                    exit_price = risk_manager.entry_price - target_loss_points
                else:  # Short position
                    exit_price = risk_manager.entry_price + target_loss_points
                
                logging.info(f"Stop loss at exact price: entry={float(risk_manager.entry_price)}, exit={float(exit_price)}, points={float(target_loss_points)}")
            elif exit_reason == "take_profit":
                # For futures, we need to work backwards from desired P&L to price
                point_value = money.to_decimal(20.0)  # $20 per point for NQ
                
                # Calculate how many points we need to move to achieve exact take profit
                # take_profit_pct is % of portfolio, so we convert to dollars first
                target_profit_dollars = risk_manager.entry_portfolio_value * (risk_manager.take_profit_pct / 100)
                
                # Calculate how many points that equals based on contracts
                target_profit_points = target_profit_dollars / (point_value * risk_manager.current_contracts)
                
                # Calculate exact exit price
                if risk_manager.position == 1:  # Long position
                    exit_price = risk_manager.entry_price + target_profit_points
                else:  # Short position
                    exit_price = risk_manager.entry_price - target_profit_points
                
                logging.info(f"Take profit at exact price: entry={float(risk_manager.entry_price)}, exit={float(exit_price)}, points={float(target_profit_points)}")
            elif exit_reason == "trailing_stop":
                exit_price = current_low  # Use low price for trailing stop (worst case)
            
            # Record portfolio value before exit for profit checking
            portfolio_before_exit = risk_manager.net_worth
            
            # Exit the position with the appropriate price
            risk_manager.exit_position(exit_price, current_date, exit_reason)
            
            # Calculate profit from this trade
            trade_profit = risk_manager.net_worth - portfolio_before_exit
            
            # Check for unrealistic profits
            price_diff = abs(money.to_decimal(exit_price) - risk_manager.entry_price)
            expected_max_profit = price_diff * money.to_decimal(20)  # $20 per point for NQ
            
            if trade_profit > unrealistic_profit_threshold and trade_profit > expected_max_profit * multiplier_threshold:
                unrealistic_profit_count += 1
                logger.warning(f"Unrealistic profit detected: ${float(trade_profit):.2f} "
                              f"from price change of {price_diff:.2f} points. "
                              f"Expected max: ${expected_max_profit:.2f}")
            
            # Record exit
            if risk_manager.position == 1:  # Was long, now exiting
                sell_dates.append(current_date)
                sell_prices.append(exit_price)
            else:  # Was short, now exiting
                buy_dates.append(current_date)
                buy_prices.append(exit_price)
                
            # Count exit reason
            exit_reasons[exit_reason] += 1
        
        # Only process model predictions if trading is enabled for the day
        if risk_manager.daily_trading_enabled:
            # Close positions on second-to-last candle of the day
            if is_second_to_last_candle and risk_manager.position != 0:
                if verbose > 0:
                    logger.info(f"Second-to-last candle of day detected at {current_date_et}. Closing position.")
                
                # For second-to-last candle exit, use open price instead of close price
                exit_price = test_data.loc[test_data.index[current_index], "Open"]
                
                # Record portfolio value before exit for profit checking
                portfolio_before_exit = risk_manager.net_worth
                
                # Calculate current unrealized P&L
                if risk_manager.position != 0:
                    # Convert exit_price to Decimal to match entry_price type
                    exit_price_decimal = money.to_decimal(exit_price)
                    if risk_manager.position == 1:  # Long position
                        unrealized_pnl = (exit_price_decimal - risk_manager.entry_price) * money.to_decimal(20) * risk_manager.current_contracts
                    else:  # Short position
                        unrealized_pnl = (risk_manager.entry_price - exit_price_decimal) * money.to_decimal(20) * risk_manager.current_contracts
                    
                    # Log unrealized P&L before exit
                    logger.info(f"Unrealized P&L before second-to-last candle exit: ${float(unrealized_pnl):.2f}")
                    
                    # Exit position with second-to-last candle reason
                    risk_manager.exit_position(exit_price, current_date, "second_to_last_candle")
                    
                    # Calculate actual profit/loss from this trade
                    trade_profit = risk_manager.net_worth - portfolio_before_exit
                    logger.info(f"Second-to-last candle trade P&L: ${float(trade_profit):.2f} (using open price: ${float(exit_price):.2f})")
                    
                    # Record exit
                    if risk_manager.position == 1:  # Was long, now exiting
                        sell_dates.append(current_date)
                        sell_prices.append(exit_price)
                    else:  # Was short, now exiting
                        buy_dates.append(current_date)
                        buy_prices.append(exit_price)
                    
                    # Count exit reason
                    exit_reasons["second_to_last_candle"] += 1
                    logger.info(f"Position closed at second-to-last candle. Exit reason recorded: second_to_last_candle")
            
            # Get model prediction only if not on the last candle of the day
            if not is_last_candle and not is_second_to_last_candle:
                # Get model prediction - using deterministic parameter
                action, _ = model.predict(obs, deterministic=deterministic)
                current_action = int(action)
                action_counts[current_action] += 1
                
                # Update risk manager's stops if we have an open position
                if risk_manager.position != 0:
                    risk_manager.update_stops(
                        close_price=current_price,
                        high_price=current_high,
                        low_price=current_low
                    )
                
                # Process model's action
                if current_action == 0:  # Model suggests long
                    if risk_manager.position == -1:  # Currently short, need to exit
                        # For short exit on model signal, use the close price
                        exit_price = current_price  # Use closing price for model signal exits
                        
                        # Record portfolio value before exit for profit checking
                        portfolio_before_exit = risk_manager.net_worth
                        
                        # Exit short position
                        risk_manager.exit_position(exit_price, current_date, "model_signal")
                        
                        # Calculate profit from this trade
                        trade_profit = risk_manager.net_worth - portfolio_before_exit
                        
                        # Check for unrealistic profits
                        price_diff = abs(money.to_decimal(exit_price) - risk_manager.entry_price)
                        expected_max_profit = price_diff * money.to_decimal(20)  # $20 per point for NQ
                        
                        if trade_profit > unrealistic_profit_threshold and trade_profit > expected_max_profit * multiplier_threshold:
                            unrealistic_profit_count += 1
                            logger.warning(f"Unrealistic profit detected: ${float(trade_profit):.2f} "
                                          f"from price change of {price_diff:.2f} points. "
                                          f"Expected max: ${expected_max_profit:.2f}")
                        
                        exit_reasons["model_signal"] += 1
                        
                        # Record exit
                        buy_dates.append(current_date)
                        buy_prices.append(exit_price)
                        
                    elif risk_manager.position == 0:  # Not in a position, enter long
                        # Calculate position size
                        contracts = risk_manager.calculate_position_size(current_price)
                        
                        # Only enter position if contracts > 0 (sufficient portfolio size)
                        if contracts > 0:
                            # Enter long position
                            risk_manager.enter_position(1, current_price, current_date, contracts)
                            
                            # Record entry
                            buy_dates.append(current_date)
                            buy_prices.append(current_price)
                        else:
                            logger.warning(f"Skipped entering long position - insufficient portfolio size (${float(risk_manager.net_worth):.2f})")
                    
                elif current_action == 1:  # Model suggests short
                    if risk_manager.position == 1:  # Currently long, need to exit
                        # For long exit on model signal, use the close price
                        exit_price = current_price  # Use closing price for model signal exits
                        
                        # Record portfolio value before exit for profit checking
                        portfolio_before_exit = risk_manager.net_worth
                        
                        # Exit long position
                        risk_manager.exit_position(exit_price, current_date, "model_signal")
                        
                        # Calculate profit from this trade
                        trade_profit = risk_manager.net_worth - portfolio_before_exit
                        
                        # Check for unrealistic profits
                        price_diff = abs(money.to_decimal(exit_price) - risk_manager.entry_price)
                        expected_max_profit = price_diff * money.to_decimal(20)  # $20 per point for NQ
                        
                        if trade_profit > unrealistic_profit_threshold and trade_profit > expected_max_profit * multiplier_threshold:
                            unrealistic_profit_count += 1
                            logger.warning(f"Unrealistic profit detected: ${float(trade_profit):.2f} "
                                          f"from price change of {price_diff:.2f} points. "
                                          f"Expected max: ${expected_max_profit:.2f}")
                        
                        exit_reasons["model_signal"] += 1
                        
                        # Record exit
                        sell_dates.append(current_date)
                        sell_prices.append(exit_price)
                        
                    elif risk_manager.position == 0:  # Not in a position, enter short
                        # Calculate position size
                        contracts = risk_manager.calculate_position_size(current_price)
                        
                        # Only enter position if contracts > 0 (sufficient portfolio size)
                        if contracts > 0:
                            # Enter short position
                            risk_manager.enter_position(-1, current_price, current_date, contracts)
                            
                            # Record entry
                            sell_dates.append(current_date)
                            sell_prices.append(current_price)
                        else:
                            logger.warning(f"Skipped entering short position - insufficient portfolio size (${float(risk_manager.net_worth):.2f})")
                
                else:  # Model suggests hold (action 2)
                    # No position change, just maintain current position
                    pass
            # If we're on the last candle of the day or second-to-last candle, hold
            else:
                if verbose > 1:
                    if is_last_candle:
                        logger.info(f"Last candle of the day: {current_date_et}. Holding position (no new trades).")
                    elif is_second_to_last_candle:
                        logger.info(f"Second-to-last candle of the day: {current_date_et}. Holding position (no new trades).")
                
                # Force hold action on last candle
                current_action = 2
                action_counts[2] += 1
        else:
            # If trading is disabled for the day, use hold action
            action = 2
            action_counts[2] += 1
        
        # Take a step in the environment
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        
        # Record data for plotting
        dates.append(current_date)
        price_history.append(current_price)
        portfolio_history.append(float(risk_manager.net_worth))  # Convert Decimal to float for plotting
        
        # Check if we're done
        if done:
            # If we still have an open position at the end, close it
            if risk_manager.position != 0:
                # For end of period, use close price (most realistic)
                exit_price = current_price
                
                # Record portfolio value before exit for profit checking
                portfolio_before_exit = risk_manager.net_worth
                
                # Exit position
                risk_manager.exit_position(exit_price, current_date, "end_of_period")
                
                # Calculate profit from this trade
                trade_profit = risk_manager.net_worth - portfolio_before_exit
                
                # Check for unrealistic profits
                price_diff = abs(money.to_decimal(exit_price) - risk_manager.entry_price)
                expected_max_profit = price_diff * money.to_decimal(20)  # $20 per point for NQ
                
                if trade_profit > unrealistic_profit_threshold and trade_profit > expected_max_profit * multiplier_threshold:
                    unrealistic_profit_count += 1
                    logger.warning(f"Unrealistic profit detected: ${float(trade_profit):.2f} "
                                  f"from price change of {price_diff:.2f} points. "
                                  f"Expected max: ${expected_max_profit:.2f}")
                
                # Record exit
                if risk_manager.position == 1:  # Was long, now exiting
                    sell_dates.append(current_date)
                    sell_prices.append(exit_price)
                else:  # Was short, now exiting
                    buy_dates.append(current_date)
                    buy_prices.append(exit_price)
            break
        
        # Check if we need to close positions at the end of the trading day
        if close_at_end_of_day and risk_manager.position != 0:
            # Convert current timestamp to eastern time for market hours check
            eastern = pytz.timezone('US/Eastern')
            current_date_eastern = current_date.tz_convert(eastern)
            
            # Get next timestamp in eastern time
            if current_index + 1 < len(test_data):
                next_date_eastern = test_data.index[current_index + 1].tz_convert(eastern)
            else:
                next_date_eastern = None
            
            # Check if we're at the end of the trading day (current day != next day)
            # or if we're at the last candle of the day (4:00 PM ET)
            is_end_of_day = False
            if next_date_eastern is None:
                is_end_of_day = True
            elif current_date_eastern.date() != next_date_eastern.date():
                is_end_of_day = True
            elif current_date_eastern.hour == 16 and current_date_eastern.minute == 0:  # 4:00 PM ET
                is_end_of_day = True
            
            # Skip end-of-day closing if we already closed at the second-to-last candle
            if is_end_of_day and not is_second_to_last_candle:
                if verbose > 0:
                    logger.info(f"End of trading day detected at {current_date_eastern}. Closing position.")
                
                # For end of day, use close price
                exit_price = current_price
                
                # Record portfolio value before exit for profit checking
                portfolio_before_exit = risk_manager.net_worth
                
                # Calculate current unrealized P&L
                if risk_manager.position != 0:
                    # Convert exit_price to Decimal to match entry_price type
                    exit_price_decimal = money.to_decimal(exit_price)
                    if risk_manager.position == 1:  # Long position
                        unrealized_pnl = (exit_price_decimal - risk_manager.entry_price) * money.to_decimal(20) * risk_manager.current_contracts
                    else:  # Short position
                        unrealized_pnl = (risk_manager.entry_price - exit_price_decimal) * money.to_decimal(20) * risk_manager.current_contracts
                    
                    # Log unrealized P&L before exit
                    logger.info(f"Unrealized P&L before end-of-day exit: ${float(unrealized_pnl):.2f}")
                    
                    # Exit position with end_of_day reason
                    risk_manager.exit_position(exit_price, current_date, "end_of_day")
                    
                    # Calculate actual profit/loss from this trade
                    trade_profit = risk_manager.net_worth - portfolio_before_exit
                    logger.info(f"End-of-day trade P&L: ${float(trade_profit):.2f}")
                    
                    # Record exit
                    if risk_manager.position == 1:  # Was long, now exiting
                        sell_dates.append(current_date)
                        sell_prices.append(exit_price)
                    else:  # Was short, now exiting
                        buy_dates.append(current_date)
                        buy_prices.append(exit_price)
                    
                    # Count exit reason
                    exit_reasons["end_of_day"] += 1
                    logger.info(f"End of day position closed. Exit reason recorded: end_of_day")
                else:
                    logger.info("No position to close at end of day")
    
    # Log warning if unrealistic profits were detected
    if unrealistic_profit_count > 0:
        logger.warning(f"Detected {unrealistic_profit_count} trades with potentially unrealistic profits")
    
    # Calculate performance metrics using the money module
    initial_balance_decimal = money.to_decimal(initial_balance)
    total_return_pct = money.calculate_return_pct(risk_manager.net_worth, initial_balance_decimal)
    
    # Log results
    if verbose > 0:
        logger.info(f"Trading completed with risk management:")
        logger.info(f"  Initial balance: ${money.format_money_str(initial_balance_decimal)}")
        logger.info(f"  Final balance: ${money.format_money_str(risk_manager.net_worth)}")
        logger.info(f"  Total return: {money.format_money_str(total_return_pct)}%")
        logger.info(f"  Total trades: {len(risk_manager.trade_history) // 2}")
        logger.info(f"  Exit reasons: {exit_reasons}")
        if unrealistic_profit_count > 0:
            logger.info(f"  Unrealistic profit trades: {unrealistic_profit_count}")
    
    # Return results
    return {
        "final_portfolio_value": float(money.format_money(risk_manager.net_worth, 2)),
        "total_return_pct": float(money.format_money(total_return_pct, 2)),
        "trade_count": len(risk_manager.trade_history) // 2,
        "final_position": risk_manager.position,
        "dates": dates,
        "price_history": price_history,
        "portfolio_history": portfolio_history,
        "trade_history": risk_manager.trade_history,
        "buy_dates": buy_dates,
        "buy_prices": buy_prices,
        "sell_dates": sell_dates,
        "sell_prices": sell_prices,
        "exit_reasons": exit_reasons,
        "action_counts": action_counts,
        "profitable_trades": sum(1 for trade in risk_manager.trade_history if "profit" in trade and float(trade["profit"]) > 0),
        "unrealistic_profit_count": unrealistic_profit_count
    }

def plot_results(results: Dict, plots_dir: str = None) -> None:
    """
    Plot trading results using Plotly.
    
    Args:
        results: Results dictionary from trade_with_risk_management
        plots_dir: Directory to save the plot to (if None, just show the plot)
    """
    dates = results["dates"]
    price_history = results["price_history"]
    portfolio_history = results["portfolio_history"]
    
    # Get trade history to identify end-of-day trades
    trade_history = results["trade_history"]
    
    # Separate end-of-day trades from regular trades
    eod_buy_dates = []
    eod_buy_prices = []
    eod_sell_dates = []
    eod_sell_prices = []
    
    regular_buy_dates = []
    regular_buy_prices = []
    regular_sell_dates = []
    regular_sell_prices = []
    
    # Process trade history to separate end-of-day trades
    for i in range(1, len(trade_history)):
        trade = trade_history[i]
        # Exit trades will have an exit_reason field
        if "exit_reason" in trade and trade["exit_reason"] == "end_of_day":
            if trade["action"] == "buy":  # Closing a short position
                eod_buy_dates.append(trade["date"])
                eod_buy_prices.append(trade["price"])
            elif trade["action"] == "sell":  # Closing a long position
                eod_sell_dates.append(trade["date"])
                eod_sell_prices.append(trade["price"])
    
    # Regular trades are those not in end-of-day lists
    for date, price in zip(results["buy_dates"], results["buy_prices"]):
        if date not in eod_buy_dates:
            regular_buy_dates.append(date)
            regular_buy_prices.append(price)
    
    for date, price in zip(results["sell_dates"], results["sell_prices"]):
        if date not in eod_sell_dates:
            regular_sell_dates.append(date)
            regular_sell_prices.append(price)
            
    # Create subplots with 2 rows in one column and shared X-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "Asset Price and Trade Signals",
            "Portfolio Value"
        )
    )

    # Plot Price line in the first row
    fig.add_trace(
        go.Scatter(x=dates, y=price_history, name="Price", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Plot Regular Buy signals with triangle-up markers
    fig.add_trace(
        go.Scatter(
            x=regular_buy_dates,
            y=regular_buy_prices,
            name="Buy",
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=12)
        ),
        row=1, col=1
    )
    
    # Plot Regular Sell signals with triangle-down markers
    fig.add_trace(
        go.Scatter(
            x=regular_sell_dates,
            y=regular_sell_prices,
            name="Sell",
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12)
        ),
        row=1, col=1
    )
    
    # Plot End-of-Day Buy signals with different markers
    if eod_buy_dates:
        fig.add_trace(
            go.Scatter(
                x=eod_buy_dates,
                y=eod_buy_prices,
                name="EOD Buy",
                mode="markers",
                marker=dict(symbol="triangle-up", color="cyan", size=12, line=dict(color="blue", width=2))
            ),
            row=1, col=1
        )
    
    # Plot End-of-Day Sell signals with different markers
    if eod_sell_dates:
        fig.add_trace(
            go.Scatter(
                x=eod_sell_dates,
                y=eod_sell_prices,
                name="EOD Sell",
                mode="markers",
                marker=dict(symbol="triangle-down", color="magenta", size=12, line=dict(color="blue", width=2))
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
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add portfolio performance annotation
    fig.add_annotation(
        text=f"Initial: ${config['environment']['initial_balance']:,.2f}<br>Final: ${results['final_portfolio_value']:,.2f}<br>Return: {results['total_return_pct']}%",
        xref="paper", yref="paper",
        x=1.0, y=0.4,
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    
    # Add exit reasons annotation
    exit_reasons_text = "<br>".join([f"{reason}: {count}" for reason, count in results["exit_reasons"].items() if count > 0])
    fig.add_annotation(
        text=f"Exit Reasons:<br>{exit_reasons_text}",
        xref="paper", yref="paper",
        x=1.0, y=0.2,
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    
    # Update layout for titles and axes
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title="Asset Price ($)", tickprefix="$"),
        xaxis2=dict(title="Date"),
        yaxis2=dict(title="Portfolio Value ($)", tickprefix="$"),
        legend=dict(
            x=0.5,
            y=1.15,
            xanchor="center",
            orientation="h"
        ),
        margin=dict(l=60, r=60, t=100, b=50)
    )
    
    # Save the plot if a directory is provided
    if plots_dir:
        import os
        from datetime import datetime
        
        # Create the directory if it doesn't exist
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{plots_dir}/trade_results_{timestamp}.html"
        
        # Save the plot
        fig.write_html(filename)
        logger.info(f"Plot saved to {filename}")
    
    # Always show the plot
    fig.show()

def save_trade_history(trade_history: List[Dict], filename: str = "risk_managed_trade_history.csv") -> None:
    """
    Export the trade history to a CSV file.

    Args:
        trade_history: List of trade event dictionaries
        filename: Output filename
    """
    # Convert to DataFrame
    trade_history_df = pd.DataFrame(trade_history)
    
    # If the DataFrame is empty, just save an empty file and return
    if trade_history_df.empty:
        trade_history_df.to_csv(filename, index=False)
        return
    
    # Handle different trade history formats
    # Some might have 'action' field, others might have 'trade_type' field
    action_field = 'action' if 'action' in trade_history_df.columns else 'trade_type'
    
    # Check if the required columns for grouping exist
    if 'date' in trade_history_df.columns and action_field in trade_history_df.columns:
        # Filter out duplicate trades
        # After analyzing the data, we understand that:
        # 1. Same timestamp, same action: This is a duplicate and we should keep only one (preferably with profit/exit_reason)
        # 2. Same timestamp, different actions: This is NOT a duplicate - it's an exit of one position and entry of another
        
        filtered_df = pd.DataFrame()
        
        # Group by date and action to find duplicates with same timestamp and action
        grouped = trade_history_df.groupby(['date', action_field])
        
        for (date, action), group in grouped:
            if len(group) > 1:
                # Duplicate entries with same timestamp and action
                # Keep the one with profit/exit_reason if available
                if 'profit' in group.columns and group['profit'].notna().any():
                    filtered_df = pd.concat([filtered_df, group[group['profit'].notna()]])
                else:
                    # If no profit info, keep the first one
                    filtered_df = pd.concat([filtered_df, group.iloc[[0]]])
            else:
                # Not a duplicate, keep it
                filtered_df = pd.concat([filtered_df, group])
                
        # Use the filtered dataframe if it's not empty
        if not filtered_df.empty:
            trade_history_df = filtered_df
    else:
        # If we don't have the required columns for grouping, just use the original DataFrame
        pass
        
    # Save to CSV
    trade_history_df.to_csv(filename, index=False)
    logging.info(f"Trade history saved to {filename}")

def main():
    """
    Main function to run trading with risk management.
    """
    # Set random seeds for reproducibility
    import numpy as np
    import torch
    import random
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    
    # Load data
    _, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=config["data"]["train_ratio"]
    )
    
    # Get risk management configuration from config.yaml
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", True)
    
    # Initialize risk parameters
    stop_loss_pct = None
    take_profit_pct = None
    trailing_stop_pct = None
    position_size = 1.0
    max_risk_per_trade_pct = 2.0
    daily_risk_limit = None  # Add daily risk limit parameter
    
    # Apply risk management configuration if enabled
    if risk_enabled:
        # Daily risk limit configuration
        daily_risk_config = risk_config.get("daily_risk_limit", {})
        if daily_risk_config.get("enabled", False):
            daily_risk_limit = daily_risk_config.get("max_daily_loss", 1000.0)
        
        # Stop loss configuration
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_pct = stop_loss_config.get("percentage", 1.0)
        
        # Take profit configuration
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_pct = take_profit_config.get("percentage", 2.0)
        
        # Trailing stop configuration
        trailing_stop_config = risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
        
        # Position sizing configuration
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            position_size = position_sizing_config.get("size_multiplier", 1.0)
            max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 2.0)
    
    # Log risk management configuration
    logger.info("Risk Management Configuration:")
    logger.info(f"  Risk Management Enabled: {risk_enabled}")
    if risk_enabled:
        logger.info(f"  Daily Risk Limit: {'Enabled' if daily_risk_limit is not None else 'Disabled'}" + 
                   (f" (${daily_risk_limit:.2f})" if daily_risk_limit is not None else ""))
        logger.info(f"  Stop Loss: {'Enabled' if stop_loss_pct is not None else 'Disabled'}" + 
                   (f" ({stop_loss_pct}%)" if stop_loss_pct is not None else ""))
        logger.info(f"  Take Profit: {'Enabled' if take_profit_pct is not None else 'Disabled'}" + 
                   (f" ({take_profit_pct}%)" if take_profit_pct is not None else ""))
        logger.info(f"  Trailing Stop: {'Enabled' if trailing_stop_pct is not None else 'Disabled'}" + 
                   (f" ({trailing_stop_pct}%)" if trailing_stop_pct is not None else ""))
        logger.info(f"  Position Size Multiplier: {position_size}")
        logger.info(f"  Max Risk Per Trade: {max_risk_per_trade_pct}%")
    
    # Run trading with risk management
    results = trade_with_risk_management(
        model_path="best_model",
        test_data=test_data,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        position_size=position_size,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 0.0),
        verbose=config["training"].get("verbose", 1),
        deterministic=True,  # Use deterministic action selection for trading
        close_at_end_of_day=True,  # Close positions at the end of the trading day
        daily_risk_limit=daily_risk_limit  # Add daily risk limit parameter
    )
    
    # Plot results
    plot_results(results)
    
    # Save trade history
    save_trade_history(results["trade_history"])
    
    # Print summary
    print("\nTrading Summary:")
    print(f"Initial Balance: ${config['environment']['initial_balance']:.2f}")
    print(f"Final Balance: ${results['final_portfolio_value']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['trade_count']}")
    print(f"Exit Reasons: {results['exit_reasons']}")

if __name__ == "__main__":
    main() 