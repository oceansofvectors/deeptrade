from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging

# Configure decimal context for financial calculations
# 28 digits of precision should be more than enough for any financial calculation
# ROUND_HALF_UP is the standard rounding mode for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

# Constants
POINT_VALUE = Decimal('20.0')  # $20 per point for NQ futures
TICK_SIZE = Decimal('0.25')    # NQ futures have 0.25 point ticks

def to_decimal(value):
    """
    Convert a value to Decimal with proper handling of various input types.
    
    Args:
        value: Value to convert (float, int, str, or Decimal)
        
    Returns:
        Decimal: Converted value
    """
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float)):
        return Decimal(str(value))
    elif isinstance(value, str):
        return Decimal(value)
    else:
        raise TypeError(f"Cannot convert {type(value)} to Decimal")

def calculate_price_change(entry_price, exit_price, position):
    """
    Calculate price change with proper precision.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        position: Position (1 for long, -1 for short)
        
    Returns:
        Decimal: Price change in points
    """
    entry_price = to_decimal(entry_price)
    exit_price = to_decimal(exit_price)
    
    if position == 1:  # Long position
        return exit_price - entry_price
    else:  # Short position
        return entry_price - exit_price

def calculate_dollar_change(price_change, contracts):
    """
    Calculate dollar change based on price change and number of contracts.
    
    For NQ futures, each full point is worth $20.
    The price_change should be the raw difference between prices.
    
    Args:
        price_change: Price change in points
        contracts: Number of contracts
        
    Returns:
        Decimal: Dollar change
    """
    price_change = to_decimal(price_change)
    contracts = to_decimal(contracts)
    
    # For NQ futures, each point is $20
    # Ensure we're calculating based on the exact price difference
    return price_change * POINT_VALUE * contracts

def calculate_transaction_cost(price, contracts, transaction_cost_pct):
    """
    Calculate transaction cost with proper precision.
    
    Args:
        price: Trade price
        contracts: Number of contracts
        transaction_cost_pct: Transaction cost as percentage
        
    Returns:
        Decimal: Transaction cost in dollars
    """
    price = to_decimal(price)
    contracts = to_decimal(contracts)
    transaction_cost_pct = to_decimal(transaction_cost_pct)
    
    return price * contracts * POINT_VALUE * (transaction_cost_pct / Decimal('100'))

def calculate_net_profit(dollar_change, transaction_cost):
    """
    Calculate net profit with proper precision.
    
    Args:
        dollar_change: Dollar change from price movement
        transaction_cost: Transaction cost in dollars
        
    Returns:
        Decimal: Net profit in dollars
    """
    dollar_change = to_decimal(dollar_change)
    transaction_cost = to_decimal(transaction_cost)
    
    return dollar_change - transaction_cost

def calculate_return_pct(final_value, initial_value):
    """
    Calculate percentage return with proper precision.
    
    Args:
        final_value: Final portfolio value
        initial_value: Initial portfolio value
        
    Returns:
        Decimal: Percentage return
    """
    final_value = to_decimal(final_value)
    initial_value = to_decimal(initial_value)
    
    if initial_value == Decimal('0'):
        return Decimal('0')
    
    return ((final_value - initial_value) / initial_value) * Decimal('100')

def format_money(value, decimal_places=2):
    """
    Format a monetary value with proper rounding and precision.
    
    Args:
        value: Value to format
        decimal_places: Number of decimal places to round to
        
    Returns:
        Decimal: Formatted value
    """
    value = to_decimal(value)
    return value.quantize(Decimal(f'0.{"0" * decimal_places}'))

def format_money_str(value, decimal_places=2):
    """
    Format a monetary value as a string with proper rounding and precision.
    
    Args:
        value: Value to format
        decimal_places: Number of decimal places to round to
        
    Returns:
        str: Formatted value as string
    """
    formatted = format_money(value, decimal_places)
    return f"{formatted:,.{decimal_places}f}"

def calculate_stop_loss_price(entry_price, stop_loss_pct, position):
    """
    Calculate stop loss price with proper precision.
    
    Args:
        entry_price: Entry price
        stop_loss_pct: Stop loss percentage
        position: Position (1 for long, -1 for short)
        
    Returns:
        Decimal: Stop loss price
    """
    entry_price = to_decimal(entry_price)
    stop_loss_pct = to_decimal(stop_loss_pct)
    
    if position == 1:  # Long position
        return entry_price * (Decimal('1') - stop_loss_pct / Decimal('100'))
    else:  # Short position
        return entry_price * (Decimal('1') + stop_loss_pct / Decimal('100'))

def calculate_take_profit_price(entry_price, take_profit_pct, position):
    """
    Calculate take profit price with proper precision.
    
    Args:
        entry_price: Entry price
        take_profit_pct: Take profit percentage
        position: Position (1 for long, -1 for short)
        
    Returns:
        Decimal: Take profit price
    """
    entry_price = to_decimal(entry_price)
    take_profit_pct = to_decimal(take_profit_pct)
    
    if position == 1:  # Long position
        return entry_price * (Decimal('1') + take_profit_pct / Decimal('100'))
    else:  # Short position
        return entry_price * (Decimal('1') - take_profit_pct / Decimal('100'))

def calculate_trailing_stop_price(reference_price, trailing_stop_pct, position):
    """
    Calculate trailing stop price with proper precision.
    
    Args:
        reference_price: Reference price (highest for long, lowest for short)
        trailing_stop_pct: Trailing stop percentage
        position: Position (1 for long, -1 for short)
        
    Returns:
        Decimal: Trailing stop price
    """
    reference_price = to_decimal(reference_price)
    trailing_stop_pct = to_decimal(trailing_stop_pct)
    
    if position == 1:  # Long position
        return reference_price * (Decimal('1') - trailing_stop_pct / Decimal('100'))
    else:  # Short position
        return reference_price * (Decimal('1') + trailing_stop_pct / Decimal('100'))

def calculate_portfolio_stop_loss_price(entry_price, stop_loss_pct, position, portfolio_value, contracts, initial_price_per_point=20.0):
    """
    Calculate stop loss price based on a percentage of the portfolio value instead of entry price.
    
    Args:
        entry_price: Entry price
        stop_loss_pct: Stop loss percentage of portfolio
        position: Position (1 for long, -1 for short)
        portfolio_value: Current portfolio value
        contracts: Number of contracts
        initial_price_per_point: Dollar value per point (default: $20 for NQ futures)
        
    Returns:
        Decimal: Stop loss price
    """
    entry_price = to_decimal(entry_price)
    stop_loss_pct = to_decimal(stop_loss_pct)
    portfolio_value = to_decimal(portfolio_value)
    contracts = to_decimal(contracts)
    initial_price_per_point = to_decimal(initial_price_per_point)
    
    # Calculate maximum loss in dollars (percentage of portfolio)
    max_loss_dollars = portfolio_value * (stop_loss_pct / Decimal('100'))
    
    # Calculate how many points this represents based on contracts and price per point
    max_loss_points = max_loss_dollars / (contracts * initial_price_per_point)
    
    # Calculate the stop loss price based on entry price and max loss points
    if position == 1:  # Long position
        return entry_price - max_loss_points
    else:  # Short position
        return entry_price + max_loss_points

def calculate_portfolio_take_profit_price(entry_price, take_profit_pct, position, portfolio_value, contracts, initial_price_per_point=20.0):
    """
    Calculate take profit price based on a percentage of the portfolio value instead of entry price.
    
    Args:
        entry_price: Entry price
        take_profit_pct: Take profit percentage of portfolio
        position: Position (1 for long, -1 for short)
        portfolio_value: Current portfolio value
        contracts: Number of contracts
        initial_price_per_point: Dollar value per point (default: $20 for NQ futures)
        
    Returns:
        Decimal: Take profit price
    """
    entry_price = to_decimal(entry_price)
    take_profit_pct = to_decimal(take_profit_pct)
    portfolio_value = to_decimal(portfolio_value)
    contracts = to_decimal(contracts)
    initial_price_per_point = to_decimal(initial_price_per_point)
    
    # Calculate target profit in dollars (percentage of portfolio)
    target_profit_dollars = portfolio_value * (take_profit_pct / Decimal('100'))
    
    # Calculate how many points this represents based on contracts and price per point
    target_profit_points = target_profit_dollars / (contracts * initial_price_per_point)
    
    # Calculate the take profit price based on entry price and target profit points
    if position == 1:  # Long position
        return entry_price + target_profit_points
    else:  # Short position
        return entry_price - target_profit_points 