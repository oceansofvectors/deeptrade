from datetime import datetime, timedelta
from collections import defaultdict
import pytz
import logging

logger = logging.getLogger(__name__)

# Constants for bar synchronization
FIVE_SEC_PER_BAR = 5
BARS_PER_FIVE_MIN = (5 * 60) // FIVE_SEC_PER_BAR  # 60
UTC = pytz.UTC           # convenience alias
ROUND_TO = timedelta(minutes=5)

# Globals for real-time bar handling
bar_buckets = defaultdict(list)  # Organize bars by 5-minute intervals

# Function to aggregate a list of 5-second bars into one 5-minute bar.
def aggregate_bars(bars):
    """
    Given a list of real-time bars (5-sec each), compute a 5-minute bar.
    """
    if not bars:
        return None
    
    # Handle IB bars which may have different attribute names
    try:    
        # Check if we have IB bar objects
        if hasattr(bars[0], 'open_'):
            open_price = bars[0].open_
            high_price = max(bar.high for bar in bars)
            low_price = min(bar.low for bar in bars)
            close_price = bars[-1].close
            volume = sum(bar.volume for bar in bars)
            
            # Handle different time formats
            if isinstance(bars[0].time, datetime):
                start_time = bars[0].time
                end_time = bars[-1].time
            else:
                try:
                    start_time = datetime.fromtimestamp(bars[0].time)
                    end_time = datetime.fromtimestamp(bars[-1].time)
                except (TypeError, ValueError):
                    start_time = datetime.now() - timedelta(minutes=5)
                    end_time = datetime.now()
        else:
            # Handle dictionary format
            open_price = bars[0]['open']
            high_price = max(bar['high'] for bar in bars)
            low_price = min(bar['low'] for bar in bars)
            close_price = bars[-1]['close']
            volume = sum(bar['volume'] for bar in bars)
            
            # Get time values
            start_time = bars[0].get('time', datetime.now() - timedelta(minutes=5))
            end_time = bars[-1].get('time', datetime.now())
        
        # Ensure times are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        else:
            start_time = start_time.astimezone(UTC)
            
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)
        else:
            end_time = end_time.astimezone(UTC)
            
        # Create the interval end time as key for the bar
        interval_end = end_of_interval(end_time)
        
        # Create and return the aggregated bar
        return {
            'time': interval_end,  # Use interval end for indexing
            'start': start_time,
            'end': end_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
    except Exception as e:
        logger.error(f"Error aggregating bars: {e}")
        return None

def get_interval_key(timestamp):
    """
    Convert a timestamp to its 5-minute interval key.
    Example: 09:32:45 -> '09:30'
    
    Handles timezone-aware datetime objects by removing the timezone info.
    """
    # Handle timezone-aware datetimes by removing timezone info if present
    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
        
    # Floor to nearest 5-minute interval
    minute = (timestamp.minute // 5) * 5
    interval_time = timestamp.replace(minute=minute, second=0, microsecond=0)
    return interval_time.strftime('%Y-%m-%d %H:%M')

def end_of_interval(ts: datetime, width=ROUND_TO) -> datetime:
    """
    Return the *end* of the current `width`-minute interval.
    `ts` must be timezone-aware (UTC recommended).
    """
    ts = ts.astimezone(UTC)
    floored = ts - timedelta(
        minutes=ts.minute % (width.seconds // 60),
        seconds=ts.second,
        microseconds=ts.microsecond,
    )
    return floored + width  # ⬅️  end, not start

def synchronize_bars():
    """
    Check if we have complete 5-minute bars to process.
    Returns the latest complete bar if available, None otherwise.
    """
    now_utc = datetime.now(UTC) - timedelta(seconds=1)
    last_end = end_of_interval(now_utc)
    key = last_end.strftime("%Y-%m-%d %H:%M")

    if key in bar_buckets and len(bar_buckets[key]) >= BARS_PER_FIVE_MIN - 5: # At least 55 bars in case connection lost, its ok.
        logger.info(f"Processing synchronized 5-minute bar for interval {key} with exactly {BARS_PER_FIVE_MIN} 5-second bars")
        bars = bar_buckets.pop(key)
        
        # Clean up old buckets to prevent memory growth
        current_time = datetime.now(UTC)
        for old_key in list(bar_buckets.keys()):
            try:
                old_time = datetime.strptime(old_key, '%Y-%m-%d %H:%M').replace(tzinfo=UTC)
                if (current_time - old_time).total_seconds() > 3600:  # Older than 1 hour
                    del bar_buckets[old_key]
                    logger.debug(f"Cleaned up old bucket: {old_key}")
            except Exception as e:
                logger.warning(f"Error cleaning up old bar bucket {old_key}: {e}")
        
        return aggregate_bars(bars)

    # Calculate and log the percentage completion of the current interval
    current_interval = end_of_interval(datetime.now(UTC))
    current_key = current_interval.strftime('%Y-%m-%d %H:%M')
    
    # Count bars in the current interval
    current_bars_count = len(bar_buckets.get(current_key, []))
    
    # Calculate elapsed seconds in this interval
    now = datetime.now(UTC)
    interval_start = current_interval - ROUND_TO
    elapsed_seconds = (now - interval_start).total_seconds()
    total_seconds = 5 * 60  # 5 minutes = 300 seconds
    time_percent = min(100, round((elapsed_seconds / total_seconds) * 100, 1))
    
    # Expected bars based on elapsed time
    expected_bars = int(elapsed_seconds / FIVE_SEC_PER_BAR)
    bars_percent = min(100, round((current_bars_count / BARS_PER_FIVE_MIN) * 100, 1))
    
    logger.info(f"Current interval: {current_key} | Time: {time_percent}% complete | "
                f"Bars: {current_bars_count}/{BARS_PER_FIVE_MIN} ({bars_percent}%) | "
                f"Next prediction at interval completion")
    
    return None 