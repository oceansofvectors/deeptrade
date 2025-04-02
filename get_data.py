import yfinance as yf
import pandas as pd
import numpy as np
from typing import List

# Monkey patch: set np.NaN to np.nan if it's missing
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta
import logging
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NEW: import YAML configuration
from config import config

# Helper function to ensure numeric values
def ensure_numeric(df, columns):
    """
    Ensure columns contain only numeric values.
    
    Args:
        df: DataFrame to process
        columns: List of column names to check/convert
        
    Returns:
        DataFrame with numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in essential columns
    df = df.dropna(subset=[col for col in columns if col in df.columns])
    return df

# Import filter_market_hours from walk_forward module
# We use try/except to handle circular import issue
try:
    from walk_forward import filter_market_hours
except ImportError:
    # Define a basic version if walk_forward isn't importable
    def filter_market_hours(data):
        logger.warning("Could not import filter_market_hours from walk_forward module. Using unfiltered data.")
        return data

def download_data(symbol: str = "NQ=F", period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Download historical data from Yahoo Finance.
    
    Args:
        symbol: Ticker symbol to download
        period: Time period to download (e.g. '60d' for 60 days)
        interval: Data interval (e.g. '5m' for 5 minutes)
        
    Returns:
        DataFrame: Historical price data
    """
    logger.info(f"Downloading {symbol} data for period {period} with interval {interval}")
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(symbol, period=period, interval=interval)
        
        if data.empty:
            logger.error(f"Failed to download data for {symbol}")
            return None
            
        logger.info(f"Downloaded {len(data)} rows of data")
        logger.info(f"Downloaded columns: {data.columns.tolist()}")
        logger.info(f"Downloaded index type: {type(data.index)}")
        
        # Save raw data to CSV
        os.makedirs('data', exist_ok=True)
        raw_data = data.copy()
        raw_data.index.name = 'Datetime'  # Ensure the index has a name
        raw_filename = f'data/{symbol.replace("=", "_")}_raw.csv'
        raw_data.to_csv(raw_filename)
        logger.info(f"Saved raw data to {raw_filename}")
        
        # Create a processed dataframe with the index as the 'time' column
        processed_data = pd.DataFrame()
        processed_data['time'] = data.index.astype('int64') // 10**9  # Convert to Unix timestamp
        
        # Handle different formats of yfinance columns (single-level or multi-level)
        if isinstance(data.columns, pd.MultiIndex):
            logger.info("Detected multi-index columns from yfinance")
            # Extract the price columns we need
            for price_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                for col in data.columns:
                    if isinstance(col, tuple) and col[0] == price_col:
                        processed_data[price_col.lower()] = data[col].values
                        break
        else:
            # Standard column names
            for yf_col, our_col in [('Open', 'open'), ('High', 'high'), 
                                     ('Low', 'low'), ('Close', 'close'), 
                                     ('Volume', 'volume')]:
                if yf_col in data.columns:
                    processed_data[our_col] = data[yf_col].values
        
        logger.info(f"Processed data columns: {processed_data.columns.tolist()}")
        logger.info(f"Sample of processed data:\n{processed_data.head()}")
        
        # Ensure we have the required columns
        required_cols = ['time', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in processed_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns after processing: {missing_cols}")
            return None
        
        # Save processed data to CSV
        processed_data.to_csv('data/nq.csv', index=False)
        logger.info(f"Saved processed data to data/nq.csv")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Ensure columns in DataFrame are numeric types and handle any conversion issues.
    
    Args:
        df: DataFrame to process
        columns: List of column names to ensure are numeric
        
    Returns:
        DataFrame with numeric columns
    """
    for col in columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Error converting {col} to numeric: {e}. Using best effort conversion.")
                # Try a more robust conversion
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

def process_technical_indicators(df: pd.DataFrame, train_ratio: float = 0.7) -> pd.DataFrame:
    """
    Process and compute technical indicators for a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        train_ratio: Proportion of data to use as training set for normalization
        
    Returns:
        DataFrame with computed technical indicators
    """
    try:
        logger.info("Processing technical indicators")
        
        # Check if we have the required columns - allow for both uppercase and lowercase
        required_caps = ['Open', 'High', 'Low', 'Close']
        required_lower = ['open', 'high', 'low', 'close']
        
        if all(col in df.columns for col in required_caps):
            # We have uppercase column names
            price_cols = required_caps
            logger.info("Using uppercase price column names")
        elif all(col in df.columns for col in required_lower):
            # We have lowercase column names
            price_cols = required_lower
            logger.info("Using lowercase price column names")
        else:
            logger.error(f"Missing required columns for indicator processing. Available: {df.columns.tolist()}")
            return df
        
        # Use the detected column names
        open_col = price_cols[0]  # 'Open' or 'open'
        high_col = price_cols[1]  # 'High' or 'high'
        low_col = price_cols[2]   # 'Low' or 'low'
        close_col = price_cols[3] # 'Close' or 'close'
        
        # Fill NaN values in price columns
        for col in price_cols:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col} column. Filling with forward fill.")
                df[col] = df[col].fillna(method='ffill')
        
        # Add 'Volume' column with defaults if not present
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume' if 'volume' in df.columns else None
        if volume_col is None:
            logger.warning("Volume column not found. Adding placeholder values.")
            df['volume'] = 1000  # Default volume
            volume_col = 'volume'
        
        # Calculate normalized close - create lowercase version for consistency
        df['close_norm'] = df[close_col].pct_change().fillna(0)
        
        # Add trend direction based on price movement
        df['trend_direction'] = np.sign(df['close_norm']).fillna(0).astype(int)
        
        # Check if 'Up Trend' column exists in the CSV
        if 'Up Trend' in df.columns and 'Down Trend' in df.columns:
            # Use the existing trend data from CSV
            df['trend_direction'] = np.where(df['Up Trend'].notna(), 1, 
                                           np.where(df['Down Trend'].notna(), -1, 0))
        
        # Add all the requested technical indicators
        
        # RSI (Relative Strength Index)
        if config["indicators"]["rsi"]["enabled"]:
            # Check if RSI data is already in the CSV
            if 'RSI' in df.columns:
                # Normalize RSI to [0, 1] range if needed
                if df['RSI'].max() > 1:
                    df['RSI'] = df['RSI'] / 100.0
            else:
                df['RSI'] = ta.rsi(df[close_col], length=config["indicators"]["rsi"]["length"]) / 100.0
        else:
            df['RSI'] = 0.0
            
        # CCI (Commodity Channel Index)
        if config["indicators"].get("cci", {}).get("enabled", False):
            if 'CCI' in df.columns:
                # Normalize CCI if needed
                if df['CCI'].max() > 1 or df['CCI'].min() < -1:
                    max_abs = max(abs(df['CCI'].max()), abs(df['CCI'].min()))
                    df['CCI'] = df['CCI'] / max_abs
            else:
                df['CCI'] = ta.cci(df[high_col], df[low_col], df[close_col], 
                                   length=config["indicators"]["cci"]["length"]) / 100.0
        
        # ADX (Average Directional Index)
        if config["indicators"].get("adx", {}).get("enabled", False):
            if 'ADX' in df.columns:
                # Normalize ADX if needed
                if df['ADX'].max() > 1:
                    df['ADX'] = df['ADX'] / 100.0
            else:
                adx_length = config["indicators"]["adx"]["length"]
                adx_result = ta.adx(df[high_col], df[low_col], df[close_col], length=adx_length)
                
                # Check the actual column names in the result
                logger.info(f"ADX result columns: {adx_result.columns}")
                
                # Extract ADX value - handle different column naming conventions
                if f'ADX_{adx_length}' in adx_result.columns:
                    df['ADX'] = adx_result[f'ADX_{adx_length}'] / 100.0
                else:
                    # Assume the first column is ADX
                    df['ADX'] = adx_result.iloc[:, 0] / 100.0
                
            # ADX Positive and Negative Directional Indicators
            if config["indicators"].get("adx_pos", {}).get("enabled", False):
                if 'ADX_POS' in df.columns:
                    if df['ADX_POS'].max() > 1:
                        df['ADX_POS'] = df['ADX_POS'] / 100.0
                else:
                    if 'adx_result' not in locals():
                        adx_length = config["indicators"]["adx"]["length"]
                        adx_result = ta.adx(df[high_col], df[low_col], df[close_col], length=adx_length)
                    
                    if f'DMP_{adx_length}' in adx_result.columns:
                        df['ADX_POS'] = adx_result[f'DMP_{adx_length}'] / 100.0
                    else:
                        # Assume the second column is DMP
                        df['ADX_POS'] = adx_result.iloc[:, 1] / 100.0
                    
            if config["indicators"].get("adx_neg", {}).get("enabled", False):
                if 'ADX_NEG' in df.columns:
                    if df['ADX_NEG'].max() > 1:
                        df['ADX_NEG'] = df['ADX_NEG'] / 100.0
                else:
                    if 'adx_result' not in locals():
                        adx_length = config["indicators"]["adx"]["length"]
                        adx_result = ta.adx(df[high_col], df[low_col], df[close_col], length=adx_length)
                    
                    if f'DMN_{adx_length}' in adx_result.columns:
                        df['ADX_NEG'] = adx_result[f'DMN_{adx_length}'] / 100.0
                    else:
                        # Assume the third column is DMN
                        df['ADX_NEG'] = adx_result.iloc[:, 2] / 100.0
        
        # Stochastic Oscillator
        # K is the fast stochastic and D is the slow stochastic
        k_length = config["indicators"].get("stoch_k", {}).get("length", 14)
        d_length = config["indicators"].get("stoch_d", {}).get("length", 3)
        
        if (config["indicators"].get("stoch_k", {}).get("enabled", False) or 
            config["indicators"].get("stoch_d", {}).get("enabled", False)):
            if 'STOCH_K' in df.columns or 'STOCH_D' in df.columns:
                # Use existing stochastic data
                if 'STOCH_K' in df.columns and config["indicators"].get("stoch_k", {}).get("enabled", False):
                    if df['STOCH_K'].max() > 1:
                        df['STOCH_K'] = df['STOCH_K'] / 100.0
                if 'STOCH_D' in df.columns and config["indicators"].get("stoch_d", {}).get("enabled", False):
                    if df['STOCH_D'].max() > 1:
                        df['STOCH_D'] = df['STOCH_D'] / 100.0
            else:
                # Calculate stochastic indicators
                stoch = ta.stoch(df[high_col], df[low_col], df[close_col], k=k_length, d=d_length, smooth_k=3)
                
                # Check the actual column names in the result
                logger.info(f"Stochastic result columns: {stoch.columns}")
                
                if config["indicators"].get("stoch_k", {}).get("enabled", False) and 'K' not in df.columns:
                    # Try different possible column names
                    if f'STOCHk_{k_length}_{d_length}_3' in stoch.columns:
                        df['STOCH_K'] = stoch[f'STOCHk_{k_length}_{d_length}_3'] / 100.0
                    elif 'STOCHk' in stoch.columns:
                        df['STOCH_K'] = stoch['STOCHk'] / 100.0
                    else:
                        # Assume the first column is K
                        df['STOCH_K'] = stoch.iloc[:, 0] / 100.0
                        
                if config["indicators"].get("stoch_d", {}).get("enabled", False) and 'D' not in df.columns:
                    # Try different possible column names
                    if f'STOCHd_{k_length}_{d_length}_3' in stoch.columns:
                        df['STOCH_D'] = stoch[f'STOCHd_{k_length}_{d_length}_3'] / 100.0
                    elif 'STOCHd' in stoch.columns:
                        df['STOCH_D'] = stoch['STOCHd'] / 100.0
                    else:
                        # Assume the second column is D
                        df['STOCH_D'] = stoch.iloc[:, 1] / 100.0
        
        # MACD (Moving Average Convergence Divergence)
        if config["indicators"].get("macd", {}).get("enabled", False):
            # Check if MACD components are already in the CSV
            if 'MACD' in df.columns and 'Signal' in df.columns and 'Histogram' in df.columns:
                # Use existing MACD data, normalize if needed
                max_val = max(abs(df['MACD'].max()), abs(df['MACD'].min()), 
                              abs(df['Signal'].max()), abs(df['Signal'].min()),
                              abs(df['Histogram'].max()), abs(df['Histogram'].min()))
                
                if max_val > 1:
                    df['MACD'] = df['MACD'] / max_val
                    df['MACD_SIGNAL'] = df['Signal'] / max_val
                    df['MACD_HIST'] = df['Histogram'] / max_val
                else:
                    df['MACD_SIGNAL'] = df['Signal']
                    df['MACD_HIST'] = df['Histogram']
            else:
                # Calculate MACD
                macd = ta.macd(df[close_col], 
                               fast=config["indicators"]["macd"]["fast"], 
                               slow=config["indicators"]["macd"]["slow"], 
                               signal=config["indicators"]["macd"]["signal"])
                
                # Check the actual column names in the result
                logger.info(f"MACD result columns: {macd.columns}")
                
                # Extract MACD, Signal, and Histogram values
                fast = config["indicators"]["macd"]["fast"]
                slow = config["indicators"]["macd"]["slow"]
                signal = config["indicators"]["macd"]["signal"]
                
                if f'MACD_{fast}_{slow}_{signal}' in macd.columns:
                    df['MACD'] = macd[f'MACD_{fast}_{slow}_{signal}']
                    df['MACD_SIGNAL'] = macd[f'MACDs_{fast}_{slow}_{signal}']
                    df['MACD_HIST'] = macd[f'MACDh_{fast}_{slow}_{signal}']
                else:
                    # Assume the columns are in order: MACD, MACD Signal, MACD Histogram
                    df['MACD'] = macd.iloc[:, 0]
                    df['MACD_SIGNAL'] = macd.iloc[:, 1]
                    df['MACD_HIST'] = macd.iloc[:, 2]
                
                # Normalize MACD values
                max_val = max(abs(df['MACD'].max()), abs(df['MACD'].min()), 
                             abs(df['MACD_SIGNAL'].max()), abs(df['MACD_SIGNAL'].min()),
                             abs(df['MACD_HIST'].max()), abs(df['MACD_HIST'].min()))
                
                if max_val > 0:  # Avoid division by zero
                    df['MACD'] = df['MACD'] / max_val
                    df['MACD_SIGNAL'] = df['MACD_SIGNAL'] / max_val
                    df['MACD_HIST'] = df['MACD_HIST'] / max_val
        
        # ROC (Rate of Change)
        if config["indicators"].get("roc", {}).get("enabled", False):
            if 'ROC' in df.columns:
                # Use existing ROC data, normalize if needed
                max_abs = max(abs(df['ROC'].max()), abs(df['ROC'].min()))
                if max_abs > 1:
                    df['ROC'] = df['ROC'] / max_abs
            else:
                df['ROC'] = ta.roc(df[close_col], length=config["indicators"]["roc"]["length"])
                
                # Normalize ROC
                max_abs = max(abs(df['ROC'].max()), abs(df['ROC'].min()))
                if max_abs > 0:  # Avoid division by zero
                    df['ROC'] = df['ROC'] / max_abs
        
        # Williams %R
        if config["indicators"].get("williams_r", {}).get("enabled", False):
            if 'WILLIAMS_R' in df.columns:
                # Use existing Williams %R data, normalize if needed
                if df['WILLIAMS_R'].min() < -1:
                    df['WILLIAMS_R'] = df['WILLIAMS_R'] / 100.0
            else:
                df['WILLIAMS_R'] = ta.willr(df[high_col], df[low_col], df[close_col], 
                                           length=config["indicators"]["williams_r"]["length"]) / 100.0
        
        # SMA (Simple Moving Average)
        if config["indicators"].get("sma", {}).get("enabled", False):
            sma_length = config["indicators"]["sma"]["length"]
            if 'SMA' in df.columns:
                # Use existing SMA data
                df['SMA_NORM'] = (df[close_col] - df['SMA']) / df[close_col]
            else:
                df['SMA'] = ta.sma(df[close_col], length=sma_length)
                # Normalize SMA relative to close price
                df['SMA_NORM'] = (df[close_col] - df['SMA']) / df[close_col]
        
        # EMA (Exponential Moving Average)
        if config["indicators"].get("ema", {}).get("enabled", False):
            ema_length = config["indicators"]["ema"]["length"]
            if 'EMA' in df.columns:
                # Use existing EMA data
                df['EMA_NORM'] = (df[close_col] - df['EMA']) / df[close_col]
            else:
                df['EMA'] = ta.ema(df[close_col], length=ema_length)
                # Normalize EMA relative to close price
                df['EMA_NORM'] = (df[close_col] - df['EMA']) / df[close_col]
        
        # Disparity Index
        if config["indicators"].get("disparity", {}).get("enabled", False):
            if 'DISPARITY' in df.columns:
                # Use existing disparity data
                pass
            else:
                if 'SMA' not in df.columns:
                    sma_length = config["indicators"].get("sma", {}).get("length", 20)
                    df['SMA'] = ta.sma(df[close_col], length=sma_length)
                df['DISPARITY'] = ((df[close_col] / df['SMA']) - 1)
        
        # ATR (Average True Range)
        if config["indicators"].get("atr", {}).get("enabled", False):
            if 'ATR' in df.columns:
                # Use existing ATR data, normalize if needed
                df['ATR'] = df['ATR'] / df[close_col]
            else:
                atr_length = config["indicators"]["atr"]["length"]
                atr = ta.atr(df[high_col], df[low_col], df[close_col], length=atr_length)
                
                # Check if atr is a Series or DataFrame
                if isinstance(atr, pd.Series):
                    df['ATR'] = atr / df[close_col]
                else:
                    # If it's a DataFrame, get the first column
                    logger.info(f"ATR result columns: {atr.columns}")
                    if f'ATR_{atr_length}' in atr.columns:
                        df['ATR'] = atr[f'ATR_{atr_length}'] / df[close_col]
                    else:
                        df['ATR'] = atr.iloc[:, 0] / df[close_col]
        
        # OBV (On-Balance Volume)
        if config["indicators"].get("obv", {}).get("enabled", False):
            if 'OBV' in df.columns:
                # Use existing OBV data, normalize
                max_obv = df['OBV'].max()
                min_obv = df['OBV'].min()
                if max_obv != min_obv:
                    df['OBV_NORM'] = (df['OBV'] - min_obv) / (max_obv - min_obv)
                else:
                    df['OBV_NORM'] = 0
            else:
                df['OBV'] = ta.obv(df[close_col], df[volume_col])
                # Normalize OBV
                max_obv = df['OBV'].max()
                min_obv = df['OBV'].min()
                if max_obv != min_obv:
                    df['OBV_NORM'] = (df['OBV'] - min_obv) / (max_obv - min_obv)
                else:
                    df['OBV_NORM'] = 0
        
        # CMF (Chaikin Money Flow)
        if config["indicators"].get("cmf", {}).get("enabled", False):
            if 'CMF' in df.columns:
                # Use existing CMF data
                pass
            else:
                df['CMF'] = ta.cmf(df[high_col], df[low_col], df[close_col], df[volume_col], 
                                   length=config["indicators"]["cmf"]["length"])
        
        # PSAR (Parabolic SAR)
        if config["indicators"].get("psar", {}).get("enabled", False):
            if 'PSAR' in df.columns:
                # Use existing PSAR data
                df['PSAR_NORM'] = (df[close_col] - df['PSAR']) / df[close_col]
                df['PSAR_DIR'] = np.where(df[close_col] > df['PSAR'], 1, -1)
            else:
                af = config["indicators"]["psar"]["af"]
                max_af = config["indicators"]["psar"]["max_af"]
                psar = ta.psar(df[high_col], df[low_col], af=af, max_af=max_af)
                
                # Check the actual column names in the result
                logger.info(f"PSAR result columns: {psar.columns}")
                
                # Try different possible column names
                psar_col = f'PSARl_{af}_{max_af}'
                if psar_col not in psar.columns:
                    # Try alternative naming
                    if 'PSARl' in psar.columns:
                        psar_col = 'PSARl'
                    else:
                        # Assume first column is PSAR
                        df['PSAR'] = psar.iloc[:, 0]
                else:
                    df['PSAR'] = psar[psar_col]
                    
                df['PSAR_NORM'] = (df[close_col] - df['PSAR']) / df[close_col]
                # Create a direction indicator (1 if price above PSAR, -1 if below)
                df['PSAR_DIR'] = np.where(df[close_col] > df['PSAR'], 1, -1)
        
        # Volume indicator
        if config["indicators"].get("volume", {}).get("enabled", False):
            if 'VOLUME_NORM' in df.columns:
                # Use existing volume data
                pass
            else:
                # Calculate volume moving average
                ma_length = config["indicators"]["volume"].get("ma_length", 20)
                volume_ma = ta.sma(df[volume_col], length=ma_length)
                
                # Normalize volume using moving average
                df['VOLUME_NORM'] = (df[volume_col] / volume_ma) - 1
        
        # Fill NaN values with appropriate defaults
        df = df.fillna(0)
        
        # Create an initial 'position' column (copy of trend_direction)
        df['position'] = df['trend_direction']
        
        # Create a list of all columns that will be used by the model
        model_columns = ['Close', 'close_norm', 'trend_direction', 'position']
        
        # Add all technical indicators that were enabled and calculated
        indicators_to_normalize = []
        for indicator in ['RSI', 'CCI', 'ADX', 'ADX_POS', 'ADX_NEG', 'STOCH_K', 'STOCH_D', 
                         'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC', 'WILLIAMS_R', 
                         'SMA_NORM', 'EMA_NORM', 'DISPARITY', 'ATR', 'OBV_NORM', 
                         'CMF', 'PSAR_NORM', 'PSAR_DIR', 'VOLUME_NORM']:
            if indicator in df.columns:
                model_columns.append(indicator)
                
                # Skip indicators that are already guaranteed to be within bounds
                if indicator not in ['trend_direction', 'RSI', 'PSAR_DIR']:
                    indicators_to_normalize.append(indicator)
                
                # Fill NaN values in indicators with appropriate defaults
                if indicator in ['RSI', 'STOCH_K', 'STOCH_D']:
                    # These indicators are typically in the range [0, 1]
                    df[indicator] = df[indicator].fillna(0.5)
                elif indicator in ['CCI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC', 'WILLIAMS_R', 
                                  'SMA_NORM', 'EMA_NORM', 'DISPARITY', 'CMF', 'VOLUME_NORM']:
                    # These indicators are typically centered around 0
                    df[indicator] = df[indicator].fillna(0.0)
                elif indicator in ['ADX', 'ADX_POS', 'ADX_NEG', 'ATR', 'OBV_NORM']:
                    # These indicators are typically positive
                    df[indicator] = df[indicator].fillna(0.0)
                elif indicator in ['PSAR_DIR']:
                    # Direction indicators are typically -1 or 1
                    df[indicator] = df[indicator].fillna(0)
                elif indicator in ['PSAR_NORM']:
                    # PSAR normalized is typically small
                    df[indicator] = df[indicator].fillna(0.0)
        
        # Fill any remaining NaN values in trend_direction and position
        df['trend_direction'] = df['trend_direction'].fillna(0)
        df['position'] = df['position'].fillna(0)
        
        # Two-stage normalization for technical indicators using only training data statistics
        logger.info("Performing two-stage normalization for technical indicators using only training data statistics")
        
        # Calculate split index for normalization
        split_idx = int(len(df) * train_ratio) if train_ratio > 0 else len(df)
        
        for indicator in indicators_to_normalize:
            # Get min and max values from training data only
            train_indicator_min = df.iloc[:split_idx][indicator].min()
            train_indicator_max = df.iloc[:split_idx][indicator].max()
            
            # Check if values need normalization
            if train_indicator_min < -1.0 or train_indicator_max > 1.0 or (indicator in ['CCI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC']):
                # Normalize to [-1, 1] range using training data statistics
                if train_indicator_min != train_indicator_max:  # Avoid division by zero
                    df[indicator] = 2.0 * (df[indicator] - train_indicator_min) / (train_indicator_max - train_indicator_min) - 1.0
                else:
                    df[indicator] = 0.0  # If all values are the same, set to 0
                
                # Clip to ensure within bounds
                df[indicator] = np.clip(df[indicator], -1.0, 1.0)
                logger.info(f"Normalized {indicator} using training data statistics (min: {train_indicator_min}, max: {train_indicator_max})")
        
        return df
    except Exception as e:
        logger.error(f"Error processing technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return df

def get_data(symbol: str = "NQ=F",
                 period: str = "60d",
                 interval: str = "5m",
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.15,
                 use_yfinance: bool = True):
    """
    Load data, compute technical indicators, normalize prices,
    and split the data into training, validation and testing sets.

    Args:
        symbol (str): Asset ticker symbol.
        period (str): Historical period to download (e.g., '60d' for 60 days).
        interval (str): Data interval (e.g., '5m' for 5 minutes).
        train_ratio (float): Proportion of data to use for training (default 0.7).
        validation_ratio (float): Proportion of data to use for validation (default 0.15).
                                 Test ratio is calculated as 1 - train_ratio - validation_ratio.
        use_yfinance (bool): Whether to download fresh data from Yahoo Finance (default True).

    Returns:
        tuple: (train_df, validation_df, test_df) - DataFrames containing processed data.
    """
    try:
        # Option 1: Download data directly from Yahoo Finance
        if use_yfinance:
            logger.info(f"Downloading data directly from Yahoo Finance: {symbol}")
            df = download_data(symbol, period, interval)
            
            # Debug: print DataFrame structure
            logger.info(f"Downloaded data structure: columns={df.columns.tolist() if df is not None else None}")
            
            if df is None:
                logger.error("Failed to download data from Yahoo Finance.")
                # Try to fall back to CSV if available
                if os.path.exists('data/nq.csv'):
                    logger.info("Falling back to local CSV file: data/nq.csv")
                    df = pd.read_csv('data/nq.csv')
                else:
                    return None, None, None
        
        # Option 2: Load from local CSV if download is disabled
        elif os.path.exists('data/nq.csv'):
            logger.info("Loading data from local CSV file: data/nq.csv")
            # Read the CSV file, explicitly convert numeric columns
            df = pd.read_csv('data/nq.csv')
        else:
            logger.error("CSV file not found: data/nq.csv and use_yfinance is False")
            return None, None, None
            
        # Ensure we have required columns
        required_cols = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
            return None, None, None
            
        # Debug: print the first 5 rows before any conversion
        logger.info(f"First 5 rows before time conversion:\n{df.head()}")
        logger.info(f"Time column type: {df['time'].dtype}")
        
        # Remove any rows where numeric columns contain non-numeric data
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors='coerce').notna()]
                df[col] = pd.to_numeric(df[col])
        
        logger.info(f"Data loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Check if the DataFrame is empty
        if df.empty:
            logger.error("The loaded DataFrame is empty!")
            return None, None, None
        
        # Convert time column to datetime - handle different possible formats
        try:
            # Try different approaches to convert time to datetime depending on its current format
            if pd.api.types.is_numeric_dtype(df['time']):
                # If time is already numeric (timestamp), convert to datetime
                logger.info("Converting numeric timestamp to datetime")
                df['time'] = pd.to_datetime(df['time'], unit='s')
            else:
                # If time is string or already datetime
                logger.info("Converting string or datetime to datetime")
                df['time'] = pd.to_datetime(df['time'])
                
            # Set time as index
            df = df.set_index('time')
            logger.info("Successfully converted time column to datetime index")
        except Exception as e:
            logger.error(f"Error converting time column: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
        
        # Rename columns to match the expected format
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns={col: column_mapping[col] for col in column_mapping if col in df.columns})
        
        # Ensure all price columns are numeric
        essential_columns = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            essential_columns.append('Volume')
        
        # Use the helper function to ensure numeric data
        df = ensure_numeric(df, essential_columns)
        
        # Drop any duplicate columns and NaN values
        df = df.loc[:, ~df.columns.duplicated()]
        
        logger.info(f"After preprocessing. Shape: {df.shape}")
        logger.info(f"Columns after preprocessing: {df.columns.tolist()}")
        print(df.head())
        
        # Process technical indicators
        df = process_technical_indicators(df, train_ratio)

        # After all processing, check if we should filter to market hours only
        if config["data"].get("market_hours_only", False):
            logger.info("Filtering data to include only NYSE market hours")
            try:
                # Try to import again in case it wasn't available earlier
                from walk_forward import filter_market_hours
                df = filter_market_hours(df)
            except ImportError:
                logger.warning("Could not import filter_market_hours from walk_forward module. Using unfiltered data.")
        
        # Split data into training, validation, and testing sets
        if train_ratio <= 0:
            # If train_ratio is 0 or negative, return all data as train data and empty validation/test sets
            train_df = df.copy()
            validation_df = pd.DataFrame(columns=df.columns)
            test_df = pd.DataFrame(columns=df.columns)
            logger.info("Data loaded and processed. Train data: %d rows (all data), Validation data: 0 rows, Test data: 0 rows", 
                       len(train_df))
        else:
            # Normal splitting
            train_split_idx = int(len(df) * train_ratio)
            validation_split_idx = train_split_idx + int(len(df) * validation_ratio)
            
            train_df = df.iloc[:train_split_idx].copy()
            validation_df = df.iloc[train_split_idx:validation_split_idx].copy()
            test_df = df.iloc[validation_split_idx:].copy()
            
            logger.info("Data loaded and processed. Train data: %d rows, Validation data: %d rows, Test data: %d rows", 
                       len(train_df), len(validation_df), len(test_df))
                       
        return train_df, validation_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

if __name__ == "__main__":
    get_data()