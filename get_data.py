import yfinance as yf
import pandas as pd
import numpy as np

# Monkey patch: set np.NaN to np.nan if it's missing
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NEW: import YAML configuration
from config import config

# Import filter_market_hours from walk_forward module
# We use try/except to handle circular import issue
try:
    from walk_forward import filter_market_hours
except ImportError:
    # Define a basic version if walk_forward isn't importable
    def filter_market_hours(data):
        logger.warning("Could not import filter_market_hours from walk_forward module. Using unfiltered data.")
        return data

def get_data(symbol: str = "NQ=F",
                 period: str = "60d",
                 interval: str = "5m",
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.15):
    """
    Load data from local CSV file, compute technical indicators, normalize prices,
    and split the data into training, validation and testing sets.

    Args:
        symbol (str): Asset ticker symbol (not used when loading from CSV).
        period (str): Historical period (not used when loading from CSV).
        interval (str): Data interval (not used when loading from CSV).
        train_ratio (float): Proportion of data to use for training (default 0.7).
        validation_ratio (float): Proportion of data to use for validation (default 0.15).
                                 Test ratio is calculated as 1 - train_ratio - validation_ratio.

    Returns:
        tuple: (train_df, validation_df, test_df) - DataFrames containing processed data.
    """
    logger.info("Loading data from local CSV file: data/nq.csv")
    
    try:
        # Check if file exists
        if not os.path.exists('data/nq.csv'):
            logger.error("CSV file not found: data/nq.csv")
            return None, None, None
            
        # Read the CSV file with error handling
        try:
            df = pd.read_csv('data/nq.csv')
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            return None, None, None
        except pd.errors.ParserError:
            logger.error("Error parsing CSV file - it may be malformed")
            return None, None, None
        
        logger.info(f"CSV file loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Check if the DataFrame is empty
        if df.empty:
            logger.error("The loaded DataFrame is empty!")
            return None, None, None
            
        # Print the first few rows to debug
        logger.info(f"First 5 rows of data:\n{df.head()}")
        
        # Convert time column to datetime and set as index
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        
        # Rename columns to match the expected format
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'Volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Drop any duplicate columns and NaN values
        df = df.loc[:, ~df.columns.duplicated()]
        # Instead of dropping all rows with NaN values, we'll only drop rows with NaN in essential columns
        essential_columns = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            essential_columns.append('Volume')
        
        # Only drop rows where essential columns have NaN values
        df.dropna(subset=essential_columns, inplace=True)
        
        logger.info(f"After preprocessing. Shape: {df.shape}")
        print(df.head())
        
        # Compute Supertrend indicator if enabled
        if config["indicators"]["supertrend"]["enabled"]:
            try:
                supertrend_df = ta.supertrend(
                    df['High'], df['Low'], df['Close'],
                    length=config["indicators"]["supertrend"]["length"],
                    multiplier=config["indicators"]["supertrend"]["multiplier"]
                )
                
                if supertrend_df is not None:
                    supertrend_df.columns = ["Supertrend", "TrendDirection", "SupertrendLong", "SupertrendShort"]
                    df = df.join(supertrend_df)
                    # Drop extra Supertrend columns
                    df.drop(columns=["SupertrendLong", "SupertrendShort", "Supertrend"], inplace=True)
                    # Convert trend direction to integer and rename for consistency
                    df['TrendDirection'] = df['TrendDirection'].astype(int)
                    df.rename(columns={'TrendDirection': 'trend_direction'}, inplace=True)
                else:
                    logger.warning("Supertrend calculation returned None, using Up Trend/Down Trend columns instead")
                    # Check if 'Up Trend' column exists in the CSV
                    if 'Up Trend' in df.columns:
                        # Use the existing trend data from CSV
                        df['trend_direction'] = np.where(df['Up Trend'].notna(), 1, 
                                                      np.where(df['Down Trend'].notna(), -1, 0))
                    else:
                        df['trend_direction'] = 0
            except Exception as e:
                logger.error(f"Error calculating Supertrend: {e}")
                # Check if 'Up Trend' column exists in the CSV
                if 'Up Trend' in df.columns:
                    # Use the existing trend data from CSV
                    df['trend_direction'] = np.where(df['Up Trend'].notna(), 1, 
                                                  np.where(df['Down Trend'].notna(), -1, 0))
                else:
                    df['trend_direction'] = 0
        else:
            # Check if 'Up Trend' column exists in the CSV
            if 'Up Trend' in df.columns:
                # Use the existing trend data from CSV
                df['trend_direction'] = np.where(df['Up Trend'].notna(), 1, 
                                               np.where(df['Down Trend'].notna(), -1, 0))
            else:
                df['trend_direction'] = 0

        # Add all the requested technical indicators
        
        # RSI (Relative Strength Index)
        if config["indicators"]["rsi"]["enabled"]:
            # Check if RSI data is already in the CSV
            if 'RSI' in df.columns:
                # Normalize RSI to [0, 1] range if needed
                if df['RSI'].max() > 1:
                    df['RSI'] = df['RSI'] / 100.0
            else:
                df['RSI'] = ta.rsi(df['Close'], length=config["indicators"]["rsi"]["length"]) / 100.0
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
                df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], 
                                   length=config["indicators"]["cci"]["length"]) / 100.0
        
        # ADX (Average Directional Index)
        if config["indicators"].get("adx", {}).get("enabled", False):
            if 'ADX' in df.columns:
                # Normalize ADX if needed
                if df['ADX'].max() > 1:
                    df['ADX'] = df['ADX'] / 100.0
            else:
                adx_length = config["indicators"]["adx"]["length"]
                adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=adx_length)
                
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
                        adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=adx_length)
                    
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
                        adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=adx_length)
                    
                    if f'DMN_{adx_length}' in adx_result.columns:
                        df['ADX_NEG'] = adx_result[f'DMN_{adx_length}'] / 100.0
                    else:
                        # Assume the third column is DMN
                        df['ADX_NEG'] = adx_result.iloc[:, 2] / 100.0
        
        # Stochastic Oscillator
        if (config["indicators"].get("stoch_k", {}).get("enabled", False) or 
            config["indicators"].get("stoch_d", {}).get("enabled", False)):
            
            # Check if K and D are already in the CSV
            if 'K' in df.columns and config["indicators"].get("stoch_k", {}).get("enabled", False):
                df['STOCH_K'] = df['K'] / 100.0 if df['K'].max() > 1 else df['K']
            
            if 'D' in df.columns and config["indicators"].get("stoch_d", {}).get("enabled", False):
                df['STOCH_D'] = df['D'] / 100.0 if df['D'].max() > 1 else df['D']
                
            # If not in CSV, calculate them
            if ('K' not in df.columns or 'D' not in df.columns):
                k_length = config["indicators"]["stoch_k"]["length"]
                d_length = config["indicators"]["stoch_d"]["length"]
                stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=k_length, d=d_length)
                
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
                fast = config["indicators"]["macd"]["fast"]
                slow = config["indicators"]["macd"]["slow"]
                signal = config["indicators"]["macd"]["signal"]
                macd = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
                
                # Check the actual column names in the result
                logger.info(f"MACD result columns: {macd.columns}")
                
                # Try different possible column names
                macd_col = f'MACD_{fast}_{slow}_{signal}'
                signal_col = f'MACDs_{fast}_{slow}_{signal}'
                hist_col = f'MACDh_{fast}_{slow}_{signal}'
                
                if macd_col not in macd.columns:
                    # Try alternative naming
                    if 'MACD' in macd.columns:
                        macd_col = 'MACD'
                    else:
                        # Assume first column is MACD
                        macd_values = macd.iloc[:, 0]
                        df['MACD'] = macd_values / max(abs(macd_values.max()), abs(macd_values.min())) if macd_values.max() != macd_values.min() else 0
                        
                    if 'MACDs' in macd.columns:
                        signal_col = 'MACDs'
                    else:
                        # Assume second column is Signal
                        signal_values = macd.iloc[:, 1]
                        df['MACD_SIGNAL'] = signal_values / max(abs(signal_values.max()), abs(signal_values.min())) if signal_values.max() != signal_values.min() else 0
                        
                    if 'MACDh' in macd.columns:
                        hist_col = 'MACDh'
                    else:
                        # Assume third column is Histogram
                        hist_values = macd.iloc[:, 2]
                        df['MACD_HIST'] = hist_values / max(abs(hist_values.max()), abs(hist_values.min())) if hist_values.max() != hist_values.min() else 0
                else:
                    # Use the expected column names
                    max_val = max(abs(macd[macd_col].max()), abs(macd[macd_col].min()))
                    if max_val > 0:
                        df['MACD'] = macd[macd_col] / max_val
                        df['MACD_SIGNAL'] = macd[signal_col] / max_val
                        df['MACD_HIST'] = macd[hist_col] / max_val
                    else:
                        df['MACD'] = 0
                        df['MACD_SIGNAL'] = 0
                        df['MACD_HIST'] = 0
        
        # ROC (Rate of Change)
        if config["indicators"].get("roc", {}).get("enabled", False):
            if 'ROC' in df.columns:
                # Normalize if needed
                if df['ROC'].max() > 1 or df['ROC'].min() < -1:
                    max_abs = max(abs(df['ROC'].max()), abs(df['ROC'].min()))
                    df['ROC'] = df['ROC'] / max_abs
            else:
                df['ROC'] = ta.roc(df['Close'], length=config["indicators"]["roc"]["length"]) / 100.0
        
        # Williams %R
        if config["indicators"].get("williams_r", {}).get("enabled", False):
            if 'WILLIAMS_R' in df.columns:
                # Normalize if needed
                if df['WILLIAMS_R'].min() < -1:
                    df['WILLIAMS_R'] = df['WILLIAMS_R'] / 100.0
            else:
                df['WILLIAMS_R'] = ta.willr(df['High'], df['Low'], df['Close'], 
                                            length=config["indicators"]["williams_r"]["length"]) / 100.0
        
        # SMA (Simple Moving Average)
        if config["indicators"].get("sma", {}).get("enabled", False):
            sma_length = config["indicators"]["sma"]["length"]
            if 'SMA' in df.columns:
                # Use existing SMA data
                df['SMA_NORM'] = (df['Close'] - df['SMA']) / df['Close']
            else:
                df['SMA'] = ta.sma(df['Close'], length=sma_length)
                # Normalize SMA relative to close price
                df['SMA_NORM'] = (df['Close'] - df['SMA']) / df['Close']
        
        # EMA (Exponential Moving Average)
        if config["indicators"].get("ema", {}).get("enabled", False):
            ema_length = config["indicators"]["ema"]["length"]
            if 'EMA' in df.columns:
                # Use existing EMA data
                df['EMA_NORM'] = (df['Close'] - df['EMA']) / df['Close']
            else:
                df['EMA'] = ta.ema(df['Close'], length=ema_length)
                # Normalize EMA relative to close price
                df['EMA_NORM'] = (df['Close'] - df['EMA']) / df['Close']
        
        # Disparity Index
        if config["indicators"].get("disparity", {}).get("enabled", False):
            if 'DISPARITY' in df.columns:
                # Use existing disparity data
                pass
            else:
                if 'SMA' not in df.columns:
                    sma_length = config["indicators"].get("sma", {}).get("length", 20)
                    df['SMA'] = ta.sma(df['Close'], length=sma_length)
                df['DISPARITY'] = ((df['Close'] / df['SMA']) - 1)
        
        # ATR (Average True Range)
        if config["indicators"].get("atr", {}).get("enabled", False):
            if 'ATR' in df.columns:
                # Use existing ATR data, normalize if needed
                df['ATR'] = df['ATR'] / df['Close']
            else:
                atr_length = config["indicators"]["atr"]["length"]
                atr = ta.atr(df['High'], df['Low'], df['Close'], length=atr_length)
                
                # Check if atr is a Series or DataFrame
                if isinstance(atr, pd.Series):
                    df['ATR'] = atr / df['Close']
                else:
                    # If it's a DataFrame, get the first column
                    logger.info(f"ATR result columns: {atr.columns}")
                    if f'ATR_{atr_length}' in atr.columns:
                        df['ATR'] = atr[f'ATR_{atr_length}'] / df['Close']
                    else:
                        df['ATR'] = atr.iloc[:, 0] / df['Close']
        
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
                df['OBV'] = ta.obv(df['Close'], df['Volume'])
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
                df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], 
                                   length=config["indicators"]["cmf"]["length"])
        
        # PSAR (Parabolic SAR)
        if config["indicators"].get("psar", {}).get("enabled", False):
            if 'PSAR' in df.columns:
                # Use existing PSAR data
                df['PSAR_NORM'] = (df['Close'] - df['PSAR']) / df['Close']
                df['PSAR_DIR'] = np.where(df['Close'] > df['PSAR'], 1, -1)
            else:
                af = config["indicators"]["psar"]["af"]
                max_af = config["indicators"]["psar"]["max_af"]
                psar = ta.psar(df['High'], df['Low'], af=af, max_af=max_af)
                
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
                    
                df['PSAR_NORM'] = (df['Close'] - df['PSAR']) / df['Close']
                # Create a direction indicator (1 if price above PSAR, -1 if below)
                df['PSAR_DIR'] = np.where(df['Close'] > df['PSAR'], 1, -1)

        # Only drop rows with NaN values in essential columns
        essential_columns = ['Close', 'close_norm']
        logger.info(f"Shape before dropping NaN in essential columns: {df.shape}")
        
        # Check if we have valid Close values before trying to normalize
        if df['Close'].notna().any():
            # Fill NaN values in Close with forward fill then backward fill
            df['Close'] = df['Close'].ffill().bfill()
            
            # Normalize Close price to the range [0,1] using all available data if train data is empty
            if train_ratio <= 0:
                close_min = df['Close'].min()
                close_max = df['Close'].max()
                df['close_norm'] = (df['Close'] - close_min) / (close_max - close_min)
            else:
                # Calculate split index
                split_idx = int(len(df) * train_ratio)
                if split_idx > 0:
                    train_close_min = df.iloc[:split_idx]['Close'].min()
                    train_close_max = df.iloc[:split_idx]['Close'].max()
                    # Avoid division by zero
                    if train_close_max > train_close_min:
                        df['close_norm'] = (df['Close'] - train_close_min) / (train_close_max - train_close_min)
                    else:
                        df['close_norm'] = 0.5  # Default if all values are the same
                else:
                    # If train data is empty, normalize using all data
                    close_min = df['Close'].min()
                    close_max = df['Close'].max()
                    df['close_norm'] = (df['Close'] - close_min) / (close_max - close_min)
        else:
            logger.error("No valid Close prices found in the data")
            return None, None, None
        
        # Check for remaining NaN values in essential columns and log
        nan_count = df[essential_columns].isna().sum()
        if nan_count.sum() > 0:
            logger.warning(f"NaN values in essential columns after filling: {nan_count}")
            # Fill any remaining NaNs with 0
            df[essential_columns] = df[essential_columns].fillna(0)
        
        logger.info(f"Shape after handling NaN values: {df.shape}")

        # Create an initial 'position' column (copy of trend_direction)
        df['position'] = df['trend_direction']
        
        # Create a list of all columns that will be used by the model
        model_columns = ['Close', 'close_norm', 'trend_direction', 'position']
        
        # Add all technical indicators that were enabled and calculated
        indicators_to_normalize = []
        for indicator in ['RSI', 'CCI', 'ADX', 'ADX_POS', 'ADX_NEG', 'STOCH_K', 'STOCH_D', 
                         'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC', 'WILLIAMS_R', 
                         'SMA_NORM', 'EMA_NORM', 'DISPARITY', 'ATR', 'OBV_NORM', 
                         'CMF', 'PSAR_NORM', 'PSAR_DIR']:
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
                                  'SMA_NORM', 'EMA_NORM', 'DISPARITY', 'CMF']:
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