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
import pytz

# Import indicators
from indicators.rsi import calculate_rsi
from indicators.sma import calculate_sma
from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr
from indicators.supertrend import calculate_supertrend
from indicators.stochastic import calculate_stochastic
from indicators.adx import calculate_adx
from indicators.cci import calculate_cci
from indicators.roc import calculate_roc
from indicators.williams_r import calculate_williams_r
from indicators.obv import calculate_obv
from indicators.cmf import calculate_cmf
from indicators.psar import calculate_psar
from indicators.vwap import calculate_vwap
from indicators.disparity import calculate_disparity
from indicators.volume import calculate_volume_indicator
from indicators.day_of_week import calculate_day_of_week
from indicators.minutes_since_open import calculate_minutes_since_open
from indicators.rrcf_anomaly import calculate_rrcf_anomaly
from indicators.z_score import calculate_zscore

# Import normalization functions
from normalization import get_standardized_column_names, normalize_data, scale_window

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
        raw_data.index.name = 'timestamp'  # Ensure the index has a name
        raw_filename = f'data/{symbol.replace("=", "_")}_raw.csv'
        raw_data.to_csv(raw_filename)
        logger.info(f"Saved raw data to {raw_filename}")
        
        # Create a processed dataframe with the index as the 'timestamp' column
        processed_data = pd.DataFrame()
        processed_data['timestamp'] = data.index.astype('int64') // 10**9  # Convert to Unix timestamp
        
        # Extract data using standard lowercase column names
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        for yf_col, our_col in column_mapping.items():
            if yf_col in data.columns:
                processed_data[our_col] = data[yf_col].values
        
        logger.info(f"Processed data columns: {processed_data.columns.tolist()}")
        logger.info(f"Sample of processed data:\n{processed_data.head()}")
        
        # Ensure we have the required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
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
        logger.info("========== Starting technical indicator processing ==========")
        logger.info(f"Input data shape: {df.shape}")
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for indicator processing: {missing_cols}. Available: {df.columns.tolist()}")
            return df
        
        # Fill NaN values in price columns
        for col in ['open', 'high', 'low', 'close']:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col} column. Filling with forward fill.")
                df[col] = df[col].fillna(method='ffill')
        
        # Calculate normalized close
        window = 100  # Use last 100 bars for min-max scaling
        # For each point, calculate min/max over the previous window periods
        rolling_min = df['close'].rolling(window=window, min_periods=1).min()
        rolling_max = df['close'].rolling(window=window, min_periods=1).max()
        
        # Avoid division by zero and ensure values are in [0, 1]
        df['close_norm'] = np.where(
            rolling_max > rolling_min,
            (df['close'] - rolling_min) / (rolling_max - rolling_min),
            0.5  # Default to middle value when there's no range
        )
        # Ensure NaN values are filled
        df['close_norm'] = df['close_norm'].fillna(0.5)
        
        # Add all the requested technical indicators using the individual modules
        
        # RSI (Relative Strength Index)
        if config["indicators"]["rsi"]["enabled"]:
            df = calculate_rsi(df, length=config["indicators"]["rsi"]["length"], 
                              target_col='RSI')
            
        # CCI (Commodity Channel Index)
        if config["indicators"].get("cci", {}).get("enabled", False):
            df = calculate_cci(df, length=config["indicators"]["cci"]["length"],
                              target_col='CCI')
        
        # ADX (Average Directional Index)
        if config["indicators"].get("adx", {}).get("enabled", False):
            df = calculate_adx(df, length=config["indicators"]["adx"]["length"],
                              adx_col='ADX')
            
            # Separately add DI+ and DI- columns if needed
            if config["indicators"].get("adx_pos", {}).get("enabled", False) or config["indicators"].get("adx_neg", {}).get("enabled", False):
                # Calculate ADX with pandas_ta directly to get DI+ and DI-
                adx_result = ta.adx(df['high'], df['low'], df['close'], length=config["indicators"]["adx"]["length"])
                
                # Add DI+ if enabled
                if config["indicators"].get("adx_pos", {}).get("enabled", False):
                    df['ADX_POS'] = adx_result[f'DMP_{config["indicators"]["adx"]["length"]}']
                    df['ADX_POS'] = df['ADX_POS'].fillna(0.0)
                
                # Add DI- if enabled
                if config["indicators"].get("adx_neg", {}).get("enabled", False):
                    df['ADX_NEG'] = adx_result[f'DMN_{config["indicators"]["adx"]["length"]}']
                    df['ADX_NEG'] = df['ADX_NEG'].fillna(0.0)
        
        # Stochastic Oscillator
        if (config["indicators"].get("stoch_k", {}).get("enabled", False) or 
            config["indicators"].get("stoch_d", {}).get("enabled", False)):
            
            k_period = config["indicators"].get("stoch_k", {}).get("length", 14)
            d_period = config["indicators"].get("stoch_d", {}).get("length", 3)
            smooth_k = 3
            
            df = calculate_stochastic(df, k_period=k_period, d_period=d_period, smooth_k=smooth_k,
                                     k_col='STOCH_K', d_col='STOCH_D')
        
        # MACD (Moving Average Convergence Divergence)
        if config["indicators"].get("macd", {}).get("enabled", False):
            df = calculate_macd(df, fast_period=config["indicators"]["macd"]["fast"],
                               slow_period=config["indicators"]["macd"]["slow"],
                               signal_period=config["indicators"]["macd"]["signal"],
                               macd_col='MACD', signal_col='MACD_SIGNAL', histogram_col='MACD_HIST')
        
        # ROC (Rate of Change)
        if config["indicators"].get("roc", {}).get("enabled", False):
            df = calculate_roc(df, length=config["indicators"]["roc"]["length"],
                              target_col='ROC')
        
        # Williams %R
        if config["indicators"].get("williams_r", {}).get("enabled", False):
            df = calculate_williams_r(df, length=config["indicators"]["williams_r"]["length"],
                                     target_col='WILLIAMS_R')
        
        # SMA (Simple Moving Average)
        if config["indicators"].get("sma", {}).get("enabled", False):
            df = calculate_sma(df, length=config["indicators"]["sma"]["length"],
                              target_col='SMA')
        
        # EMA (Exponential Moving Average)
        if config["indicators"].get("ema", {}).get("enabled", False):
            df = calculate_ema(df, length=config["indicators"]["ema"]["length"],
                              target_col='EMA')
        
        # Disparity Index
        if config["indicators"].get("disparity", {}).get("enabled", False):
            df = calculate_disparity(df, length=config["indicators"].get("disparity", {}).get("length", 20),
                                    target_col='DISPARITY')
        
        # ATR (Average True Range)
        if config["indicators"].get("atr", {}).get("enabled", False):
            df = calculate_atr(df, length=config["indicators"]["atr"]["length"],
                              target_col='ATR')
        
        # OBV (On-Balance Volume)
        if config["indicators"].get("obv", {}).get("enabled", False):
            df = calculate_obv(df,
                              target_col='OBV')
        
        # CMF (Chaikin Money Flow)
        if config["indicators"].get("cmf", {}).get("enabled", False):
            df = calculate_cmf(df, length=config["indicators"]["cmf"]["length"], 
                              target_col='CMF')
        
        # PSAR (Parabolic SAR)
        if config["indicators"].get("psar", {}).get("enabled", False):
            df = calculate_psar(df, af=config["indicators"]["psar"]["af"],
                               max_af=config["indicators"]["psar"]["max_af"], 
                               dir_col='PSAR_DIR')
        
        # Volume indicator
        if config["indicators"].get("volume", {}).get("enabled", False):
            df = calculate_volume_indicator(df,
                                          ma_length=config["indicators"]["volume"].get("ma_length", 20),
                                          target_col='VOLUME_NORM')
        
        # VWAP (Volume Weighted Average Price)
        if config["indicators"].get("vwap", {}).get("enabled", False):
            df = calculate_vwap(df, target_col='VWAP')
        
        # Calculate Supertrend indicator
        if 'supertrend' in config["indicators"] and config["indicators"]["supertrend"]["enabled"]:
            df = calculate_supertrend(df, 
                                     length=config["indicators"]["supertrend"]["length"],
                                     multiplier=config["indicators"]["supertrend"]["multiplier"],
                                     smooth_periods=config["indicators"]["supertrend"].get("smooth_periods", 3),
                                     lookback_periods=config["indicators"]["supertrend"].get("lookback_periods", 2),
                                     target_col='supertrend')
        
        # Add day of week indicator (needed for pattern detection)
        if config["indicators"].get("day_of_week", {}).get("enabled", True):
            df = calculate_day_of_week(df, dow_col='DOW', sin_col='DOW_SIN', cos_col='DOW_COS')
            
        # Add minutes since cash open indicator (9:30 AM ET)
        if config["indicators"].get("minutes_since_open", {}).get("enabled", False):
            df = calculate_minutes_since_open(df, sin_col='MSO_SIN', cos_col='MSO_COS')
        
        # Add RRCF anomaly detection indicator
        if config["indicators"].get("rrcf_anomaly", {}).get("enabled", False):
            rrcf_config = config["indicators"]["rrcf_anomaly"]
            df = calculate_rrcf_anomaly(
                df, 
                feature_cols=rrcf_config.get("feature_cols", ["close", "volume"]),
                window_size=rrcf_config.get("window_size", 100),
                num_trees=rrcf_config.get("num_trees", 40),
                tree_size=rrcf_config.get("tree_size", 256),
                target_col='RRCF_ANOMALY',
                random_seed=rrcf_config.get("random_seed", 42)
            )

        # Add Z-Score indicator
        if config["indicators"].get("z_score", {}).get("enabled", False):
            df = calculate_zscore(df, length=config["indicators"]["z_score"].get("length", 50),
                                target_col='ZScore')
        
        # Create an initial 'position' column (should start with no position)
        df['position'] = 0  # Initialize with no position
        
        # Initial list of model columns for observation space
        model_columns = ['close', 'close_norm', 'DOW_SIN', 'DOW_COS']
        
        # Add minutes since open indicators if present
        if 'MSO_SIN' in df.columns and 'MSO_COS' in df.columns:
            model_columns.extend(['MSO_SIN', 'MSO_COS'])
            
        # Add position for the environment
        model_columns.append('position')
        
        # List of indicators that need normalization
        indicators_to_normalize = []
        
        # Add supertrend if it exists
        if 'supertrend' in df.columns:
            model_columns.append('supertrend')
        
        # Add all technical indicators that were enabled and calculated
        for indicator in ['RSI', 'CCI', 'ADX', 'ADX_POS', 'ADX_NEG', 'STOCH_K', 'STOCH_D', 
                         'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC', 'WILLIAMS_R', 
                         'SMA', 'EMA', 'DISPARITY', 'ATR', 'OBV', 
                         'CMF', 'PSAR_DIR', 'VOLUME_NORM', 'VWAP_NORM', 'RRCF_ANOMALY', 'ZScore']:
            if indicator in df.columns:
                model_columns.append(indicator)
                
                # Skip indicators that are already guaranteed to be within bounds
                if indicator not in ['supertrend', 'RSI', 'PSAR_DIR', 'RRCF_ANOMALY']:
                    indicators_to_normalize.append(indicator)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        # Create the missing normalized columns needed by the model
        # SMA_NORM from SMA
        if 'SMA' in df.columns and 'SMA_NORM' not in df.columns:
            df['SMA_NORM'] = df['SMA']
            model_columns.append('SMA_NORM')
            
        # EMA_NORM from EMA
        if 'EMA' in df.columns and 'EMA_NORM' not in df.columns:
            df['EMA_NORM'] = df['EMA']
            model_columns.append('EMA_NORM')
            
        # VOLUME_MA from VOLUME_NORM
        if 'VOLUME_NORM' in df.columns and 'VOLUME_MA' not in df.columns:
            df['VOLUME_MA'] = df['VOLUME_NORM']
            model_columns.append('VOLUME_MA')
        
        # Two-stage normalization for technical indicators using only training data statistics
        logger.info("Calculating technical indicators completed. Normalization will be performed per window to avoid data leakage.")
        
        return df
    except Exception as e:
        logger.error(f"Error processing technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return df

def get_data(symbol: str = "NQ=F",
             period: str = "60d",
             interval: str = "5m",
             train_ratio: float = config["data"]["train_ratio"],
             validation_ratio: float = config["data"]["validation_ratio"],
             use_yfinance: bool = False,
             normalize_and_save: bool = False):
    """
    Load data, compute technical indicators, normalize prices,
    and split the data into training, validation and testing sets.

    Args:
        symbol (str): Asset ticker symbol (not used when loading from CSV).
        period (str): Historical period to download (not used when loading from CSV).
        interval (str): Data interval (not used when loading from CSV).
        train_ratio (float): Proportion of data to use for training (default from config).
        validation_ratio (float): Proportion of data to use for validation (default from config).
                                 Test ratio is calculated as 1 - train_ratio - validation_ratio.
        use_yfinance (bool): Whether to download fresh data from Yahoo Finance (default False).
        normalize_and_save (bool): Whether to normalize data and save normalized files (default False).

    Returns:
        tuple: (train_df, validation_df, test_df) - DataFrames containing processed data.
    """
    try:
        logger.info("==================== STARTING DATA PROCESSING ====================")
        logger.info(f"Parameters: symbol={symbol}, period={period}, interval={interval}")
        logger.info(f"Data split: train={train_ratio}, validation={validation_ratio}, test={1-train_ratio-validation_ratio}")
        
        # Option 1: Download data directly from Yahoo Finance
        if use_yfinance:
            logger.info(f"Step 1: Downloading data directly from Yahoo Finance: {symbol}")
            df = download_data(symbol, period, interval)
            
            # Debug: print DataFrame structure
            logger.info(f"Downloaded data structure: columns={df.columns.tolist() if df is not None else None}")
            
            if df is None:
                logger.error("Failed to download data from Yahoo Finance.")
                # Try to fall back to CSV if available
                if os.path.exists('data/NQ_2024_unix.csv'):
                    logger.info("Falling back to local CSV file: data/NQ_2024_unix.csv")
                    df = pd.read_csv('data/NQ_2024_unix.csv')
                else:
                    return None, None, None
        
        # Option 2: Load from local CSV if download is disabled
        elif os.path.exists('data/NQ_2024_unix.csv'):
            logger.info("Step 1: Loading data from local CSV file: data/NQ_2024_unix.csv")
            # Read the CSV file, explicitly convert numeric columns
            df = pd.read_csv('data/NQ_2024_unix.csv')
                
        else:
            logger.error("CSV file not found: data/NQ_2024_unix.csv and use_yfinance is False")
            return None, None, None
            
        # Ensure we have required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
            return None, None, None
            
        # Debug: print the first 5 rows before any conversion
        logger.info(f"First 5 rows before time conversion:\n{df.head()}")
        logger.info(f"Timestamp column type: {df['timestamp'].dtype}")
        
        # Remove any rows where numeric columns contain non-numeric data
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors='coerce').notna()]
                df[col] = pd.to_numeric(df[col])
        
        logger.info(f"Data loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Check if the DataFrame is empty
        if df.empty:
            logger.error("The loaded DataFrame is empty!")
            return None, None, None
        
        # Convert timestamp column to datetime
        try:
            # Try different approaches to convert timestamp to datetime depending on its current format
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                # If timestamp is already numeric, convert to datetime
                logger.info("Converting numeric timestamp to datetime")
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                # If timestamp is string or already datetime
                logger.info("Converting string or datetime to datetime")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Set timestamp as index
            df = df.set_index('timestamp')
            logger.info("Successfully converted timestamp column to datetime index")
        except Exception as e:
            logger.error(f"Error converting timestamp column: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
        
        # Ensure all price columns are numeric
        essential_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Use the helper function to ensure numeric data
        df = ensure_numeric(df, essential_columns)
        
        # Drop any duplicate columns and NaN values
        df = df.loc[:, ~df.columns.duplicated()]
        
        logger.info(f"After preprocessing. Shape: {df.shape}")
        logger.info(f"Columns after preprocessing: {df.columns.tolist()}")
        print(df.head())
        
        # Process technical indicators
        logger.info("Step 4: Processing technical indicators")
        df = process_technical_indicators(df, train_ratio)
        logger.info(f"Technical indicators processed. Available columns: {df.columns.tolist()}")

        # After all processing, check if we should filter to market hours only
        if config["data"].get("market_hours_only", False):
            logger.info("Step 5: Filtering data to include only NYSE market hours")
            try:
                # Try to import again in case it wasn't available earlier
                from walk_forward import filter_market_hours
                df = filter_market_hours(df)
            except ImportError:
                logger.warning("Could not import filter_market_hours from walk_forward module. Using unfiltered data.")
        
        # Trim the first 30 rows which may contain incomplete indicator data
        logger.info("Step 6: Trimming first 30 rows with incomplete indicator initialization")
        df = df.iloc[35:].copy()
        logger.info(f"After trimming, data shape: {df.shape}")
        
        # Split the data
        logger.info("Step 7: Splitting data")
        train_split_idx = int(len(df) * train_ratio)
        validation_split_idx = train_split_idx + int(len(df) * validation_ratio)
        
        train_df = df.iloc[:train_split_idx].copy()
        validation_df = df.iloc[train_split_idx:validation_split_idx].copy()
        test_df = df.iloc[validation_split_idx:].copy()
        
        logger.info("Data loaded and processed. Train data: %d rows, Validation data: %d rows, Test data: %d rows", 
                   len(train_df), len(validation_df), len(test_df))
                   
        # Normalize data if requested
        if normalize_and_save:
            logger.info("Step 8: Normalizing data and saving normalized files")
            try:
                # Use functions from normalization.py
                # Get columns to normalize
                cols_to_normalize = get_standardized_column_names(df)
                logger.info(f"Normalizing {len(cols_to_normalize)} columns using MinMaxScaler")
                
                # Use scale_window function to normalize data
                # This uses MinMaxScaler with default range (-1, 1)
                scaler, train_df_norm, validation_df_norm, test_df_norm = scale_window(
                    train_data=train_df, 
                    val_data=validation_df, 
                    test_data=test_df, 
                    cols_to_scale=cols_to_normalize,
                    feature_range=(-1, 1)
                )
                
                # Save normalized DataFrames
                train_df_norm.to_csv('data/train_df_normalized.csv')
                validation_df_norm.to_csv('data/validation_df_normalized.csv')
                test_df_norm.to_csv('data/test_df_normalized.csv')
                
                # Save the scaler
                if scaler is not None:
                    os.makedirs('data/models', exist_ok=True)
                    import pickle
                    with open('data/models/feature_scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    logger.info("Saved feature scaler to data/models/feature_scaler.pkl")
                
                logger.info("Normalized data saved to data/train_df_normalized.csv, data/validation_df_normalized.csv, and data/test_df_normalized.csv")
            except Exception as e:
                logger.error(f"Error normalizing data: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("==================== DATA PROCESSING COMPLETE ====================")
        
        # Save DataFrames to CSV files
        os.makedirs('data', exist_ok=True)
        logger.info("Saving DataFrames to CSV files")
        
        if train_df is not None:
            train_df.to_csv('data/train_df.csv')
            logger.info("Saved train_df to data/train_df.csv")
            
        if validation_df is not None:
            validation_df.to_csv('data/validation_df.csv')
            logger.info("Saved validation_df to data/validation_df.csv")
            
        if test_df is not None:
            test_df.to_csv('data/test_df.csv')
            logger.info("Saved test_df to data/test_df.csv")
        
        return train_df, validation_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

if __name__ == "__main__":
    get_data(use_yfinance=False, normalize_and_save=True)