#!/usr/bin/env python3
"""
Script to process and normalize all synthetic episode files for pretraining.

This script processes all CSV files in the ./synthetic directory, calculates technical
indicators, and normalizes them for use as episodes in pretraining. Each episode is 
processed and normalized independently to prevent data leakage between episodes. 
Files are processed in parallel for better performance.

Usage:
    python normalize_synthetic_episodes.py [--output-dir OUTPUT_DIR] [--no-sigmoid] [--no-scalers] [--n-jobs N_JOBS]
"""

import argparse
import logging
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

# Import technical indicator calculation functions
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
from normalization import normalize_data, get_standardized_column_names

# Import configuration
from config import config

# Monkey patch for pandas_ta compatibility
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta

def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('synthetic_normalization.log')
        ]
    )

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
                logging.warning(f"Error converting {col} to numeric: {e}")
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

def process_technical_indicators_for_episode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and compute technical indicators for a single episode DataFrame.
    
    Args:
        df: DataFrame with OHLCV data (can be capitalized or lowercase columns)
        
    Returns:
        DataFrame with computed technical indicators
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug("Starting technical indicator processing for episode")
        
        # Convert to working copy
        df_work = df.copy()
        
        # Check which column format we have and create lowercase versions for indicator calculations
        has_caps = all(col in df_work.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        has_lower = all(col in df_work.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        if has_caps:
            # Map capitalized columns to lowercase for indicator calculations
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Timestamp': 'timestamp'
            }
            
            for cap_col, low_col in column_mapping.items():
                if cap_col in df_work.columns:
                    df_work[low_col] = df_work[cap_col]
        elif has_lower:
            # Data is already lowercase, just ensure we have timestamp if needed
            if 'timestamp' not in df_work.columns and 'Timestamp' in df_work.columns:
                df_work['timestamp'] = df_work['Timestamp']
        else:
            logger.error(f"Cannot find OHLCV columns in expected format. Available columns: {df_work.columns.tolist()}")
            return df
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_work.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for indicator processing: {missing_cols}")
            return df
        
        # Ensure numeric data
        df_work = ensure_numeric(df_work, required_cols)
        
        # Fill NaN values in price columns
        for col in ['open', 'high', 'low', 'close']:
            if df_work[col].isna().any():
                logger.warning(f"Found NaN values in {col} column. Filling with forward fill.")
                df_work[col] = df_work[col].fillna(method='ffill')
        
        # Calculate normalized close
        window = min(100, len(df_work))  # Use last 100 bars or episode length
        rolling_min = df_work['close'].rolling(window=window, min_periods=1).min()
        rolling_max = df_work['close'].rolling(window=window, min_periods=1).max()
        
        # Avoid division by zero and ensure values are in [0, 1]
        df_work['close_norm'] = np.where(
            rolling_max > rolling_min,
            (df_work['close'] - rolling_min) / (rolling_max - rolling_min),
            0.5  # Default to middle value when there's no range
        )
        df_work['close_norm'] = df_work['close_norm'].fillna(0.5)
        
        # Calculate all enabled technical indicators
        
        # RSI (Relative Strength Index)
        if config["indicators"]["rsi"]["enabled"]:
            df_work = calculate_rsi(df_work, length=config["indicators"]["rsi"]["length"], 
                                   target_col='RSI')
            
        # CCI (Commodity Channel Index)
        if config["indicators"].get("cci", {}).get("enabled", False):
            df_work = calculate_cci(df_work, length=config["indicators"]["cci"]["length"],
                                   target_col='CCI')
        
        # ADX (Average Directional Index)
        if config["indicators"].get("adx", {}).get("enabled", False):
            df_work = calculate_adx(df_work, length=config["indicators"]["adx"]["length"],
                                   adx_col='ADX')
            
            # Add DI+ and DI- columns if needed
            if (config["indicators"].get("adx_pos", {}).get("enabled", False) or 
                config["indicators"].get("adx_neg", {}).get("enabled", False)):
                
                adx_result = ta.adx(df_work['high'], df_work['low'], df_work['close'], 
                                   length=config["indicators"]["adx"]["length"])
                
                if config["indicators"].get("adx_pos", {}).get("enabled", False):
                    df_work['ADX_POS'] = adx_result[f'DMP_{config["indicators"]["adx"]["length"]}']
                    df_work['ADX_POS'] = df_work['ADX_POS'].fillna(0.0)
                
                if config["indicators"].get("adx_neg", {}).get("enabled", False):
                    df_work['ADX_NEG'] = adx_result[f'DMN_{config["indicators"]["adx"]["length"]}']
                    df_work['ADX_NEG'] = df_work['ADX_NEG'].fillna(0.0)
        
        # Stochastic Oscillator
        if (config["indicators"].get("stoch_k", {}).get("enabled", False) or 
            config["indicators"].get("stoch_d", {}).get("enabled", False)):
            
            k_period = config["indicators"].get("stoch_k", {}).get("length", 14)
            d_period = config["indicators"].get("stoch_d", {}).get("length", 3)
            smooth_k = 3
            
            df_work = calculate_stochastic(df_work, k_period=k_period, d_period=d_period, 
                                          smooth_k=smooth_k, k_col='STOCH_K', d_col='STOCH_D')
        
        # MACD (Moving Average Convergence Divergence)
        if config["indicators"].get("macd", {}).get("enabled", False):
            df_work = calculate_macd(df_work, fast_period=config["indicators"]["macd"]["fast"],
                                    slow_period=config["indicators"]["macd"]["slow"],
                                    signal_period=config["indicators"]["macd"]["signal"],
                                    macd_col='MACD', signal_col='MACD_SIGNAL', 
                                    histogram_col='MACD_HIST')
        
        # ROC (Rate of Change)
        if config["indicators"].get("roc", {}).get("enabled", False):
            df_work = calculate_roc(df_work, length=config["indicators"]["roc"]["length"],
                                   target_col='ROC')
        
        # Williams %R
        if config["indicators"].get("williams_r", {}).get("enabled", False):
            df_work = calculate_williams_r(df_work, length=config["indicators"]["williams_r"]["length"],
                                          target_col='WILLIAMS_R')
        
        # SMA (Simple Moving Average)
        if config["indicators"].get("sma", {}).get("enabled", False):
            df_work = calculate_sma(df_work, length=config["indicators"]["sma"]["length"],
                                   target_col='SMA')
        
        # EMA (Exponential Moving Average)
        if config["indicators"].get("ema", {}).get("enabled", False):
            df_work = calculate_ema(df_work, length=config["indicators"]["ema"]["length"],
                                   target_col='EMA')
        
        # Disparity Index
        if config["indicators"].get("disparity", {}).get("enabled", False):
            df_work = calculate_disparity(df_work, 
                                         length=config["indicators"].get("disparity", {}).get("length", 20),
                                         target_col='DISPARITY')
        
        # ATR (Average True Range)
        if config["indicators"].get("atr", {}).get("enabled", False):
            df_work = calculate_atr(df_work, length=config["indicators"]["atr"]["length"],
                                   target_col='ATR')
        
        # OBV (On-Balance Volume)
        if config["indicators"].get("obv", {}).get("enabled", False):
            df_work = calculate_obv(df_work, target_col='OBV')
        
        # CMF (Chaikin Money Flow)
        if config["indicators"].get("cmf", {}).get("enabled", False):
            df_work = calculate_cmf(df_work, length=config["indicators"]["cmf"]["length"], 
                                   target_col='CMF')
        
        # PSAR (Parabolic SAR)
        if config["indicators"].get("psar", {}).get("enabled", False):
            df_work = calculate_psar(df_work, af=config["indicators"]["psar"]["af"],
                                    max_af=config["indicators"]["psar"]["max_af"], 
                                    dir_col='PSAR_DIR')
        
        # Volume indicator
        if config["indicators"].get("volume", {}).get("enabled", False):
            df_work = calculate_volume_indicator(df_work,
                                                ma_length=config["indicators"]["volume"].get("ma_length", 20),
                                                target_col='VOLUME_NORM')
        
        # VWAP (Volume Weighted Average Price)
        if config["indicators"].get("vwap", {}).get("enabled", False):
            df_work = calculate_vwap(df_work, target_col='VWAP')
        
        # Supertrend indicator
        if config["indicators"].get("supertrend", {}).get("enabled", False):
            df_work = calculate_supertrend(df_work, 
                                          length=config["indicators"]["supertrend"]["length"],
                                          multiplier=config["indicators"]["supertrend"]["multiplier"],
                                          smooth_periods=config["indicators"]["supertrend"].get("smooth_periods", 3),
                                          lookback_periods=config["indicators"]["supertrend"].get("lookback_periods", 2),
                                          target_col='supertrend')
        
        # Day of week indicator
        if config["indicators"].get("day_of_week", {}).get("enabled", True):
            df_work = calculate_day_of_week(df_work, dow_col='DOW', sin_col='DOW_SIN', cos_col='DOW_COS')
            
        # Minutes since cash open indicator
        if config["indicators"].get("minutes_since_open", {}).get("enabled", False):
            df_work = calculate_minutes_since_open(df_work, sin_col='MSO_SIN', cos_col='MSO_COS')
        
        # RRCF anomaly detection indicator
        if config["indicators"].get("rrcf_anomaly", {}).get("enabled", False):
            rrcf_config = config["indicators"]["rrcf_anomaly"]
            df_work = calculate_rrcf_anomaly(
                df_work, 
                feature_cols=rrcf_config.get("feature_cols", ["close", "volume"]),
                window_size=rrcf_config.get("window_size", 100),
                num_trees=rrcf_config.get("num_trees", 40),
                tree_size=rrcf_config.get("tree_size", 256),
                target_col='RRCF_ANOMALY',
                random_seed=rrcf_config.get("random_seed", 42)
            )

        # Z-Score indicator
        if config["indicators"].get("z_score", {}).get("enabled", False):
            df_work = calculate_zscore(df_work, length=config["indicators"]["z_score"].get("length", 50),
                                      target_col='ZScore')
        
        # Create position column
        df_work['position'] = 0
        
        # Create normalized indicator columns that the model expects
        # SMA_NORM from SMA
        if 'SMA' in df_work.columns and 'SMA_NORM' not in df_work.columns:
            df_work['SMA_NORM'] = df_work['SMA']
            
        # EMA_NORM from EMA
        if 'EMA' in df_work.columns and 'EMA_NORM' not in df_work.columns:
            df_work['EMA_NORM'] = df_work['EMA']
            
        # VOLUME_MA from VOLUME_NORM
        if 'VOLUME_NORM' in df_work.columns and 'VOLUME_MA' not in df_work.columns:
            df_work['VOLUME_MA'] = df_work['VOLUME_NORM']
        
        # Fill any remaining NaN values
        df_work = df_work.fillna(0)
        
        # Copy back the original structure but with calculated indicators
        result_df = df.copy()
        
        # Add all the calculated indicators to the original dataframe
        for col in df_work.columns:
            if col not in result_df.columns:
                result_df[col] = df_work[col]
        
        logger.debug(f"Technical indicators processing completed. Added columns: {[col for col in df_work.columns if col not in df.columns]}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return df

def process_single_episode(csv_file, synthetic_dir, output_dir, feature_range, use_sigmoid, sigmoid_k, save_scalers):
    """
    Process a single episode file: calculate technical indicators and normalize.
    
    Args:
        csv_file: Name of the CSV file to process
        synthetic_dir: Directory containing synthetic CSV files
        output_dir: Directory to save normalized files
        feature_range: Target range for scaled features
        use_sigmoid: Whether to apply sigmoid transformation
        sigmoid_k: Steepness parameter for sigmoid
        save_scalers: Whether to save scalers
        
    Returns:
        Tuple of (episode_name, scaler, success_flag, error_message)
    """
    logger = logging.getLogger(__name__)
    
    try:
        episode_name = os.path.splitext(csv_file)[0]
        input_path = os.path.join(synthetic_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)
        
        logger.info(f"Processing episode: {episode_name}")
        
        # Load the episode data
        data = pd.read_csv(input_path)
        
        # Check if data has the expected columns (handle both capitalized and lowercase)
        expected_cols_caps = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        expected_cols_lower = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if we have either capitalized or lowercase columns
        has_caps = all(col in data.columns for col in expected_cols_caps)
        has_lower = all(col in data.columns for col in expected_cols_lower)
        
        if not (has_caps or has_lower):
            missing_caps = [col for col in expected_cols_caps if col not in data.columns]
            missing_lower = [col for col in expected_cols_lower if col not in data.columns]
            error_msg = f"Episode {episode_name} missing expected columns. Available: {data.columns.tolist()}"
            logger.warning(error_msg)
            return episode_name, None, False, error_msg
        
        # Convert timestamp to datetime if it's not already
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        elif 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        logger.info(f"Episode {episode_name}: {len(data)} rows")
        
        # Log basic statistics - use appropriate column case
        if has_caps:
            logger.info(f"  Close range: [{data['Close'].min():.2f}, {data['Close'].max():.2f}], "
                       f"Volume range: [{data['Volume'].min():.2f}, {data['Volume'].max():.2f}]")
        else:
            logger.info(f"  close range: [{data['close'].min():.2f}, {data['close'].max():.2f}], "
                       f"volume range: [{data['volume'].min():.2f}, {data['volume'].max():.2f}]")
        
        # Step 1: Process technical indicators
        logger.debug(f"Calculating technical indicators for {episode_name}")
        data_with_indicators = process_technical_indicators_for_episode(data)
        
        # Step 2: Determine which columns to normalize
        # Get all technical indicators and price columns that should be normalized
        cols_to_normalize = get_standardized_column_names(data_with_indicators)
        
        # Also include the original OHLCV columns (use appropriate case) - EXCEPT close columns
        if has_caps:
            ohlcv_cols = ['Open', 'High', 'Low', 'Volume']  # Removed 'Close'
        else:
            ohlcv_cols = ['open', 'high', 'low', 'volume']  # Removed 'close'
            
        for col in ohlcv_cols:
            if col in data_with_indicators.columns and col not in cols_to_normalize:
                cols_to_normalize.append(col)
        
        logger.debug(f"Columns to normalize for {episode_name}: {cols_to_normalize}")
        
        # Step 3: Normalize the episode data (both OHLCV and technical indicators)
        normalized_data, scaler = normalize_data(
            data=data_with_indicators,
            cols_to_scale=cols_to_normalize,
            feature_range=feature_range,
            use_sigmoid=use_sigmoid,
            sigmoid_k=sigmoid_k,
            save_path=os.path.join(output_dir, f"{episode_name}_scaler.pkl") if save_scalers else None
        )
        
        # Save the normalized episode
        normalized_data.to_csv(output_path, index=False)
        
        logger.info(f"Successfully processed and normalized episode: {episode_name}")
        
        # Log statistics for key indicators (Note: Close columns are NOT normalized)
        key_indicators = ['RSI', 'supertrend', 'SMA_NORM', 'EMA_NORM', 'Open', 'High', 'Low', 'Volume']
        for indicator in key_indicators:
            if indicator in normalized_data.columns:
                values = normalized_data[indicator]
                logger.info(f"  {indicator}: range [{values.min():.4f}, {values.max():.4f}], mean {values.mean():.4f}")
        
        # Log Close column statistics separately (not normalized)
        close_col = 'Close' if has_caps else 'close'
        if close_col in normalized_data.columns:
            values = normalized_data[close_col]
            logger.info(f"  {close_col} (NOT normalized): range [{values.min():.4f}, {values.max():.4f}], mean {values.mean():.4f}")
        
        return episode_name, scaler, True, None
        
    except Exception as e:
        error_msg = f"Error processing episode {csv_file}: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return episode_name, None, False, error_msg

def normalize_synthetic_episodes_parallel(synthetic_dir="./synthetic", 
                                         output_dir=None,
                                         feature_range=(-1, 1),
                                         use_sigmoid=True,
                                         sigmoid_k=2.0,
                                         save_scalers=True,
                                         n_jobs=None):
    """
    Process technical indicators and normalize all synthetic episode files in parallel.
    
    Args:
        synthetic_dir: Directory containing synthetic CSV files
        output_dir: Directory to save normalized files (if None, overwrites originals)
        feature_range: Target range for scaled features
        use_sigmoid: Whether to apply sigmoid transformation
        sigmoid_k: Steepness parameter for sigmoid
        save_scalers: Whether to save scalers
        n_jobs: Number of parallel jobs (if None, uses all available cores)
        
    Returns:
        Dictionary mapping episode filenames to their respective scalers
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(synthetic_dir):
        logger.error(f"Synthetic directory not found: {synthetic_dir}")
        return {}
    
    # Get all CSV files in the synthetic directory
    csv_files = [f for f in os.listdir(synthetic_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {synthetic_dir}")
        return {}
    
    logger.info(f"Found {len(csv_files)} CSV files to process in {synthetic_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = synthetic_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set number of jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    
    logger.info(f"Processing files using {n_jobs} parallel workers")
    logger.info(f"Each file will be processed with technical indicators and then normalized")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_episode,
        synthetic_dir=synthetic_dir,
        output_dir=output_dir,
        feature_range=feature_range,
        use_sigmoid=use_sigmoid,
        sigmoid_k=sigmoid_k,
        save_scalers=save_scalers
    )
    
    # Process files in parallel
    scalers = {}
    successful_episodes = []
    failed_episodes = []
    
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, csv_files)
    
    # Process results
    for episode_name, scaler, success, error_msg in results:
        if success:
            scalers[episode_name] = scaler
            successful_episodes.append(episode_name)
        else:
            failed_episodes.append((episode_name, error_msg))
    
    logger.info(f"Successfully processed {len(successful_episodes)} out of {len(csv_files)} episodes")
    
    if failed_episodes:
        logger.warning(f"Failed to process {len(failed_episodes)} episodes:")
        for episode_name, error_msg in failed_episodes:
            logger.warning(f"  - {episode_name}: {error_msg}")
    
    # Save a master processing info file
    if save_scalers and scalers:
        import pickle
        master_info = {
            'total_episodes': len(scalers),
            'feature_range': feature_range,
            'use_sigmoid': use_sigmoid,
            'sigmoid_k': sigmoid_k,
            'processed_episodes': list(scalers.keys()),
            'failed_episodes': [ep for ep, _ in failed_episodes],
            'technical_indicators_processed': True,
            'enabled_indicators': {k: v for k, v in config["indicators"].items() if v.get("enabled", False)}
        }
        
        master_info_path = os.path.join(output_dir, "processing_info.pkl")
        with open(master_info_path, "wb") as f:
            pickle.dump(master_info, f)
        logger.info(f"Saved master processing info to {master_info_path}")
    
    return scalers

def main():
    """Main function to process and normalize synthetic episodes."""
    parser = argparse.ArgumentParser(description='Process technical indicators and normalize synthetic episode files for pretraining')
    
    # Set default values as constants for better maintainability
    DEFAULT_SYNTHETIC_DIR = './synthetic'
    DEFAULT_OUTPUT_DIR = './synthetic_normalized'
    DEFAULT_FEATURE_RANGE = [-1.0, 1.0]
    DEFAULT_SIGMOID_K = 2.0
    DEFAULT_LOG_LEVEL = 'INFO'
    
    parser.add_argument(
        '--synthetic-dir', 
        type=str, 
        default=DEFAULT_SYNTHETIC_DIR,
        help=f'Directory containing synthetic CSV files (default: {DEFAULT_SYNTHETIC_DIR})'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save processed and normalized files (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--feature-range', 
        type=float, 
        nargs=2, 
        default=DEFAULT_FEATURE_RANGE,
        help=f'Target range for scaled features (default: {DEFAULT_FEATURE_RANGE[0]} {DEFAULT_FEATURE_RANGE[1]})'
    )
    parser.add_argument(
        '--no-sigmoid', 
        action='store_true',
        help='Disable sigmoid transformation after min-max scaling (default: sigmoid enabled)'
    )
    parser.add_argument(
        '--sigmoid-k', 
        type=float, 
        default=DEFAULT_SIGMOID_K,
        help=f'Steepness parameter for sigmoid (default: {DEFAULT_SIGMOID_K})'
    )
    parser.add_argument(
        '--no-scalers', 
        action='store_true',
        help='Do not save scaler objects for each episode (default: save scalers)'
    )
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=None,
        help='Number of parallel jobs (default: use all available CPU cores)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'Logging level (default: {DEFAULT_LOG_LEVEL})'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files in output directory (default: skip existing files)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting synthetic episodes processing: technical indicators + normalization")
    logger.info(f"Synthetic directory: {args.synthetic_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Feature range: {args.feature_range}")
    logger.info(f"Use sigmoid: {not args.no_sigmoid}")
    if not args.no_sigmoid:
        logger.info(f"Sigmoid k parameter: {args.sigmoid_k}")
    logger.info(f"Save scalers: {not args.no_scalers}")
    logger.info(f"Parallel jobs: {args.n_jobs if args.n_jobs else 'All available CPU cores'}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Overwrite existing files: {args.overwrite}")
    
    # Check if synthetic directory exists
    if not os.path.exists(args.synthetic_dir):
        logger.error(f"Synthetic directory not found: {args.synthetic_dir}")
        sys.exit(1)
    
    # Check if output directory exists and handle overwrite option
    if os.path.exists(args.output_dir) and not args.overwrite:
        logger.warning(f"Output directory already exists: {args.output_dir}")
        logger.info("Use --overwrite flag to overwrite existing files, or choose a different output directory")
        
        # Check if there are any CSV files that would be overwritten
        existing_files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv')]
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing CSV files in output directory")
    
    # Process and normalize all episodes
    scalers = normalize_synthetic_episodes_parallel(
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        feature_range=tuple(args.feature_range),
        use_sigmoid=not args.no_sigmoid,
        sigmoid_k=args.sigmoid_k,
        save_scalers=not args.no_scalers,
        n_jobs=args.n_jobs
    )
    
    if scalers:
        logger.info(f"Successfully processed {len(scalers)} episodes with technical indicators and normalization")
        logger.info("Episodes processed:")
        for episode_name in sorted(scalers.keys()):
            logger.info(f"  - {episode_name}")
    else:
        logger.error("No episodes were successfully processed")
        sys.exit(1)
    
    logger.info("Processing completed successfully!")
    logger.info("Each episode now contains:")
    logger.info("  - Original OHLCV data (Open, High, Low, Volume normalized; Close NOT normalized)")
    logger.info("  - All enabled technical indicators (normalized)")
    logger.info("  - Model-ready features for training")

if __name__ == "__main__":
    main() 