import logging
import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler

def sigmoid_transform(train_data: pd.DataFrame, 
                     val_data: pd.DataFrame, 
                     test_data: pd.DataFrame, 
                     cols_to_transform: List[str], 
                     k: float = 1.0,
                     window_folder: str = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply sigmoid transformation to data columns while avoiding look-forward bias.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        cols_to_transform: List of columns to transform
        k: Steepness parameter for sigmoid (default 1.0)
        window_folder: Optional folder to save the transform parameters
        
    Returns:
        Tuple of transform parameters and transformed datasets
    """
    logger = logging.getLogger(__name__)
    
    # Make copies to avoid modifying the original data
    train = train_data.copy()
    val = val_data.copy()
    test = test_data.copy()
    
    # Filter columns that actually exist in the data
    cols_to_transform = [col for col in cols_to_transform if col in train.columns]
    
    if not cols_to_transform:
        logger.warning("No columns to transform found in the data")
        return {}, train, val, test
    
    logger.info(f"Applying sigmoid transformation to {len(cols_to_transform)} columns using data from {len(train)} training rows")
    logger.info(f"Columns to transform: {cols_to_transform}")
    
    transform_params = {}
    
    # For each column, calculate transformation parameters from training data only
    for col in cols_to_transform:
        if col in train.columns:
            # Calculate mean and standard deviation from training data
            mean = train[col].mean()
            std = max(train[col].std(), 1e-6)  # Avoid division by zero
            
            # Store parameters
            transform_params[col] = {'mean': mean, 'std': std, 'k': k}
            
            # Apply sigmoid: 2 / (1 + exp(-k * (x - mean) / std)) - 1  (maps to [-1, 1])
            train[col] = 2 / (1 + np.exp(-k * ((train[col] - mean) / std))) - 1
            val[col] = 2 / (1 + np.exp(-k * ((val[col] - mean) / std))) - 1
            test[col] = 2 / (1 + np.exp(-k * ((test[col] - mean) / std))) - 1
            
            logger.info(f"Applied sigmoid transform to {col}: mean={mean:.4f}, std={std:.4f}, k={k}")
    
    # Save the transform parameters if a window folder is provided
    if window_folder:
        params_path = os.path.join(window_folder, "sigmoid_params.pkl")
        with open(params_path, "wb") as f:
            pickle.dump(transform_params, f)
        logger.info(f"Sigmoid parameters saved to {params_path}")
    
    return transform_params, train, val, test

def scale_window(train_data: pd.DataFrame, 
                val_data: pd.DataFrame, 
                test_data: pd.DataFrame, 
                cols_to_scale: List[str], 
                feature_range: Tuple[float, float] = (-1, 1),
                window_folder: str = None,
                use_sigmoid: bool = True,
                sigmoid_k: float = 2.0) -> Tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale technical indicators in each window to prevent data leakage.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        cols_to_scale: List of columns to scale
        feature_range: Target range for scaled features, default (-1, 1)
        window_folder: Optional folder to save the scaler
        use_sigmoid: Whether to apply sigmoid transformation after min-max scaling
        sigmoid_k: Steepness parameter for sigmoid if used
        
    Returns:
        Tuple containing the scaler and the transformed datasets
    """
    logger = logging.getLogger(__name__)
    
    # Make copies to avoid modifying the original data
    train = train_data.copy()
    val = val_data.copy()
    test = test_data.copy()
    
    # Filter columns that actually exist in the data
    cols_to_scale = [col for col in cols_to_scale if col in train.columns]
    
    if not cols_to_scale:
        logger.warning("No columns to scale found in the data")
        return None, train, val, test
    
    logger.info(f"Scaling {len(cols_to_scale)} indicator columns using data from {len(train)} training rows")
    logger.info(f"Columns to scale: {cols_to_scale}")
    
    # Create and fit the scaler on training data only
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit on training data
    scaler.fit(train[cols_to_scale])
    
    # Transform training, validation, and test data
    train[cols_to_scale] = scaler.transform(train[cols_to_scale])
    val[cols_to_scale] = scaler.transform(val[cols_to_scale])
    test[cols_to_scale] = scaler.transform(test[cols_to_scale])
    
    # Check if any test values fall outside the feature range
    # This is evidence that the scaler never saw these rows (validation)
    min_vals = test[cols_to_scale].min()
    max_vals = test[cols_to_scale].max()
    
    out_of_range_cols = []
    for col in cols_to_scale:
        if min_vals[col] < feature_range[0] or max_vals[col] > feature_range[1]:
            out_of_range_cols.append(col)
            logger.info(f"Column {col} has test values outside of range {feature_range}: min={min_vals[col]}, max={max_vals[col]}")
    
    if not out_of_range_cols:
        logger.warning("No test values found outside of the feature range. This might indicate transformation issues.")
    
    # Save the scaler if a window folder is provided
    if window_folder:
        scaler_path = os.path.join(window_folder, "indicator_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Indicator scaler saved to {scaler_path}")
    
    # Apply sigmoid transformation if requested
    if use_sigmoid:
        logger.info(f"Applying sigmoid transformation after min-max scaling")
        if out_of_range_cols:
            logger.info(f"Applying sigmoid to handle {len(out_of_range_cols)} columns with out-of-range values")
            
            # Apply sigmoid transformation only to columns with out-of-range values
            sigmoid_params, train, val, test = sigmoid_transform(
                train_data=train, 
                val_data=val,
                test_data=test,
                cols_to_transform=out_of_range_cols,
                k=sigmoid_k,
                window_folder=window_folder
            )
            
    return scaler, train, val, test

def normalize_data(data: pd.DataFrame, 
                  cols_to_scale: List[str], 
                  feature_range: Tuple[float, float] = (-1, 1),
                  scaler: Optional[MinMaxScaler] = None,
                  save_path: Optional[str] = None,
                  use_sigmoid: bool = False,
                  sigmoid_k: float = 2.0,
                  sigmoid_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize data using MinMaxScaler.
    
    Args:
        data: DataFrame to normalize
        cols_to_scale: List of columns to scale
        feature_range: Target range for scaled features, default (-1, 1)
        scaler: Optional pre-fitted scaler to use
        save_path: Optional path to save the scaler
        use_sigmoid: Whether to apply sigmoid transformation after scaling
        sigmoid_k: Steepness parameter for sigmoid if used
        sigmoid_params: Pre-calculated sigmoid parameters
        
    Returns:
        Tuple of normalized DataFrame and scaler
    """
    logger = logging.getLogger(__name__)
    
    # Make a copy to avoid modifying the original data
    normalized_data = data.copy()

    # Remove any existing normalized indicator columns to force recreation
    for col in ['SMA_NORM', 'EMA_NORM', 'VOLUME_MA', 'VWAP_NORM', 'PSAR_NORM', 'OBV_NORM']:
        if col in normalized_data.columns:
            normalized_data.drop(columns=col, inplace=True)
    
    # Filter columns that actually exist in the data
    cols_to_scale = [col for col in cols_to_scale if col in normalized_data.columns]
    
    if not cols_to_scale:
        logger.warning("No columns to scale found in the data")
        return normalized_data, scaler
    
    # Create a new scaler if none is provided
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        # Fit the scaler on the data
        scaler.fit(normalized_data[cols_to_scale])
        logger.info(f"Created new scaler for {len(cols_to_scale)} columns with range {feature_range}")
    
    # Apply the transformation
    normalized_data[cols_to_scale] = scaler.transform(normalized_data[cols_to_scale])
    logger.info(f"Normalized {len(cols_to_scale)} columns in DataFrame with {len(normalized_data)} rows")
    
    # Apply sigmoid transformation if requested
    if use_sigmoid:
        logger.info(f"Applying sigmoid transformation after min-max scaling")
        
        # Find columns with values outside feature range
        min_vals = normalized_data[cols_to_scale].min()
        max_vals = normalized_data[cols_to_scale].max()
        
        out_of_range_cols = []
        for col in cols_to_scale:
            if min_vals[col] < feature_range[0] or max_vals[col] > feature_range[1]:
                out_of_range_cols.append(col)
        
        if out_of_range_cols:
            logger.info(f"Found {len(out_of_range_cols)} columns with out-of-range values")
            
            # Apply sigmoid transformation to each out-of-range column
            for col in out_of_range_cols:
                if sigmoid_params and col in sigmoid_params:
                    # Use provided parameters
                    mean = sigmoid_params[col]['mean']
                    std = sigmoid_params[col]['std']
                    k = sigmoid_params[col]['k']
                else:
                    # Calculate parameters
                    mean = normalized_data[col].mean()
                    std = max(normalized_data[col].std(), 1e-6)
                    k = sigmoid_k
                
                # Apply sigmoid transform
                normalized_data[col] = 2 / (1 + np.exp(-k * ((normalized_data[col] - mean) / std))) - 1
    
    # Create normalized versions of specific indicators that the model expects
    
    # SMA_NORM from SMA_20 or SMA
    if 'SMA_20' in normalized_data.columns and 'SMA_NORM' not in normalized_data.columns:
        logger.info("Creating SMA_NORM from SMA_20")
        if 'SMA_20' in cols_to_scale:
            # SMA_20 was already normalized, so just rename it
            normalized_data['SMA_NORM'] = normalized_data['SMA_20']
        else:
            # Normalize SMA_20 separately
            sma_mean = normalized_data['SMA_20'].mean()
            sma_std = max(normalized_data['SMA_20'].std(), 1e-6)
            normalized_data['SMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['SMA_20'] - sma_mean) / sma_std))) - 1
    elif 'SMA' in normalized_data.columns and 'SMA_NORM' not in normalized_data.columns:
        logger.info("Creating SMA_NORM from SMA")
        if 'SMA' in cols_to_scale:
            # SMA was already normalized, so just rename it
            normalized_data['SMA_NORM'] = normalized_data['SMA']
        else:
            # Normalize SMA separately
            close_mean = normalized_data['close'].mean() if 'close' in normalized_data.columns else normalized_data['SMA'].mean()
            close_std = max(normalized_data['close'].std(), 1e-6) if 'close' in normalized_data.columns else max(normalized_data['SMA'].std(), 1e-6)
            normalized_data['SMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['SMA'] - close_mean) / close_std))) - 1
    
    # EMA_NORM from EMA_20 or EMA
    if 'EMA_20' in normalized_data.columns and 'EMA_NORM' not in normalized_data.columns:
        logger.info("Creating EMA_NORM from EMA_20")
        if 'EMA_20' in cols_to_scale:
            # EMA_20 was already normalized, so just rename it
            normalized_data['EMA_NORM'] = normalized_data['EMA_20']
        else:
            # Normalize EMA_20 separately
            ema_mean = normalized_data['EMA_20'].mean()
            ema_std = max(normalized_data['EMA_20'].std(), 1e-6)
            normalized_data['EMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['EMA_20'] - ema_mean) / ema_std))) - 1
    elif 'EMA' in normalized_data.columns and 'EMA_NORM' not in normalized_data.columns:
        logger.info("Creating EMA_NORM from EMA")
        if 'EMA' in cols_to_scale:
            # EMA was already normalized, so just rename it
            normalized_data['EMA_NORM'] = normalized_data['EMA']
        else:
            # Normalize EMA separately
            close_mean = normalized_data['close'].mean() if 'close' in normalized_data.columns else normalized_data['EMA'].mean()
            close_std = max(normalized_data['close'].std(), 1e-6) if 'close' in normalized_data.columns else max(normalized_data['EMA'].std(), 1e-6)
            normalized_data['EMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['EMA'] - close_mean) / close_std))) - 1
    
    # VOLUME_MA from VOLUME or VOLUME_NORM
    if 'VOLUME' in normalized_data.columns and 'VOLUME_MA' not in normalized_data.columns:
        logger.info("Creating VOLUME_MA from VOLUME")
        if 'VOLUME' in cols_to_scale:
            # VOLUME was already normalized, so just rename it
            normalized_data['VOLUME_MA'] = normalized_data['VOLUME']
        else:
            # Normalize VOLUME separately
            vol_mean = normalized_data['VOLUME'].mean()
            vol_std = max(normalized_data['VOLUME'].std(), 1e-6)
            normalized_data['VOLUME_MA'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['VOLUME'] - vol_mean) / vol_std))) - 1
    elif 'VOLUME_NORM' in normalized_data.columns and 'VOLUME_MA' not in normalized_data.columns:
        logger.info("Creating VOLUME_MA from VOLUME_NORM")
        normalized_data['VOLUME_MA'] = normalized_data['VOLUME_NORM']
    
    # VWAP_NORM from VWAP if needed
    if 'VWAP' in normalized_data.columns and 'VWAP_NORM' not in normalized_data.columns:
        logger.info("Creating VWAP_NORM from VWAP")
        if 'VWAP' in cols_to_scale:
            # VWAP was already normalized, so just rename it
            normalized_data['VWAP_NORM'] = normalized_data['VWAP']
        else:
            # Normalize VWAP separately
            vwap_mean = normalized_data['VWAP'].mean()
            vwap_std = max(normalized_data['VWAP'].std(), 1e-6)
            normalized_data['VWAP_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['VWAP'] - vwap_mean) / vwap_std))) - 1
    
    # PSAR_NORM from PSAR if needed
    if 'PSAR' in normalized_data.columns and 'PSAR_NORM' not in normalized_data.columns:
        logger.info("Creating PSAR_NORM from PSAR")
        if 'PSAR' in cols_to_scale:
            # PSAR was already normalized, so just rename it
            normalized_data['PSAR_NORM'] = normalized_data['PSAR']
        else:
            # Normalize PSAR separately
            psar_mean = normalized_data['PSAR'].mean()
            psar_std = max(normalized_data['PSAR'].std(), 1e-6)
            normalized_data['PSAR_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['PSAR'] - psar_mean) / psar_std))) - 1
    
    # OBV_NORM from OBV if needed
    if 'OBV' in normalized_data.columns and 'OBV_NORM' not in normalized_data.columns:
        logger.info("Creating OBV_NORM from OBV")
        if 'OBV' in cols_to_scale:
            # OBV was already normalized, so just rename it
            normalized_data['OBV_NORM'] = normalized_data['OBV']
        else:
            # Normalize OBV separately
            obv_mean = normalized_data['OBV'].mean()
            obv_std = max(normalized_data['OBV'].std(), 1e-6)
            normalized_data['OBV_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((normalized_data['OBV'] - obv_mean) / obv_std))) - 1
    
    # Handle supertrend (should be -1 or 1, not normalized)
    if 'supertrend' in normalized_data.columns and 'SUPERTREND' not in normalized_data.columns:
        logger.info("Creating SUPERTREND from supertrend")
        normalized_data['SUPERTREND'] = normalized_data['supertrend']
    
    # Handle trend_direction if it exists but SUPERTREND doesn't
    if 'trend_direction' in normalized_data.columns and 'SUPERTREND' not in normalized_data.columns:
        logger.info("Creating SUPERTREND from trend_direction")
        normalized_data['SUPERTREND'] = normalized_data['trend_direction']
    
    # Save the scaler if a path is provided
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {save_path}")
    
    return normalized_data, scaler

def load_scaler(scaler_path: str) -> Optional[MinMaxScaler]:
    """
    Load a saved scaler from disk.
    
    Args:
        scaler_path: Path to the saved scaler file
        
    Returns:
        MinMaxScaler object or None if loading fails
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at {scaler_path}")
        return None
    
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler from {scaler_path}: {e}")
        return None

def get_standardized_column_names(df: pd.DataFrame, 
                                 skip_columns: Optional[List[str]] = None) -> List[str]:
    """
    Get a list of column names that should be standardized (numeric columns except those in skip_columns).
    
    Args:
        df: Input DataFrame
        skip_columns: List of column names to skip
        
    Returns:
        List of column names that should be standardized
    """
    # Default list of columns to skip if none provided
    if skip_columns is None:
        skip_columns = [
            'close_norm', 'position', 'trend_direction', 'supertrend', 'SUPERTREND',
            'time', 'timestamp', 'date', 'DOW', 
            'Up Trend', 'Down Trend', 'open', 'high', 'low', 'close', 
            'Open', 'High', 'Low', 'Close', 'Volume', 'volume',
            'PSAR_DIR',  # Also add PSAR_DIR as it's a categorical indicator (-1, 1)
            # Skip any already normalized versions of indicators
            'SMA_NORM', 'EMA_NORM', 'OBV_NORM', 'PSAR_NORM', 'VWAP_NORM', 'VOLUME_MA',
            # Skip cyclical time features
            'DOW_SIN', 'DOW_COS', 'MSO_SIN', 'MSO_COS'
        ]
    
    # Get all numeric columns except those in skip_columns
    cols_to_scale = []
    for col in df.columns:
        if col not in skip_columns and pd.api.types.is_numeric_dtype(df[col]):
            cols_to_scale.append(col)
    
    return cols_to_scale 