import logging
import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Optional, Union

def sigmoid_transform(train_data: pd.DataFrame, 
                     val_data: pd.DataFrame, 
                     test_data: pd.DataFrame, 
                     cols_to_transform: List[str], 
                     k: float = 1.0,
                     window_folder: str = None,
                     historical_data: pd.DataFrame = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply sigmoid transformation to data columns while avoiding look-forward bias.
    Uses expanding window approach for walk-forward validation.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        cols_to_transform: List of columns to transform
        k: Steepness parameter for sigmoid (default 1.0)
        window_folder: Optional folder to save the transform parameters
        historical_data: All historical data up to current window for parameter calculation
        
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
    
    # Use historical data if provided, otherwise use current training data
    param_data = historical_data if historical_data is not None else train
    
    if historical_data is not None:
        logger.info(f"Applying sigmoid transformation to {len(cols_to_transform)} columns using expanding window with {len(historical_data)} historical rows")
    else:
        logger.info(f"Applying sigmoid transformation to {len(cols_to_transform)} columns using data from {len(train)} training rows")
    
    logger.info(f"Columns to transform: {cols_to_transform}")
    
    transform_params = {}
    
    # For each column, calculate transformation parameters from historical/training data
    for col in cols_to_transform:
        if col in param_data.columns:
            # Calculate mean and standard deviation from historical data (expanding window)
            mean = param_data[col].mean()
            std = param_data[col].std()
            
            # Ensure minimum std to avoid division by zero and extreme sensitivity
            min_std = std * 0.1 if std > 0 else 1e-6
            std = max(std, min_std)
            
            # Store parameters
            transform_params[col] = {'mean': mean, 'std': std, 'k': k}
            
            # Apply sigmoid: 2 / (1 + exp(-k * (x - mean) / std)) - 1  (maps to [-1, 1])
            train[col] = 2 / (1 + np.exp(-k * ((train[col] - mean) / std))) - 1
            val[col] = 2 / (1 + np.exp(-k * ((val[col] - mean) / std))) - 1
            test[col] = 2 / (1 + np.exp(-k * ((test[col] - mean) / std))) - 1
            
            logger.info(f"Applied sigmoid transform to {col}: mean={mean:.4f}, std={std:.4f}, k={k} (from {len(param_data)} historical rows)")
    
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
                window_folder: str = None,
                sigmoid_k: float = 2.0,
                historical_data: pd.DataFrame = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply sigmoid transformation to technical indicators in each window to prevent data leakage.
    Uses expanding window approach for walk-forward validation.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        cols_to_scale: List of columns to scale
        window_folder: Optional folder to save the transform parameters
        sigmoid_k: Steepness parameter for sigmoid transformation
        historical_data: All historical data up to current window for parameter calculation
        
    Returns:
        Tuple containing the sigmoid parameters and the transformed datasets
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
        return {}, train, val, test
    
    if historical_data is not None:
        logger.info(f"Applying sigmoid transformation to {len(cols_to_scale)} indicator columns using expanding window with {len(historical_data)} historical rows")
    else:
        logger.info(f"Applying sigmoid transformation to {len(cols_to_scale)} indicator columns using data from {len(train)} training rows")
    
    logger.info(f"Columns to scale: {cols_to_scale}")
    
    # Apply sigmoid transformation directly
    sigmoid_params, train, val, test = sigmoid_transform(
        train_data=train, 
        val_data=val,
        test_data=test,
        cols_to_transform=cols_to_scale,
        k=sigmoid_k,
        window_folder=window_folder,
        historical_data=historical_data
    )
            
    return sigmoid_params, train, val, test

def normalize_data(data: pd.DataFrame, 
                  cols_to_scale: List[str], 
                  sigmoid_k: float = 2.0,
                  sigmoid_params: Optional[Dict] = None,
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize data using sigmoid transformation.
    
    Args:
        data: DataFrame to normalize
        cols_to_scale: List of columns to scale
        sigmoid_k: Steepness parameter for sigmoid transformation
        sigmoid_params: Pre-calculated sigmoid parameters
        save_path: Optional path to save the sigmoid parameters
        
    Returns:
        Normalized DataFrame
    """
    logger = logging.getLogger(__name__)
    
    # Make a copy to avoid modifying the original data
    normalized_data = data.copy()
    
    # Filter columns that actually exist in the data
    cols_to_scale = [col for col in cols_to_scale if col in normalized_data.columns]
    
    if not cols_to_scale:
        logger.warning("No columns to scale found in the data")
        return normalized_data
    
    logger.info(f"Applying sigmoid transformation to {len(cols_to_scale)} columns")
    
    # Calculate or use provided sigmoid parameters
    transform_params = {}
    
    # Apply sigmoid transformation to all specified columns
    for col in cols_to_scale:
        if sigmoid_params and col in sigmoid_params:
            # Use provided parameters
            mean = sigmoid_params[col]['mean']
            std = sigmoid_params[col]['std']
            k = sigmoid_params[col]['k']
        else:
            # Calculate parameters from the data
            mean = normalized_data[col].mean()
            std = max(normalized_data[col].std(), 1e-6)  # Avoid division by zero
            k = sigmoid_k
        
        # Store parameters
        transform_params[col] = {'mean': mean, 'std': std, 'k': k}
        
        # Apply sigmoid transform: 2 / (1 + exp(-k * (x - mean) / std)) - 1  (maps to [-1, 1])
        normalized_data[col] = 2 / (1 + np.exp(-k * ((normalized_data[col] - mean) / std))) - 1
        
        logger.info(f"Applied sigmoid transform to {col}: mean={mean:.4f}, std={std:.4f}, k={k}")
    
    # Save the sigmoid parameters if a path is provided
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(transform_params, f)
        logger.info(f"Sigmoid parameters saved to {save_path}")
    
    return normalized_data

def load_sigmoid_params(params_path: str) -> Optional[Dict]:
    """
    Load saved sigmoid parameters from disk.
    
    Args:
        params_path: Path to the saved sigmoid parameters file
        
    Returns:
        Dictionary of sigmoid parameters or None if loading fails
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(params_path):
        logger.error(f"Sigmoid parameters file not found at {params_path}")
        return None
    
    try:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        logger.info(f"Loaded sigmoid parameters from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading sigmoid parameters from {params_path}: {e}")
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
            'position', 'supertrend',
            'timestamp', 'date', 'DOW',
            'open', 'high', 'low', 'close', 'volume',
            'PSAR_DIR',
            'DOW_SIN', 'DOW_COS', 'MSO_SIN', 'MSO_COS'
        ]
    
    # Get all numeric columns except those in skip_columns
    cols_to_scale = []
    for col in df.columns:
        if col not in skip_columns and pd.api.types.is_numeric_dtype(df[col]):
            cols_to_scale.append(col)
    
    return cols_to_scale 