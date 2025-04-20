#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for verifying indicator scaling prevents data leakage.

This test:
1. Creates synthetic indicator data with drift in test set
2. Scales it properly using the walk_forward.scale_window function
3. Verifies test data occasionally falls outside [-1, 1] range, confirming no leakage
4. Compares with global scaling approach to demonstrate the difference
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_indicator_scaling")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our scaling function
from walk_forward import scale_window


def test_indicator_scaling():
    """Test the indicator scaling to ensure it prevents data leakage."""
    logger.info("Testing indicator scaling for data leakage prevention")
    
    # Create synthetic data with a trend that changes in test set
    np.random.seed(42)
    n_samples = 300
    
    # Generate data with a drift in the test set to simulate changing market conditions
    train_idx = int(n_samples * 0.7)
    val_idx = int(n_samples * 0.85)
    
    # Create synthetic indicators with different patterns
    # Indicator 1: Random walk with drift in test set
    indicator1 = np.cumsum(np.random.normal(0, 1, n_samples))
    # Add extra drift to test set
    indicator1[val_idx:] += np.linspace(0, 10, n_samples - val_idx)
    
    # Indicator 2: Sine wave with increasing amplitude in test
    x = np.linspace(0, 10, n_samples)
    indicator2 = np.sin(x)
    indicator2[val_idx:] *= 2  # Double amplitude in test set
    
    # Indicator 3: Mean reverting with level shift in test
    indicator3 = np.random.normal(0, 1, n_samples)
    indicator3[val_idx:] += 5  # Add level shift in test
    
    # Create dataframes
    train_data = pd.DataFrame({
        'IND1': indicator1[:train_idx],
        'IND2': indicator2[:train_idx],
        'IND3': indicator3[:train_idx],
        'close_norm': np.random.normal(0.5, 0.1, train_idx)  # Not scaled
    })
    
    val_data = pd.DataFrame({
        'IND1': indicator1[train_idx:val_idx],
        'IND2': indicator2[train_idx:val_idx],
        'IND3': indicator3[train_idx:val_idx],
        'close_norm': np.random.normal(0.5, 0.1, val_idx - train_idx)  # Not scaled
    })
    
    test_data = pd.DataFrame({
        'IND1': indicator1[val_idx:],
        'IND2': indicator2[val_idx:],
        'IND3': indicator3[val_idx:],
        'close_norm': np.random.normal(0.5, 0.1, n_samples - val_idx)  # Not scaled
    })
    
    logger.info(f"Created synthetic data: {n_samples} samples, "
               f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Print statistics about the raw data
    logger.info("Original indicator ranges:")
    for col in ['IND1', 'IND2', 'IND3']:
        logger.info(f"  {col}: train={train_data[col].min():.2f} to {train_data[col].max():.2f}, "
                   f"test={test_data[col].min():.2f} to {test_data[col].max():.2f}")
    
    # Test our new window-based scaling function
    cols_to_scale = ['IND1', 'IND2', 'IND3']
    scaler, train_scaled, val_scaled, test_scaled = scale_window(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        cols_to_scale=cols_to_scale,
        feature_range=(-1, 1)
    )
    
    # Verify proper scaling in train set
    logger.info("\nWindow-based scaling results:")
    for col in cols_to_scale:
        train_min, train_max = train_scaled[col].min(), train_scaled[col].max()
        test_min, test_max = test_scaled[col].min(), test_scaled[col].max()
        logger.info(f"  {col}: train=({train_min:.2f} to {train_max:.2f}), "
                   f"test=({test_min:.2f} to {test_max:.2f})")
        
        # Check if test data exceeds boundaries, which confirms no leakage
        if test_min < -1 or test_max > 1:
            logger.info(f"  ✓ {col} test data exceeds [-1, 1] range, confirming no leakage")
        else:
            logger.warning(f"  ✗ {col} test data within [-1, 1] range, possible issue")
    
    # For comparison, show what happens with global scaling (the leaky approach)
    logger.info("\nFor comparison - Global scaling (leaky approach):")
    
    # Combine all data for global scaling
    all_data = pd.concat([train_data, val_data, test_data])
    
    # Use global scaler
    global_scaler = MinMaxScaler(feature_range=(-1, 1))
    all_data[cols_to_scale] = global_scaler.fit_transform(all_data[cols_to_scale])
    
    # Split back into train/val/test
    global_train = all_data.iloc[:len(train_data)]
    global_val = all_data.iloc[len(train_data):len(train_data)+len(val_data)]
    global_test = all_data.iloc[len(train_data)+len(val_data):]
    
    # Show global scaling results - all will be within [-1, 1]
    for col in cols_to_scale:
        train_min, train_max = global_train[col].min(), global_train[col].max()
        test_min, test_max = global_test[col].min(), global_test[col].max()
        logger.info(f"  {col}: train=({train_min:.2f} to {train_max:.2f}), "
                   f"test=({test_min:.2f} to {test_max:.2f})")
        
        # All test data will be within [-1, 1], showing the leakage
        if test_min >= -1 and test_max <= 1:
            logger.info(f"  ✗ {col} test values contained within [-1, 1], confirming data leakage")
    
    logger.info("\nTest completed. Proper scaling should show at least some indicators "
               "with test values outside the [-1, 1] range, which confirms no leakage.")
    
    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    train_scaled, val_scaled, test_scaled = test_indicator_scaling()
    logger.info("Test complete.") 