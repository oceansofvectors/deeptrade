#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for EMA (Exponential Moving Average) indicator.

This test:
1. Creates sample price data
2. Calculates EMA using the function in indicators/ema.py
3. Verifies the calculation is correct by checking EMA properties
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import unittest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_ema")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the EMA calculation function
from indicators.ema import calculate_ema


class TestEMA(unittest.TestCase):
    """Test class for EMA indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with clear trends
        # Use more data points to allow EMA to fully develop
        np.random.seed(42)
        
        # Create 100 days of price data with stable, uptrend and downtrend periods
        prices = []
        for i in range(100):
            if i < 20:
                # Stable
                price = 100 + np.random.randn() * 0.5
            elif i < 60:
                # Uptrend
                price = 100 + (i - 20) * 1.5 + np.random.randn() * 0.5
            else:
                # Downtrend
                price = 160 - (i - 60) * 1.5 + np.random.randn() * 0.5
            
            prices.append(price)
            
        self.test_data = pd.DataFrame({
            'close': prices
        })
    
    def test_default_parameters(self):
        """Test EMA calculation with default parameters."""
        # Calculate EMA with default parameters
        result_df = calculate_ema(self.test_data.copy())
        
        # Verify EMA column exists
        self.assertTrue('EMA' in result_df.columns)
        
        # EMA should have values for all rows
        self.assertEqual(len(result_df['EMA']), len(self.test_data))
        
        # The EMA should be fully established after the initial period
        # Check for uptrend lag (EMA should be below price in uptrend)
        # Check in the middle of the uptrend section
        uptrend_idx = 40
        uptrend_diff = result_df['close'].iloc[uptrend_idx] - result_df['EMA'].iloc[uptrend_idx]
        logger.info(f"Uptrend price-EMA difference at idx {uptrend_idx}: {uptrend_diff}")
        self.assertGreater(uptrend_diff, 0, "EMA should lag below price in a consistent uptrend")
        
        # Check for downtrend lag (EMA should be above price in downtrend)
        # Check in the middle of the downtrend section
        downtrend_idx = 80
        downtrend_diff = result_df['EMA'].iloc[downtrend_idx] - result_df['close'].iloc[downtrend_idx]
        logger.info(f"Downtrend EMA-price difference at idx {downtrend_idx}: {downtrend_diff}")
        self.assertGreater(downtrend_diff, 0, "EMA should lag above price in a consistent downtrend")
        
        # Log some values for manual inspection
        logger.info(f"EMA values: {result_df['EMA'].iloc[[0, 20, 40, 60, 80]].tolist()}")
    
    def test_custom_parameters(self):
        """Test EMA calculation with custom parameters."""
        # Calculate with custom length and column name
        custom_length = 10
        default_length = 20
        
        result_df = calculate_ema(
            self.test_data.copy(), 
            length=custom_length,
            target_col='CustomEMA'
        )
        
        default_result = calculate_ema(
            self.test_data.copy(), 
            length=default_length
        )
        
        # Verify custom EMA column exists
        self.assertTrue('CustomEMA' in result_df.columns)
        
        # EMA should have values for all rows
        self.assertEqual(len(result_df['CustomEMA']), len(self.test_data))
        
        # Compare responsiveness in a turning point (after 60 where trend changes)
        turning_point = 62
        
        # Calculate how quickly each EMA reacts to the trend change
        # We'll measure this by comparing the deviation from the price
        custom_deviation = abs(result_df['close'].iloc[turning_point] - result_df['CustomEMA'].iloc[turning_point])
        default_deviation = abs(default_result['close'].iloc[turning_point] - default_result['EMA'].iloc[turning_point])
        
        logger.info(f"Turning point deviation - Custom ({custom_length}): {custom_deviation}, Default ({default_length}): {default_deviation}")
        
        # The shorter EMA should follow price more closely (smaller deviation)
        self.assertLess(custom_deviation, default_deviation, 
                        f"Shorter EMA (length={custom_length}) should be more responsive than longer EMA (length={default_length})")
        
        # Log some values for manual inspection
        logger.info(f"Custom EMA values: {result_df['CustomEMA'].iloc[[0, 20, 40, 60, 80]].tolist()}")
    
    def test_nan_handling(self):
        """Test that EMA calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[3:5, 'close'] = np.nan
        
        # Calculate EMA
        result_df = calculate_ema(nan_data)
        
        # Verify EMA is calculated and doesn't fail
        self.assertTrue('EMA' in result_df.columns)
        
        # The NaN values should be properly handled
        self.assertTrue(np.isnan(result_df['EMA'].iloc[3]))
        self.assertTrue(np.isnan(result_df['EMA'].iloc[4]))
        self.assertTrue(np.isnan(result_df['EMA'].iloc[5]))
        
        # Check that NaN rows are handled
        logger.info(f"EMA values with NaN in input: {result_df['EMA'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 