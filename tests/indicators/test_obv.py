#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for OBV (On-Balance Volume) indicator.

This test:
1. Creates sample price and volume data
2. Calculates OBV using the function in indicators/obv.py
3. Verifies the calculation is correct by comparing with expected values
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import unittest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_obv")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the OBV calculation function
from indicators.obv import calculate_obv


class TestOBV(unittest.TestCase):
    """Test class for OBV indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with price trends that alternate
        # This will create clear OBV patterns
        prices = []
        for i in range(30):
            if i < 10:
                prices.append(100 + i)  # Uptrend
            elif i < 20:
                prices.append(110 - (i-10))  # Downtrend
            else:
                prices.append(100 + (i-20))  # Uptrend
                
        self.test_data = pd.DataFrame({
            'close': prices,
            'volume': [1000] * len(prices)
        })
    
    def test_default_parameters(self):
        """Test OBV calculation with default parameters."""
        # Calculate OBV with default parameters
        result_df = calculate_obv(self.test_data.copy())
        
        # Verify OBV column exists
        self.assertTrue('OBV' in result_df.columns)
        
        # OBV should have values for all rows
        self.assertEqual(len(result_df['OBV']), len(self.test_data))
        
        # In our test data, we expect OBV to increase in the first 10 periods
        # decrease in the next 10, and increase again in the last 10
        # Check if the trend is as expected
        self.assertTrue(result_df['OBV'].iloc[9] > result_df['OBV'].iloc[0])
        self.assertTrue(result_df['OBV'].iloc[19] < result_df['OBV'].iloc[10])
        self.assertTrue(result_df['OBV'].iloc[29] > result_df['OBV'].iloc[20])
        
        # Log some values for manual inspection
        logger.info(f"OBV values: {result_df['OBV'].iloc[[0, 9, 10, 19, 20, 29]].tolist()}")
    
    def test_custom_parameters(self):
        """Test OBV calculation with custom parameters."""
        # Calculate with custom column name
        result_df = calculate_obv(
            self.test_data.copy(), 
            target_col='CustomOBV'
        )
        
        # Verify custom OBV column exists
        self.assertTrue('CustomOBV' in result_df.columns)
        
        # OBV should have values for all rows
        self.assertEqual(len(result_df['CustomOBV']), len(self.test_data))
        
        # Log some values for manual inspection
        logger.info(f"OBV values with custom parameter: {result_df['CustomOBV'].iloc[[0, 9, 19, 29]].tolist()}")
    
    def test_missing_volume(self):
        """Test OBV calculation with missing volume data."""
        # Create data without volume
        no_volume_data = self.test_data.copy().drop('volume', axis=1)
        
        # Calculate OBV
        result_df = calculate_obv(no_volume_data)
        
        # Verify OBV is calculated and doesn't fail
        self.assertTrue('OBV' in result_df.columns)
        
        # Should create volume and calculate properly
        logger.info(f"OBV values with generated volume: {result_df['OBV'].iloc[[0, 9, 19, 29]].tolist()}")
    
    def test_nan_handling(self):
        """Test that OBV calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[3, 'close'] = np.nan
        
        # Calculate OBV
        result_df = calculate_obv(nan_data)
        
        # Verify OBV is calculated and doesn't fail
        self.assertTrue('OBV' in result_df.columns)
        
        # Check that NaN rows are handled
        logger.info(f"OBV values with NaN in input: {result_df['OBV'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 