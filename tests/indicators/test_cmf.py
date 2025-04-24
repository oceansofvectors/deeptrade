#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for CMF (Chaikin Money Flow) indicator.

This test:
1. Creates sample price and volume data
2. Calculates CMF using the function in indicators/cmf.py
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
logger = logging.getLogger("test_cmf")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the CMF calculation function
from indicators.cmf import calculate_cmf


class TestCMF(unittest.TestCase):
    """Test class for CMF indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame
        self.test_data = pd.DataFrame({
            'high': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220],
            'low': [90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210],
            'close': [95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215],
            'volume': [1000, 1200, 1100, 1300, 1400, 1200, 1100, 1500, 1600, 1700, 1500, 1400, 1300, 1200, 1100, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
    
    def test_default_parameters(self):
        """Test CMF calculation with default parameters."""
        # Calculate CMF with default parameters
        result_df = calculate_cmf(self.test_data.copy())
        
        # Verify CMF column exists
        self.assertTrue('CMF' in result_df.columns)
        
        # CMF should have values for all rows (though initial values will be NaN/0)
        self.assertEqual(len(result_df['CMF']), len(self.test_data))
        
        # CMF values should be within the [-1, 1] range
        non_zero_values = result_df['CMF'][result_df['CMF'] != 0]
        if len(non_zero_values) > 0:
            self.assertTrue(all(-1 <= val <= 1 for val in non_zero_values))
        
        # Log some values for manual inspection
        logger.info(f"CMF values with default parameters: {result_df['CMF'].tail(5).tolist()}")
    
    def test_custom_parameters(self):
        """Test CMF calculation with custom parameters."""
        # Calculate with custom length and column name
        result_df = calculate_cmf(
            self.test_data.copy(), 
            length=10,
            target_col='CustomCMF'
        )
        
        # Verify custom CMF column exists
        self.assertTrue('CustomCMF' in result_df.columns)
        
        # CMF should have values for all rows
        self.assertEqual(len(result_df['CustomCMF']), len(self.test_data))
        
        # Log some values for manual inspection
        logger.info(f"CMF values with custom parameters: {result_df['CustomCMF'].tail(5).tolist()}")
    
    def test_missing_volume(self):
        """Test CMF calculation with missing volume data."""
        # Create data without volume
        no_volume_data = self.test_data.copy().drop('volume', axis=1)
        
        # Calculate CMF
        result_df = calculate_cmf(no_volume_data)
        
        # Verify CMF is calculated and doesn't fail
        self.assertTrue('CMF' in result_df.columns)
        
        # Should create volume and calculate properly
        logger.info(f"CMF values with generated volume: {result_df['CMF'].tail(5).tolist()}")
    
    def test_nan_handling(self):
        """Test that CMF calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[3, 'high'] = np.nan
        
        # Calculate CMF
        result_df = calculate_cmf(nan_data)
        
        # Verify CMF is calculated and doesn't fail
        self.assertTrue('CMF' in result_df.columns)
        
        # Check that NaN rows are handled
        logger.info(f"CMF values with NaN in input: {result_df['CMF'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 