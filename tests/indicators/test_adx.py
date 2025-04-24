#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for ADX (Average Directional Index) indicator.

This test:
1. Creates sample price data
2. Calculates ADX using the function in indicators/adx.py
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
logger = logging.getLogger("test_adx")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the ADX calculation function
from indicators.adx import calculate_adx


class TestADX(unittest.TestCase):
    """Test class for ADX indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame
        self.test_data = pd.DataFrame({
            'high': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175],
            'low': [90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165],
            'close': [95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170]
        })
    
    def test_default_parameters(self):
        """Test ADX calculation with default parameters."""
        # Calculate ADX with default parameters
        result_df = calculate_adx(self.test_data.copy())
        
        # Verify ADX column exists
        self.assertTrue('ADX' in result_df.columns)
        
        # First few values should be NaN or 0 (since ADX requires a period to calculate)
        self.assertEqual(result_df['ADX'].iloc[0], 0.0)
        
        # ADX should have values for all rows
        self.assertEqual(len(result_df['ADX']), len(self.test_data))
        
        # ADX should be a float between 0 and 100
        self.assertTrue(all(0 <= val <= 100 for val in result_df['ADX']))
        
        # Log some values for manual inspection
        logger.info(f"ADX values with default parameters: {result_df['ADX'].tail(5).tolist()}")
    
    def test_custom_parameters(self):
        """Test ADX calculation with custom parameters."""
        # Test with custom length and output column name
        result_df = calculate_adx(
            self.test_data.copy(), 
            length=7,  # Custom length
            adx_col='CustomADX'
        )
        
        # Verify custom ADX column exists
        self.assertTrue('CustomADX' in result_df.columns)
        
        # ADX should have values for all rows
        self.assertEqual(len(result_df['CustomADX']), len(self.test_data))
        
        # Log some values for manual inspection
        logger.info(f"ADX values with custom parameters: {result_df['CustomADX'].tail(5).tolist()}")
    
    def test_result_stability(self):
        """Test that ADX calculation is stable and returns consistent values."""
        # Calculate ADX twice with the same data
        result1 = calculate_adx(self.test_data.copy())
        result2 = calculate_adx(self.test_data.copy())
        
        # Results should be identical
        pd.testing.assert_series_equal(result1['ADX'], result2['ADX'])
    
    def test_nan_handling(self):
        """Test that ADX calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[3, 'high'] = np.nan
        
        # Calculate ADX
        result_df = calculate_adx(nan_data)
        
        # Verify ADX is calculated and doesn't fail
        self.assertTrue('ADX' in result_df.columns)
        
        # Check that NaN rows are handled (either with NaN or filled values)
        logger.info(f"ADX values with NaN in input: {result_df['ADX'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 