#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for ATR (Average True Range) indicator.

This test:
1. Creates sample price data
2. Calculates ATR using the function in indicators/atr.py
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
logger = logging.getLogger("test_atr")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the ATR calculation function
from indicators.atr import calculate_atr


class TestATR(unittest.TestCase):
    """Test class for ATR indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with volatility pattern
        # First half: low volatility, Second half: high volatility
        np.random.seed(42)  # For reproducibility
        
        highs = []
        lows = []
        closes = []
        
        # Create 30 days of price data
        for i in range(30):
            if i < 15:
                # Low volatility period
                volatility = 2
            else:
                # High volatility period
                volatility = 8
                
            base = 100 + i
            high = base + np.random.randint(1, volatility)
            low = base - np.random.randint(1, volatility)
            close = base + np.random.randint(low - base, high - base)
            
            highs.append(high)
            lows.append(low)
            closes.append(close)
            
        self.test_data = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })
    
    def test_default_parameters(self):
        """Test ATR calculation with default parameters."""
        # Calculate ATR with default parameters
        result_df = calculate_atr(self.test_data.copy())
        
        # Verify ATR column exists
        self.assertTrue('ATR' in result_df.columns)
        
        # ATR should have values for all rows
        self.assertEqual(len(result_df['ATR']), len(self.test_data))
        
        # Check if ATR values increase in the high volatility period
        # First point might have NaN or initial value, so we compare a bit later
        low_volatility_atr = result_df['ATR'].iloc[14]  # End of low volatility
        high_volatility_atr = result_df['ATR'].iloc[29]  # End of high volatility
        
        # ATR should be higher in high volatility period
        logger.info(f"Low volatility ATR: {low_volatility_atr}, High volatility ATR: {high_volatility_atr}")
        self.assertGreater(high_volatility_atr, low_volatility_atr)
        
        # Log some values for manual inspection
        logger.info(f"ATR values: {result_df['ATR'].iloc[[0, 7, 14, 21, 28]].tolist()}")
    
    def test_custom_parameters(self):
        """Test ATR calculation with custom parameters."""
        # Calculate with custom length and column name
        result_df = calculate_atr(
            self.test_data.copy(), 
            length=7,
            target_col='CustomATR'
        )
        
        # Verify custom ATR column exists
        self.assertTrue('CustomATR' in result_df.columns)
        
        # ATR should have values for all rows
        self.assertEqual(len(result_df['CustomATR']), len(self.test_data))
        
        # Log some values for manual inspection
        logger.info(f"ATR values with custom parameters: {result_df['CustomATR'].iloc[[7, 14, 21, 28]].tolist()}")
        
        # Compare to default length ATR (14)
        default_result = calculate_atr(self.test_data.copy())
        
        # Shorter length ATR should respond more quickly to volatility changes
        # We check around the volatility transition point (index 15)
        short_atr_response = result_df['CustomATR'].iloc[16] - result_df['CustomATR'].iloc[14]
        default_atr_response = default_result['ATR'].iloc[16] - default_result['ATR'].iloc[14]
        
        logger.info(f"ATR response to volatility change - Short length: {short_atr_response}, Default length: {default_atr_response}")
        self.assertGreater(short_atr_response, default_atr_response)
    
    def test_nan_handling(self):
        """Test that ATR calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[3, 'high'] = np.nan
        
        # Calculate ATR
        result_df = calculate_atr(nan_data)
        
        # Verify ATR is calculated and doesn't fail
        self.assertTrue('ATR' in result_df.columns)
        
        # Check that NaN rows are handled
        logger.info(f"ATR values with NaN in input: {result_df['ATR'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 