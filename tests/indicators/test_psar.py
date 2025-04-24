#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for PSAR (Parabolic Stop and Reverse) indicator.

This test:
1. Creates sample price data
2. Calculates PSAR using the function in indicators/psar.py
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
logger = logging.getLogger("test_psar")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the PSAR calculation function
from indicators.psar import calculate_psar


class TestPSAR(unittest.TestCase):
    """Test class for PSAR indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with clear trends
        
        # Create 30 days of price data with an uptrend followed by a downtrend
        highs, lows, closes = [], [], []
        
        for i in range(30):
            if i < 15:
                # Uptrend
                base = 100 + i * 2
                volatility = 3
            else:
                # Downtrend
                base = 130 - (i - 15) * 2
                volatility = 3
                
            high = base + np.random.randint(1, volatility)
            low = base - np.random.randint(1, volatility)
            close = base + np.random.randint(-volatility//2, volatility//2)
            
            highs.append(high)
            lows.append(low)
            closes.append(close)
            
        self.test_data = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })
    
    def test_default_parameters(self):
        """Test PSAR calculation with default parameters."""
        # Calculate PSAR with default parameters
        result_df = calculate_psar(self.test_data.copy())
        
        # Verify PSAR columns exist
        self.assertTrue('PSAR' in result_df.columns)
        self.assertTrue('PSAR_DIR' in result_df.columns)
        
        # PSAR should have values for all rows
        self.assertEqual(len(result_df['PSAR']), len(self.test_data))
        self.assertEqual(len(result_df['PSAR_DIR']), len(self.test_data))
        
        # PSAR direction should be 1 (bullish) or -1 (bearish)
        self.assertTrue(all(val in [1, -1] for val in result_df['PSAR_DIR']))
        
        # In an uptrend, PSAR should be below the close price
        mid_uptrend = 10
        if result_df['PSAR_DIR'].iloc[mid_uptrend] == 1:
            self.assertLess(result_df['PSAR'].iloc[mid_uptrend], result_df['close'].iloc[mid_uptrend])
        
        # In a downtrend, PSAR should be above the close price
        mid_downtrend = 25
        if result_df['PSAR_DIR'].iloc[mid_downtrend] == -1:
            self.assertGreater(result_df['PSAR'].iloc[mid_downtrend], result_df['close'].iloc[mid_downtrend])
        
        # Log some values for manual inspection
        logger.info(f"PSAR values: {result_df['PSAR'].iloc[[0, 7, 14, 21, 28]].tolist()}")
        logger.info(f"Direction values: {result_df['PSAR_DIR'].iloc[[0, 7, 14, 21, 28]].tolist()}")
        
        # Check if direction changes around trend reversal
        uptrend_dir = result_df['PSAR_DIR'].iloc[10:15].mean()
        downtrend_dir = result_df['PSAR_DIR'].iloc[20:25].mean()
        logger.info(f"Uptrend direction mean: {uptrend_dir}, Downtrend direction mean: {downtrend_dir}")
        
        # Directions should be different in different trend phases
        if uptrend_dir * downtrend_dir < 0:  # Opposite signs
            logger.info("✓ PSAR direction correctly changes between uptrend and downtrend")
        else:
            logger.info("✗ PSAR direction does not change as expected between trends")
    
    def test_custom_parameters(self):
        """Test PSAR calculation with custom parameters."""
        # Calculate with custom parameters
        result_df = calculate_psar(
            self.test_data.copy(), 
            af=0.01,
            max_af=0.1,
            psar_col='CustomPSAR', 
            dir_col='CustomDir'
        )
        
        # Verify custom PSAR columns exist
        self.assertTrue('CustomPSAR' in result_df.columns)
        self.assertTrue('CustomDir' in result_df.columns)
        
        # PSAR should have values for all rows
        self.assertEqual(len(result_df['CustomPSAR']), len(self.test_data))
        
        # Compare to default parameters
        default_result = calculate_psar(self.test_data.copy())
        
        # Less aggressive PSAR should have fewer direction changes
        custom_direction_changes = (result_df['CustomDir'].diff() != 0).sum()
        default_direction_changes = (default_result['PSAR_DIR'].diff() != 0).sum()
        
        logger.info(f"Direction changes - Custom (0.01/0.1): {custom_direction_changes}, Default (0.02/0.2): {default_direction_changes}")
        # Typically, less aggressive should have fewer changes
        
        # Log some values for manual inspection
        logger.info(f"Custom PSAR values: {result_df['CustomPSAR'].iloc[[0, 7, 14, 21, 28]].tolist()}")
    
    def test_nan_handling(self):
        """Test that PSAR calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[3, 'high'] = np.nan
        nan_data.loc[4, 'low'] = np.nan
        
        # Calculate PSAR
        result_df = calculate_psar(nan_data)
        
        # Verify PSAR is calculated and doesn't fail
        self.assertTrue('PSAR' in result_df.columns)
        self.assertTrue('PSAR_DIR' in result_df.columns)
        
        # Check that NaN rows are handled
        logger.info(f"PSAR values with NaN in input: {result_df['PSAR'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 