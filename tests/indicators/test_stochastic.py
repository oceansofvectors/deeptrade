#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Stochastic indicator.

This test:
1. Creates sample price data
2. Calculates Stochastic using the function in indicators/stochastic.py
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
logger = logging.getLogger("test_stochastic")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the Stochastic calculation function
from indicators.stochastic import calculate_stochastic


class TestStochastic(unittest.TestCase):
    """Test class for Stochastic indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with specific patterns
        data = []
        
        # Create 50 days of price data with different patterns
        for i in range(50):
            if i < 10:  # First 10 days: uptrend
                high = 100 + i*2
                low = 100 + i*2 - 1
                close = 100 + i*2
            elif i < 20:  # Days 10-19: downtrend
                high = 120 - (i-10)*2
                low = 120 - (i-10)*2 - 1
                close = 120 - (i-10)*2
            elif i < 30:  # Days 20-29: consolidation (sideways)
                high = 100 + (i % 5)
                low = 95 + (i % 5)
                close = 97 + (i % 5)
            elif i < 40:  # Days 30-39: uptrend to peak and reversal
                if i < 35:
                    high = 100 + (i-30)*3
                    low = 100 + (i-30)*3 - 2
                    close = 100 + (i-30)*3 - 1
                else:
                    high = 115 - (i-35)*3
                    low = 115 - (i-35)*3 - 2
                    close = 115 - (i-35)*3 - 1
            else:  # Days 40-49: volatile sideways
                high = 100 + 10*np.sin(i)
                low = 90 + 10*np.sin(i)
                close = 95 + 10*np.sin(i)
            
            data.append({
                'high': high,
                'low': low,
                'close': close
            })
            
        self.test_data = pd.DataFrame(data)
    
    def test_default_parameters(self):
        """Test Stochastic calculation with default parameters."""
        # Calculate Stochastic with default parameters
        result_df = calculate_stochastic(self.test_data.copy())
        
        # Verify Stochastic K and D columns exist
        self.assertTrue('K' in result_df.columns)
        self.assertTrue('D' in result_df.columns)
        
        # Stochastic should have values for all rows (NaN for initial period is fine)
        self.assertEqual(len(result_df['K']), len(self.test_data))
        self.assertEqual(len(result_df['D']), len(self.test_data))
        
        # K and D values should be between 0 and 100
        non_nan_K = result_df['K'].dropna()
        non_nan_D = result_df['D'].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in non_nan_K))
        self.assertTrue(all(0 <= val <= 100 for val in non_nan_D))
        
        # Check Stochastic values at specific points
        # After uptrend (index 9), K should be high (close to 100)
        self.assertGreater(result_df['K'].iloc[9], 80)
        
        # After downtrend (index 19), K should be low (close to 0)
        self.assertLess(result_df['K'].iloc[19], 20)
        
        # D should be more smoothed than K
        # Calculate average absolute change for K and D
        k_changes = abs(result_df['K'].diff()).dropna().mean()
        d_changes = abs(result_df['D'].diff()).dropna().mean()
        
        logger.info(f"Average K changes: {k_changes}, Average D changes: {d_changes}")
        self.assertGreater(k_changes, d_changes)
        
        # Log some values for manual inspection
        logger.info(f"Stochastic values at key points:")
        logger.info(f"After uptrend (idx 9): K={result_df['K'].iloc[9]}, D={result_df['D'].iloc[9]}")
        logger.info(f"After downtrend (idx 19): K={result_df['K'].iloc[19]}, D={result_df['D'].iloc[19]}")
        logger.info(f"After consolidation (idx 29): K={result_df['K'].iloc[29]}, D={result_df['D'].iloc[29]}")
        logger.info(f"After reversal (idx 39): K={result_df['K'].iloc[39]}, D={result_df['D'].iloc[39]}")
    
    def test_custom_parameters(self):
        """Test Stochastic calculation with custom parameters."""
        # Calculate with custom parameters
        result_df = calculate_stochastic(
            self.test_data.copy(), 
            k_period=5,
            d_period=3,
            k_col='CustomK',
            d_col='CustomD'
        )
        
        # Verify custom column names
        self.assertTrue('CustomK' in result_df.columns)
        self.assertTrue('CustomD' in result_df.columns)
        
        # Compare to default parameters
        default_result = calculate_stochastic(self.test_data.copy())
        
        # Check that shorter k_period and d_period result in more reactive indicators
        # Compare volatility during the volatile sideways period (40-49)
        custom_k_std = result_df['CustomK'].iloc[40:50].std()
        default_k_std = default_result['K'].iloc[40:50].std()
        
        custom_d_std = result_df['CustomD'].iloc[40:50].std()
        default_d_std = default_result['D'].iloc[40:50].std()
        
        logger.info(f"Stochastic std in volatile period - Custom K: {custom_k_std}, Default K: {default_k_std}")
        logger.info(f"Stochastic std in volatile period - Custom D: {custom_d_std}, Default D: {default_d_std}")
        
        # Shorter periods should be more reactive (higher standard deviation)
        self.assertGreater(custom_k_std, default_k_std)
        self.assertGreater(custom_d_std, default_d_std)
    
    
    def test_nan_handling(self):
        """Test that Stochastic calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[5:8, 'close'] = np.nan
        
        # Calculate Stochastic
        result_df = calculate_stochastic(nan_data)
        
        # Verify Stochastic is calculated and doesn't fail
        self.assertTrue('K' in result_df.columns)
        self.assertTrue('D' in result_df.columns)
        
        # Check that NaN values are handled properly
        logger.info(f"K values with NaN in input: {result_df['K'].iloc[0:15].tolist()}")
        logger.info(f"D values with NaN in input: {result_df['D'].iloc[0:15].tolist()}")


if __name__ == "__main__":
    unittest.main() 