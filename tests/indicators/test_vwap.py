#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for VWAP (Volume Weighted Average Price) indicator.

This test:
1. Creates sample price and volume data
2. Calculates VWAP using the function in indicators/vwap.py
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
logger = logging.getLogger("test_vwap")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the VWAP calculation function
from indicators.vwap import calculate_vwap


class TestVWAP(unittest.TestCase):
    """Test class for VWAP indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price and volume data
        np.random.seed(42)  # For reproducibility
        
        # Create dates for a trading session
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=390, freq='1min')
        
        # Create price data with a general trend and some noise
        close_prices = np.linspace(100, 110, 390) + np.random.normal(0, 1, 390)
        high_prices = close_prices + np.random.uniform(0.1, 0.5, 390)
        low_prices = close_prices - np.random.uniform(0.1, 0.5, 390)
        open_prices = close_prices - np.random.uniform(-0.3, 0.3, 390)
        
        # Create volume data with some spikes
        volume = np.random.normal(1000, 200, 390)
        # Add some volume spikes
        volume[100:105] = volume[100:105] * 3
        volume[200:210] = volume[200:210] * 2
        volume[300:305] = volume[300:305] * 4
        
        # Create DataFrame with OHLCV data
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        # Set timestamp as index
        self.test_data.set_index('timestamp', inplace=True)
    
    def test_default_parameters(self):
        """Test VWAP calculation with default parameters."""
        # Calculate VWAP
        result_df = calculate_vwap(self.test_data.copy())
        
        # Verify VWAP column exists
        self.assertTrue('VWAP' in result_df.columns)
        
        # VWAP should have values for all rows
        self.assertEqual(len(result_df['VWAP']), len(self.test_data))
        
        # First value should be calculated
        self.assertFalse(pd.isna(result_df['VWAP'].iloc[0]))
        
        # VWAP at the beginning should be close to the first typical price
        first_typical_price = (self.test_data['high'].iloc[0] + self.test_data['low'].iloc[0] + self.test_data['close'].iloc[0]) / 3
        self.assertAlmostEqual(result_df['VWAP'].iloc[0], first_typical_price, delta=0.1)
        
        # VWAP should be between min and max prices
        min_price = self.test_data['low'].min()
        max_price = self.test_data['high'].max()
        for vwap in result_df['VWAP']:
            self.assertTrue(min_price <= vwap <= max_price)
        
        # Log some values for manual inspection
        logger.info(f"First VWAP value: {result_df['VWAP'].iloc[0]}")
        logger.info(f"VWAP after volume spike (index 105): {result_df['VWAP'].iloc[105]}")
        logger.info(f"VWAP after another volume spike (index 210): {result_df['VWAP'].iloc[210]}")
        logger.info(f"Final VWAP value: {result_df['VWAP'].iloc[-1]}")
    
    def test_custom_parameters(self):
        """Test VWAP calculation with custom parameters."""
        # Calculate VWAP with custom column name
        result_df = calculate_vwap(self.test_data.copy(), vwap_col='CustomVWAP')
        
        # Verify custom column name
        self.assertTrue('CustomVWAP' in result_df.columns)
        self.assertFalse('VWAP' in result_df.columns)
        
        # Verify values are still valid
        self.assertEqual(len(result_df['CustomVWAP']), len(self.test_data))
        min_price = self.test_data['low'].min()
        max_price = self.test_data['high'].max()
        for vwap in result_df['CustomVWAP']:
            self.assertTrue(min_price <= vwap <= max_price)
    
    def test_manual_calculation(self):
        """Test VWAP calculation against manual values."""
        # Create a smaller sample for manual calculation
        small_data = self.test_data.head(10).copy()
        result_df = calculate_vwap(small_data)
        
        # Calculate VWAP manually
        cumulative_tp_vol = 0
        cumulative_vol = 0
        manual_vwap = []
        
        for i in range(len(small_data)):
            row = small_data.iloc[i]
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            tp_vol = typical_price * row['volume']
            
            cumulative_tp_vol += tp_vol
            cumulative_vol += row['volume']
            
            vwap = cumulative_tp_vol / cumulative_vol
            manual_vwap.append(vwap)
        
        # Compare manual and calculated values
        for i in range(len(small_data)):
            self.assertAlmostEqual(result_df['VWAP'].iloc[i], manual_vwap[i], delta=0.001)
        
        logger.info(f"Manual VWAP calculation matches calculated values for first 10 rows")
    
    def test_intraday_reset(self):
        """Test VWAP resets at the start of each trading day."""
        # Create multi-day data to test daily reset
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=390 * 3, freq='1min')
        np.random.seed(42)  # For reproducibility
        
        # Create price data
        close_prices = []
        high_prices = []
        low_prices = []
        open_prices = []
        volume = []
        
        # Create 3 days of data
        for day in range(3):
            day_close = np.linspace(100 + day*5, 110 + day*5, 390) + np.random.normal(0, 1, 390)
            day_high = day_close + np.random.uniform(0.1, 0.5, 390)
            day_low = day_close - np.random.uniform(0.1, 0.5, 390)
            day_open = day_close - np.random.uniform(-0.3, 0.3, 390)
            day_volume = np.random.normal(1000, 200, 390)
            
            close_prices.extend(day_close)
            high_prices.extend(day_high)
            low_prices.extend(day_low)
            open_prices.extend(day_open)
            volume.extend(day_volume)
        
        # Create DataFrame with OHLCV data
        multi_day_data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        # Set timestamp as index
        multi_day_data.set_index('timestamp', inplace=True)
        
        # Calculate VWAP
        result_df = calculate_vwap(multi_day_data.copy())
        
        # Check VWAP for each day - values at start of day should be different from end of previous day
        day_boundaries = [0, 390, 780]
        
        # Log values at day boundaries for inspection
        for i in range(1, len(day_boundaries)):
            prev_day_end = day_boundaries[i] - 1
            new_day_start = day_boundaries[i]
            
            logger.info(f"End of day {i} VWAP: {result_df['VWAP'].iloc[prev_day_end]}")
            logger.info(f"Start of day {i+1} VWAP: {result_df['VWAP'].iloc[new_day_start]}")
            
            # The VWAP at the start of a new day should be different from the end of the previous day
            self.assertNotAlmostEqual(result_df['VWAP'].iloc[prev_day_end], result_df['VWAP'].iloc[new_day_start], delta=0.1)
            
            # The VWAP at the start of a day should be close to the typical price of the first bar
            first_bar_typical_price = (multi_day_data['high'].iloc[new_day_start] + multi_day_data['low'].iloc[new_day_start] + multi_day_data['close'].iloc[new_day_start]) / 3
            self.assertAlmostEqual(result_df['VWAP'].iloc[new_day_start], first_bar_typical_price, delta=0.1)
    
    def test_nan_handling(self):
        """Test that VWAP calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[nan_data.index[5:8], 'close'] = np.nan
        nan_data.loc[nan_data.index[15:18], 'volume'] = np.nan
        
        # Calculate VWAP
        result_df = calculate_vwap(nan_data)
        
        # Verify VWAP is calculated and doesn't fail
        self.assertTrue('VWAP' in result_df.columns)
        
        # Check that NaN values are handled properly
        logger.info(f"VWAP values with NaN in input:")
        logger.info(f"  Around NaN close values: {result_df['VWAP'].iloc[3:10].values}")
        logger.info(f"  Around NaN volume values: {result_df['VWAP'].iloc[13:20].values}")
        
        # Ensure there are valid values after NaN range
        self.assertTrue(not pd.isna(result_df['VWAP'].iloc[30]))


if __name__ == "__main__":
    unittest.main() 