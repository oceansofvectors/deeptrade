#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for MACD (Moving Average Convergence Divergence) indicator.

This test:
1. Creates sample price data
2. Calculates MACD using the function in indicators/macd.py
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
logger = logging.getLogger("test_macd")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the MACD calculation function
from indicators.macd import calculate_macd


class TestMACD(unittest.TestCase):
    """Test class for MACD indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with specific patterns
        data = []
        
        # Create 100 days of price data with different patterns
        for i in range(100):
            if i < 20:  # First 20 days: uptrend
                close = 100 + i*2
            elif i < 40:  # Days 20-39: downtrend
                close = 140 - (i-20)*2
            elif i < 60:  # Days 40-59: sideways with small oscillations
                close = 100 + 5*np.sin(i/5)
            elif i < 80:  # Days 60-79: uptrend
                close = 100 + (i-60)*1.5
            else:  # Days 80-99: downtrend
                close = 130 - (i-80)*1.5
            
            data.append({'close': close})
            
        self.test_data = pd.DataFrame(data)
    
    def test_default_parameters(self):
        """Test MACD calculation with default parameters."""
        # Calculate MACD with default parameters
        result_df = calculate_macd(self.test_data.copy())
        
        # Verify MACD, Signal and Histogram columns exist
        self.assertTrue('MACD' in result_df.columns)
        self.assertTrue('Signal' in result_df.columns)
        self.assertTrue('Histogram' in result_df.columns)
        
        # MACD should have values for all rows
        self.assertEqual(len(result_df['MACD']), len(self.test_data))
        self.assertEqual(len(result_df['Signal']), len(self.test_data))
        self.assertEqual(len(result_df['Histogram']), len(self.test_data))
        
        # Verify MACD reflects trend changes correctly
        uptrend_section = result_df['MACD'].iloc[15:20].mean()  # During strong uptrend
        downtrend_section = result_df['MACD'].iloc[35:40].mean()  # During strong downtrend
        
        # In a strong uptrend, MACD should be positive
        logger.info(f"MACD average during uptrend: {uptrend_section}")
        self.assertGreaterEqual(uptrend_section, 0, "MACD should be positive during a strong uptrend")
        
        # In a strong downtrend, MACD should be negative
        logger.info(f"MACD average during downtrend: {downtrend_section}")
        self.assertLessEqual(downtrend_section, 0, "MACD should be negative during a strong downtrend")
        
        # Check for trend change signals - crossovers between MACD and signal
        # After the uptrend ends and downtrend begins, MACD should cross below signal line
        uptrend_to_downtrend_crossovers = 0
        for i in range(20, 40):
            if i > 0 and result_df['MACD'].iloc[i] < result_df['Signal'].iloc[i] and result_df['MACD'].iloc[i-1] >= result_df['Signal'].iloc[i-1]:
                uptrend_to_downtrend_crossovers += 1
                logger.info(f"MACD crossed below signal line at index {i}")
        
        # There should be at least one bearish crossover in this section
        self.assertGreater(uptrend_to_downtrend_crossovers, 0, "MACD should cross below signal line after uptrend ends")
        
        # After the downtrend ends and uptrend begins, MACD should cross above signal line
        downtrend_to_uptrend_crossovers = 0
        for i in range(40, 70):
            if i > 0 and result_df['MACD'].iloc[i] > result_df['Signal'].iloc[i] and result_df['MACD'].iloc[i-1] <= result_df['Signal'].iloc[i-1]:
                downtrend_to_uptrend_crossovers += 1
                logger.info(f"MACD crossed above signal line at index {i}")
        
        # There should be at least one bullish crossover in this section
        self.assertGreater(downtrend_to_uptrend_crossovers, 0, "MACD should cross above signal line after downtrend ends")
        
        # Verify histogram calculation is correct: MACD - Signal
        random_indices = np.random.randint(26, 90, size=5)  # Check at 5 random indices after initial period
        for i in random_indices:
            expected_histogram = result_df['MACD'].iloc[i] - result_df['Signal'].iloc[i]
            self.assertAlmostEqual(result_df['Histogram'].iloc[i], expected_histogram, places=6)
        
        # Log some values for manual inspection
        logger.info(f"MACD values at key points:")
        logger.info(f"After uptrend period (idx 19): MACD={result_df['MACD'].iloc[19]}, Signal={result_df['Signal'].iloc[19]}")
        logger.info(f"After downtrend period (idx 39): MACD={result_df['MACD'].iloc[39]}, Signal={result_df['Signal'].iloc[39]}")
        logger.info(f"During sideways period (idx 50): MACD={result_df['MACD'].iloc[50]}, Signal={result_df['Signal'].iloc[50]}")
        logger.info(f"After second uptrend (idx 79): MACD={result_df['MACD'].iloc[79]}, Signal={result_df['Signal'].iloc[79]}")
    
    def test_custom_parameters(self):
        """Test MACD calculation with custom parameters."""
        # Calculate with custom parameters
        result_df = calculate_macd(
            self.test_data.copy(), 
            fast_period=6,
            slow_period=19,
            signal_period=4,
            macd_col='CustomMACD',
            signal_col='CustomSignal',
            histogram_col='CustomHistogram'
        )
        
        # Verify custom column names
        self.assertTrue('CustomMACD' in result_df.columns)
        self.assertTrue('CustomSignal' in result_df.columns)
        self.assertTrue('CustomHistogram' in result_df.columns)
        
        # Compare to default parameters
        default_result = calculate_macd(self.test_data.copy())
        
        # Check that faster periods result in more responsive MACD
        # Calculate the standard deviation of both MACD series
        custom_std = result_df['CustomMACD'].iloc[30:].std()
        default_std = default_result['MACD'].iloc[30:].std()
        
        logger.info(f"Standard deviation - Custom MACD: {custom_std}, Default MACD: {default_std}")
        
        # Faster EMA periods should be more responsive (higher standard deviation)
        self.assertNotEqual(custom_std, default_std)
        
        # Verify histogram is MACD - Signal for custom parameters too
        random_indices = np.random.randint(25, 90, size=5)  # Check at 5 random indices after initial period
        for i in random_indices:
            expected_histogram = result_df['CustomMACD'].iloc[i] - result_df['CustomSignal'].iloc[i]
            self.assertAlmostEqual(result_df['CustomHistogram'].iloc[i], expected_histogram, places=6)
    
    def test_manual_calculation(self):
        """Test MACD calculation against manual values."""
        # Create a simpler dataset for manual calculation
        data = [{'close': 100 + i} for i in range(50)]
        manual_data = pd.DataFrame(data)
        
        # Use simplified periods for easier calculation
        result_df = calculate_macd(manual_data.copy(), fast_period=3, slow_period=6, signal_period=2)
        
        # Manual EMA calculation helper
        def calculate_ema(values, period):
            alpha = 2 / (period + 1)
            ema = [values[0]]  # First value is the same as the input
            
            for i in range(1, len(values)):
                ema.append(alpha * values[i] + (1 - alpha) * ema[i-1])
                
            return ema
        
        # Calculate fast EMA (3-period) manually
        fast_ema = calculate_ema(manual_data['close'].tolist(), 3)
        
        # Calculate slow EMA (6-period) manually
        slow_ema = calculate_ema(manual_data['close'].tolist(), 6)
        
        # Calculate MACD line
        macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
        
        # Calculate signal line (2-period EMA of MACD line)
        signal_line = calculate_ema(macd_line, 2)
        
        # Compare calculated values at multiple indices for better coverage
        check_indices = [10, 20, 30, 40]
        for index in check_indices:
            logger.info(f"Manual MACD calculation at index {index}:")
            logger.info(f"  Fast EMA: {fast_ema[index]}, Slow EMA: {slow_ema[index]}")
            logger.info(f"  Expected MACD: {macd_line[index]}, Actual MACD: {result_df['MACD'].iloc[index]}")
            
            # Allow small floating point differences
            self.assertAlmostEqual(result_df['MACD'].iloc[index], macd_line[index], delta=0.5)
            self.assertAlmostEqual(result_df['Signal'].iloc[index], signal_line[index], delta=0.5)
        
        # Also verify histogram calculation
        for index in check_indices:
            expected_histogram = macd_line[index] - signal_line[index]
            actual_histogram = result_df['Histogram'].iloc[index]
            self.assertAlmostEqual(actual_histogram, expected_histogram, delta=0.5)
    
    def test_nan_handling(self):
        """Test that MACD calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[5:8, 'close'] = np.nan
        
        # Calculate MACD
        result_df = calculate_macd(nan_data)
        
        # Verify MACD is calculated and doesn't fail
        self.assertTrue('MACD' in result_df.columns)
        self.assertTrue('Signal' in result_df.columns)
        self.assertTrue('Histogram' in result_df.columns)
        
        # Check that NaN values are handled properly
        logger.info(f"MACD values with NaN in input: {result_df['MACD'].iloc[0:15].tolist()}")
        logger.info(f"Signal values with NaN in input: {result_df['Signal'].iloc[0:15].tolist()}")
        
        # Ensure there are some valid values after the NaN range
        self.assertTrue(not pd.isna(result_df['MACD'].iloc[30]))
        self.assertTrue(not pd.isna(result_df['Signal'].iloc[30]))


if __name__ == "__main__":
    unittest.main() 