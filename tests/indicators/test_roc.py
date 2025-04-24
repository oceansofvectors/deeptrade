#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for ROC (Rate of Change) indicator.

This test:
1. Creates sample price data
2. Calculates ROC using the function in indicators/roc.py
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
logger = logging.getLogger("test_roc")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the ROC calculation function
from indicators.roc import calculate_roc


class TestROC(unittest.TestCase):
    """Test class for ROC indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample price DataFrame with specific percentage changes
        
        # Start with 100 and create specific changes
        prices = [100]  # Initial price
        
        # Create 40 days of price data with known percentage changes
        changes = [
            # First 10 days: flat
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # Next 10 days: consistent 1% daily gain
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            # Next 10 days: consistent 1% daily loss
            -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01,
            # Last 10 days: alternating gains and losses
            0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02
        ]
        
        # Calculate prices from the changes
        for change in changes:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
        self.test_data = pd.DataFrame({
            'close': prices
        })
    
    def test_default_parameters(self):
        """Test ROC calculation with default parameters."""
        # Default ROC length is 10
        result_df = calculate_roc(self.test_data.copy())
        
        # Verify ROC column exists
        self.assertTrue('ROC' in result_df.columns)
        
        # ROC should have values for all rows
        self.assertEqual(len(result_df['ROC']), len(self.test_data))
        
        # Check ROC values at specific points
        # After flat period (index 10), ROC should be close to 0
        self.assertAlmostEqual(result_df['ROC'].iloc[10], 0, delta=0.01)
        
        # After 10 days of 1% gains (index 20), ROC should be close to 10%
        # Formula: ((1.01^10) - 1) * 100 ≈ 10.46%
        self.assertAlmostEqual(result_df['ROC'].iloc[20], 10.46, delta=0.5)
        
        # After 10 days of 1% losses (index 30), ROC should be close to -10%
        # Formula: ((0.99^10) - 1) * 100 ≈ -9.56%
        self.assertAlmostEqual(result_df['ROC'].iloc[30], -9.56, delta=0.5)
        
        # Log some values for manual inspection
        logger.info(f"ROC values at key points:")
        logger.info(f"After flat period (idx 10): {result_df['ROC'].iloc[10]}")
        logger.info(f"After 10 days of 1% gains (idx 20): {result_df['ROC'].iloc[20]}")
        logger.info(f"After 10 days of 1% losses (idx 30): {result_df['ROC'].iloc[30]}")
        logger.info(f"After alternating period (idx 40): {result_df['ROC'].iloc[40]}")
    
    def test_custom_parameters(self):
        """Test ROC calculation with custom parameters."""
        # Calculate with custom length and column name
        result_df = calculate_roc(
            self.test_data.copy(), 
            length=5,
            target_col='CustomROC'
        )
        
        # Verify custom ROC column exists
        self.assertTrue('CustomROC' in result_df.columns)
        
        # ROC should have values for all rows
        self.assertEqual(len(result_df['CustomROC']), len(self.test_data))
        
        # Check ROC values at specific points with 5-day ROC
        # After 5 days of 1% gains (index 15), ROC should be close to 5%
        # Formula: ((1.01^5) - 1) * 100 ≈ 5.1%
        self.assertAlmostEqual(result_df['CustomROC'].iloc[15], 5.1, delta=0.2)
        
        # After 5 days of 1% losses (index 25), ROC should be close to -5%
        # Formula: ((0.99^5) - 1) * 100 ≈ -4.9%
        self.assertAlmostEqual(result_df['CustomROC'].iloc[25], -4.9, delta=0.2)
        
        # Log some values for manual inspection
        logger.info(f"5-day ROC values: {result_df['CustomROC'].iloc[[5, 15, 25, 35]].tolist()}")
        
        # Compare to default length (10)
        default_result = calculate_roc(self.test_data.copy())
        
        # Shorter period ROC should be more responsive and less smooth
        custom_volatility = result_df['CustomROC'].iloc[30:40].std()
        default_volatility = default_result['ROC'].iloc[30:40].std()
        
        logger.info(f"ROC volatility during alternating period - Custom (5): {custom_volatility}, Default (10): {default_volatility}")
        # Shorter period should have higher volatility
        self.assertGreater(custom_volatility, default_volatility)
    
    def test_manual_calculation(self):
        """Test ROC calculation against manual calculation."""
        # Calculate 10-period ROC
        result_df = calculate_roc(self.test_data.copy(), length=10)
        
        # Manually calculate 10-period ROC at a few points
        for i in range(15, 35, 5):
            if i >= 10:
                expected_roc = ((self.test_data['close'].iloc[i] / self.test_data['close'].iloc[i-10]) - 1) * 100
                actual_roc = result_df['ROC'].iloc[i]
                
                # Check if calculation matches
                self.assertAlmostEqual(actual_roc, expected_roc, delta=0.01)
                logger.info(f"ROC at idx {i} - Actual: {actual_roc}, Expected: {expected_roc}")
    
    def test_nan_handling(self):
        """Test that ROC calculation handles NaN values properly."""
        # Create data with NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[5:8, 'close'] = np.nan
        
        # Calculate ROC
        result_df = calculate_roc(nan_data)
        
        # Verify ROC is calculated and doesn't fail
        self.assertTrue('ROC' in result_df.columns)
        
        # Check that NaN rows are handled
        logger.info(f"ROC values with NaN in input: {result_df['ROC'].iloc[0:15].tolist()}")


if __name__ == "__main__":
    unittest.main() 