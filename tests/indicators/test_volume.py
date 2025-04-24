#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Volume indicator.

This test:
1. Creates sample price and volume data
2. Calculates volume ratio using the function in indicators/volume.py
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
logger = logging.getLogger("test_volume")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the volume calculation function
from indicators.volume import calculate_volume_indicator


class TestVolumeIndicator(unittest.TestCase):
    """Test class for volume indicator calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with price and varied volume
        np.random.seed(42)  # For reproducibility
        
        # Creating volume with some patterns: low, normal, spiking
        volumes = []
        for i in range(50):
            if i < 20:
                # Normal volume with some randomness
                volumes.append(1000 + np.random.randint(-200, 200))
            elif i < 25:
                # Volume spike
                volumes.append(2000 + np.random.randint(-200, 200))
            elif i < 30:
                # Volume spike fading
                volumes.append(1500 + np.random.randint(-200, 200))
            else:
                # Back to normal with some randomness
                volumes.append(1000 + np.random.randint(-200, 200))
                
        self.test_data = pd.DataFrame({
            'high': np.random.randint(100, 110, size=len(volumes)),
            'low': np.random.randint(90, 100, size=len(volumes)),
            'close': np.random.randint(95, 105, size=len(volumes)),
            'volume': volumes
        })
    
    def test_default_parameters(self):
        """Test volume indicator calculation with default parameters."""
        # Calculate volume indicator with default parameters
        result_df = calculate_volume_indicator(self.test_data.copy())
        
        # Verify volume indicator column exists
        self.assertTrue('VOLUME_RATIO' in result_df.columns)
        
        # Volume indicator should have values for all rows (though initial values will be NaN/1)
        self.assertEqual(len(result_df['VOLUME_RATIO']), len(self.test_data))
        
        # Check volume spike is reflected in the indicator
        # First 20 points should have ratio around 1
        # Points 20-25 should have ratios significantly above 1
        early_avg = result_df['VOLUME_RATIO'][21:25].mean()
        logger.info(f"Volume ratio during spike: {early_avg}")
        
        # The spike should be noticeably higher than the average
        self.assertTrue(early_avg > 1.3)
        
        # Log some values for manual inspection
        logger.info(f"Volume ratios: {result_df['VOLUME_RATIO'].iloc[[5, 15, 22, 28, 35]].tolist()}")
    
    def test_custom_parameters(self):
        """Test volume indicator calculation with custom parameters."""
        # Calculate with custom length and column name
        result_df = calculate_volume_indicator(
            self.test_data.copy(), 
            ma_length=10,
            target_col='CustomVolumeRatio'
        )
        
        # Verify custom column exists
        self.assertTrue('CustomVolumeRatio' in result_df.columns)
        
        # All rows should have values
        self.assertEqual(len(result_df['CustomVolumeRatio']), len(self.test_data))
        
        # Log some values for manual inspection
        logger.info(f"Volume ratios with custom parameters: {result_df['CustomVolumeRatio'].iloc[[5, 15, 22, 28, 35]].tolist()}")
    
    def test_missing_volume(self):
        """Test volume indicator calculation with missing volume data."""
        # Create data without volume
        no_volume_data = self.test_data.copy().drop('volume', axis=1)
        
        # Calculate volume indicator
        result_df = calculate_volume_indicator(no_volume_data)
        
        # Verify indicator is calculated and doesn't fail
        self.assertTrue('VOLUME_RATIO' in result_df.columns)
        
        # Since volume is created with default values, after moving average period,
        # all ratios should be approximately 1.0
        stabilized_values = result_df['VOLUME_RATIO'].iloc[30:40]  
        self.assertTrue(all(0.99 <= val <= 1.01 for val in stabilized_values))
        
        # Log some values for manual inspection
        logger.info(f"Volume ratios with generated volume: {result_df['VOLUME_RATIO'].iloc[[25, 30, 35, 40, 45]].tolist()}")
    
    def test_nan_handling(self):
        """Test that volume indicator calculation handles NaN values properly."""
        # Create data with NaN values in volume
        nan_data = self.test_data.copy()
        nan_data.loc[3:5, 'volume'] = np.nan
        
        # Calculate volume indicator
        result_df = calculate_volume_indicator(nan_data)
        
        # Verify indicator is calculated and doesn't fail
        self.assertTrue('VOLUME_RATIO' in result_df.columns)
        
        # Check that NaN rows are handled
        logger.info(f"Volume ratios with NaN in input: {result_df['VOLUME_RATIO'].iloc[0:10].tolist()}")


if __name__ == "__main__":
    unittest.main() 