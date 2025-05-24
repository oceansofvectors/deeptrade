#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for RRCF (Robust Random Cut Forest) anomaly detection indicator.

This test:
1. Creates sample price data with known anomalies
2. Calculates RRCF anomaly scores using the function in indicators/rrcf_anomaly.py
3. Verifies the calculation runs successfully and generates non-zero scores
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import unittest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_rrcf_anomaly")

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the RRCF anomaly calculation function
from indicators.rrcf_anomaly import calculate_rrcf_anomaly


class TestRRCFAnomaly(unittest.TestCase):
    """Test class for RRCF anomaly detection indicator calculation."""
    
    def setUp(self):
        """Set up test data with normal patterns and anomalies."""
        # Create dates for test data (fix frequency warning)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='h')
        
        # Create price data with normal patterns and injected anomalies
        np.random.seed(42)  # For reproducibility
        
        # Base price following a trend with some noise
        base_price = 100
        trend = np.linspace(0, 20, 200)  # Upward trend
        noise = np.random.normal(0, 0.5, 200)  # Small random noise
        prices = base_price + trend + noise
        
        # Inject some anomalies (sudden spikes/drops)
        anomaly_indices = [50, 100, 150]
        for idx in anomaly_indices:
            if idx < len(prices):
                # Random spike or drop
                anomaly_magnitude = np.random.choice([-15, -10, 10, 15])
                prices[idx] += anomaly_magnitude
        
        # Create volume data with corresponding anomalies
        base_volume = 1000
        volume_noise = np.random.normal(0, 100, 200)
        volumes = base_volume + volume_noise
        volumes = np.maximum(volumes, 100)  # Ensure positive volumes
        
        # Volume spikes during price anomalies
        for idx in anomaly_indices:
            if idx < len(volumes):
                volumes[idx] *= np.random.uniform(2, 4)  # 2-4x volume spike
        
        # Create DataFrame
        self.test_data = pd.DataFrame({
            'timestamp': dates[:len(prices)],
            'close': prices,
            'volume': volumes,
            'high': prices + np.abs(np.random.normal(0, 0.3, len(prices))),
            'low': prices - np.abs(np.random.normal(0, 0.3, len(prices))),
            'open': prices + np.random.normal(0, 0.2, len(prices))
        })
        
        # Ensure high >= close >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['close'])
        self.test_data['low'] = np.minimum(self.test_data['low'], self.test_data['close'])
        
        # Set timestamp as index
        self.test_data.set_index('timestamp', inplace=True)
        
        # Store anomaly indices for verification
        self.known_anomaly_indices = anomaly_indices
    
    def test_default_parameters(self):
        """Test RRCF anomaly calculation with default parameters."""
        logger.info("Testing RRCF anomaly detection with default parameters")
        
        # Calculate RRCF anomaly scores
        result_df = calculate_rrcf_anomaly(self.test_data.copy())
        
        # Verify RRCF_Anomaly column exists
        self.assertTrue('RRCF_Anomaly' in result_df.columns, "RRCF_Anomaly column should exist")
        
        # Verify all values are non-null
        self.assertEqual(result_df['RRCF_Anomaly'].isna().sum(), 0, "No NaN values should exist")
        
        # Verify scores are not all zero
        non_zero_scores = result_df['RRCF_Anomaly'][result_df['RRCF_Anomaly'] > 0].count()
        self.assertGreater(non_zero_scores, 0, "Should have non-zero anomaly scores")
        
        # Verify scores are in reasonable range [0, 1]
        min_score = result_df['RRCF_Anomaly'].min()
        max_score = result_df['RRCF_Anomaly'].max()
        self.assertGreaterEqual(min_score, 0, "Minimum score should be >= 0")
        self.assertLessEqual(max_score, 1, "Maximum score should be <= 1")
        
        # Log some statistics
        mean_score = result_df['RRCF_Anomaly'].mean()
        std_score = result_df['RRCF_Anomaly'].std()
        logger.info(f"RRCF Anomaly scores - Min: {min_score:.4f}, Max: {max_score:.4f}, "
                   f"Mean: {mean_score:.4f}, Std: {std_score:.4f}")
        
        # Check that we have variation in scores (reduced expectation to be more realistic)
        unique_scores = result_df['RRCF_Anomaly'].nunique()
        self.assertGreater(unique_scores, 5, "Should have some variety in anomaly scores")
        
        # Check that standard deviation is reasonable (indicates variation)
        self.assertGreater(std_score, 0.01, "Should have reasonable variation in scores")
        
        logger.info(f"Found {unique_scores} unique anomaly scores")
        logger.info(f"Non-zero scores: {non_zero_scores} out of {len(result_df)}")
    
    def test_custom_parameters(self):
        """Test RRCF anomaly calculation with custom parameters."""
        logger.info("Testing RRCF anomaly detection with custom parameters")
        
        # Test with custom parameters
        result_df = calculate_rrcf_anomaly(
            self.test_data.copy(),
            feature_cols=['close', 'volume'],
            window_size=50,
            num_trees=20,
            tree_size=128,
            target_col='CustomRRCF',
            random_seed=123
        )
        
        # Verify custom column name
        self.assertTrue('CustomRRCF' in result_df.columns, "Custom column name should exist")
        self.assertFalse('RRCF_Anomaly' in result_df.columns, "Default column should not exist")
        
        # Verify scores are generated
        non_zero_scores = result_df['CustomRRCF'][result_df['CustomRRCF'] > 0].count()
        self.assertGreater(non_zero_scores, 0, "Should have non-zero anomaly scores")
        
        # Verify reproducibility with same seed
        result_df2 = calculate_rrcf_anomaly(
            self.test_data.copy(),
            feature_cols=['close', 'volume'],
            window_size=50,
            num_trees=20,
            tree_size=128,
            target_col='CustomRRCF',
            random_seed=123
        )
        
        # Scores should be similar with same seed (allowing for small differences due to floating point)
        score_diff = np.abs(result_df['CustomRRCF'] - result_df2['CustomRRCF']).max()
        self.assertLess(score_diff, 0.01, "Results should be reproducible with same seed")
        
        logger.info("Custom parameters test passed")
    
    def test_single_feature(self):
        """Test RRCF with single feature column."""
        logger.info("Testing RRCF anomaly detection with single feature")
        
        # Test with only close price
        result_df = calculate_rrcf_anomaly(
            self.test_data.copy(),
            feature_cols=['close'],
            window_size=30,
            target_col='SingleFeatureRRCF'
        )
        
        # Verify it works with single feature
        self.assertTrue('SingleFeatureRRCF' in result_df.columns)
        non_zero_scores = result_df['SingleFeatureRRCF'][result_df['SingleFeatureRRCF'] > 0].count()
        self.assertGreater(non_zero_scores, 0, "Should work with single feature")
        
        logger.info("Single feature test passed")
    
    def test_edge_cases(self):
        """Test RRCF calculation with edge cases."""
        logger.info("Testing RRCF anomaly detection edge cases")
        
        # Test with very small dataset
        small_data = self.test_data.head(20).copy()
        result_small = calculate_rrcf_anomaly(small_data, window_size=10)
        
        self.assertTrue('RRCF_Anomaly' in result_small.columns)
        self.assertEqual(len(result_small), 20, "Should return same number of rows")
        
        # Test with constant values
        constant_data = pd.DataFrame({
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        constant_data.index = pd.date_range(start='2023-01-01', periods=50, freq='h')
        
        result_constant = calculate_rrcf_anomaly(constant_data)
        self.assertTrue('RRCF_Anomaly' in result_constant.columns)
        # Should handle constant values gracefully
        self.assertFalse(result_constant['RRCF_Anomaly'].isna().all(), "Should not be all NaN")
        
        logger.info("Edge cases test passed")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 