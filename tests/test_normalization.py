import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import sys
from typing import Dict, List, Tuple

# Add parent directory to path to import from normalization module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from normalization import (
    scale_window, 
    normalize_data, 
    load_scaler, 
    get_standardized_column_names
)

class TestNormalization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        
        # Create DataFrame with some test indicators
        self.train_data = pd.DataFrame({
            'close': np.random.randn(70) * 10 + 100,
            'indicator1': np.random.randn(70) * 100,
            'indicator2': np.random.randn(70) * 50 - 10,
            'volume': np.random.randint(1000, 10000, 70)
        }, index=dates[:70])
        
        self.val_data = pd.DataFrame({
            'close': np.random.randn(15) * 10 + 100,
            'indicator1': np.random.randn(15) * 120,  # Slightly different range
            'indicator2': np.random.randn(15) * 60 - 12,  # Slightly different range
            'volume': np.random.randint(1000, 10000, 15)
        }, index=dates[70:85])
        
        self.test_data = pd.DataFrame({
            'close': np.random.randn(15) * 10 + 100,
            'indicator1': np.random.randn(15) * 150,  # Even more different range
            'indicator2': np.random.randn(15) * 70 - 15,  # Even more different range
            'volume': np.random.randint(1000, 10000, 15)
        }, index=dates[85:])
        
    def tearDown(self):
        # Remove temporary directory after tests
        shutil.rmtree(self.test_dir)
    
    def test_get_standardized_column_names(self):
        """Test that get_standardized_column_names correctly identifies columns to scale"""
        # Default skip list should exclude 'close' and 'volume'
        cols = get_standardized_column_names(self.train_data)
        self.assertIn('indicator1', cols)
        self.assertIn('indicator2', cols)
        self.assertNotIn('close', cols)
        self.assertNotIn('volume', cols)
        
        # Custom skip list
        cols = get_standardized_column_names(self.train_data, skip_columns=['indicator1'])
        self.assertNotIn('indicator1', cols)
        self.assertIn('indicator2', cols)
    
    def test_scale_window(self):
        """Test that scale_window correctly scales data across train, validation, and test sets"""
        cols_to_scale = ['indicator1', 'indicator2']
        
        # Record original values to verify transformation
        train_indicator1_mean = self.train_data['indicator1'].mean()
        train_indicator2_mean = self.train_data['indicator2'].mean()
        
        # Call scale_window
        scaler, train_scaled, val_scaled, test_scaled = scale_window(
            self.train_data,
            self.val_data,
            self.test_data,
            cols_to_scale,
            feature_range=(-1, 1),
            window_folder=self.test_dir
        )
        
        # Check that scaler was created
        self.assertIsNotNone(scaler)
        
        # Check that scaler was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "indicator_scaler.pkl")))
        
        # Check that data was transformed
        self.assertNotEqual(train_indicator1_mean, train_scaled['indicator1'].mean())
        self.assertNotEqual(train_indicator2_mean, train_scaled['indicator2'].mean())
        
        # Verify scaling is within range
        self.assertTrue(train_scaled['indicator1'].min() >= -1)
        self.assertTrue(train_scaled['indicator1'].max() <= 1)
        self.assertTrue(train_scaled['indicator2'].min() >= -1)
        self.assertTrue(train_scaled['indicator2'].max() <= 1)
        
        # Out-of-range values may exist in test data since it has more extreme values
        self.assertTrue(test_scaled['indicator1'].min() < -1 or test_scaled['indicator1'].max() > 1 or
                       test_scaled['indicator2'].min() < -1 or test_scaled['indicator2'].max() > 1)
        
        # Check that non-scaled columns are unchanged
        np.testing.assert_array_equal(self.train_data['close'].values, train_scaled['close'].values)
        np.testing.assert_array_equal(self.val_data['close'].values, val_scaled['close'].values)
        np.testing.assert_array_equal(self.test_data['close'].values, test_scaled['close'].values)
    
    def test_normalize_data(self):
        """Test normalize_data function"""
        cols_to_scale = ['indicator1', 'indicator2']
        
        # Test with new scaler
        normalized_data, scaler = normalize_data(
            self.train_data,
            cols_to_scale,
            feature_range=(-1, 1)
        )
        
        # Check that scaler was created
        self.assertIsNotNone(scaler)
        
        # Check that data was transformed
        self.assertNotEqual(
            self.train_data['indicator1'].mean(),
            normalized_data['indicator1'].mean()
        )
        
        # Verify scaling is within range
        self.assertTrue(normalized_data['indicator1'].min() >= -1)
        self.assertTrue(normalized_data['indicator1'].max() <= 1)
        
        # Test save and load
        scaler_path = os.path.join(self.test_dir, "test_scaler.pkl")
        
        # Test with save_path
        normalized_data2, scaler2 = normalize_data(
            self.train_data,
            cols_to_scale,
            feature_range=(-1, 1),
            save_path=scaler_path
        )
        
        # Check that scaler was saved
        self.assertTrue(os.path.exists(scaler_path))
        
        # Load the scaler
        loaded_scaler = load_scaler(scaler_path)
        self.assertIsNotNone(loaded_scaler)
        
        # Use loaded scaler to transform test data
        normalized_test, _ = normalize_data(
            self.test_data,
            cols_to_scale,
            scaler=loaded_scaler
        )
        
        # Check that test data was transformed
        self.assertNotEqual(
            self.test_data['indicator1'].mean(),
            normalized_test['indicator1'].mean()
        )
    
    def test_load_scaler_nonexistent_file(self):
        """Test load_scaler with nonexistent file"""
        scaler = load_scaler(os.path.join(self.test_dir, "nonexistent.pkl"))
        self.assertIsNone(scaler)

if __name__ == '__main__':
    unittest.main() 