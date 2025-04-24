import unittest
import pandas as pd
import numpy as np
import logging
from indicators.sma import calculate_sma

# Configure logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSMAIndicator(unittest.TestCase):
    """
    Test class for the Simple Moving Average (SMA) indicator
    """
    
    def setUp(self):
        """
        Set up test data with known patterns for SMA calculation verification
        """
        # Create a date range for the test data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create test data with known patterns
        prices = [
            # Flat trend (10 days)
            *[100.0] * 10,
            
            # Uptrend (15 days)
            *[100.0 + i for i in range(1, 16)],
            
            # Downtrend (15 days)
            *[115.0 - i for i in range(1, 16)],
            
            # Volatile pattern (10 days)
            105.0, 110.0, 103.0, 107.0, 112.0, 
            108.0, 113.0, 106.0, 104.0, 109.0
        ]
        
        # Create DataFrame with prices
        self.df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p + 1.0 for p in prices],
            'low': [p - 1.0 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, size=len(prices))
        })
        
        # Set date as index
        self.df.set_index('date', inplace=True)
        
        # Log the first few rows of the test data
        logger.info(f"Test data head:\n{self.df.head()}")
        
    def test_default_parameters(self):
        """
        Test SMA calculation with default parameters (length=20)
        """
        # Calculate SMA with default parameters
        result_df = calculate_sma(self.df.copy())
        
        # Check that the SMA column exists
        self.assertIn('SMA', result_df.columns)
        
        # Check that there are no NaN values in the SMA column
        self.assertFalse(result_df['SMA'].isnull().any())
        
        # Verify SMA calculation for specific points
        # For the first 20 rows, SMA should be equal to close (fillna behavior)
        for i in range(20):
            self.assertEqual(result_df.iloc[i]['SMA'], result_df.iloc[i]['close'])
        
        # After 20 rows, check a few points against manual calculation
        # At index 30, SMA should be the average of previous 20 close prices
        expected_sma = self.df['close'].iloc[11:31].mean()
        self.assertAlmostEqual(result_df.iloc[30]['SMA'], expected_sma, places=4)
        
        # Log the SMA values for manual inspection
        logger.info(f"SMA values (default parameters):\n{result_df['SMA'].tail()}")
        
    def test_custom_parameters(self):
        """
        Test SMA calculation with custom parameters
        """
        # Calculate SMA with custom length
        custom_length = 10
        custom_column = 'Custom_SMA'
        result_df = calculate_sma(self.df.copy(), length=custom_length, target_col=custom_column)
        
        # Check that the custom SMA column exists
        self.assertIn(custom_column, result_df.columns)
        
        # Check that there are no NaN values in the custom SMA column
        self.assertFalse(result_df[custom_column].isnull().any())
        
        # Verify SMA calculation for specific points
        # For the first 10 rows, SMA should be equal to close (fillna behavior)
        for i in range(10):
            self.assertEqual(result_df.iloc[i][custom_column], result_df.iloc[i]['close'])
        
        # After 10 rows, check a few points against manual calculation
        # At index 20, SMA should be the average of previous 10 close prices
        expected_sma = self.df['close'].iloc[11:21].mean()
        self.assertAlmostEqual(result_df.iloc[20][custom_column], expected_sma, places=4)
        
        # Log the custom SMA values for manual inspection
        logger.info(f"Custom SMA values (length={custom_length}):\n{result_df[custom_column].tail()}")
        
    def test_manual_calculation(self):
        """
        Test SMA calculation against manually calculated values
        """
        # Calculate SMA with a small length for easier manual verification
        length = 5
        column = 'SMA_5'
        result_df = calculate_sma(self.df.copy(), length=length, target_col=column)
        
        # Sample index to check
        idx = 30
        
        # Manually calculate the SMA
        manual_sma = sum(self.df['close'].iloc[idx-length:idx]) / length
        
        # Check that the calculated SMA matches our manual calculation
        self.assertAlmostEqual(result_df.iloc[idx][column], manual_sma, places=4)
        
        # Log the comparison
        logger.info(f"At index {idx}: Manual SMA = {manual_sma}, Calculated SMA = {result_df.iloc[idx][column]}")
        
    def test_edge_cases(self):
        """
        Test SMA calculation with edge cases
        """
        # Test with very short length
        short_length = 1
        short_column = 'SMA_1'
        short_df = calculate_sma(self.df.copy(), length=short_length, target_col=short_column)
        
        # For length=1, SMA should be equal to close price
        for i in range(len(self.df)):
            self.assertEqual(short_df.iloc[i][short_column], short_df.iloc[i]['close'])
        
        # Test with length longer than DataFrame
        long_length = 100
        long_column = 'SMA_100'
        long_df = calculate_sma(self.df.copy(), length=long_length, target_col=long_column)
        
        # For length > len(df), SMA should be equal to close price (fillna behavior)
        for i in range(len(self.df)):
            self.assertEqual(long_df.iloc[i][long_column], long_df.iloc[i]['close'])
        
    def test_nan_handling(self):
        """
        Test SMA calculation with NaN values in the data
        """
        # Create a copy of the test data with some NaN values
        nan_df = self.df.copy()
        nan_df.loc[nan_df.index[5:10], 'close'] = np.nan
        
        # Calculate SMA
        result_df = calculate_sma(nan_df.copy())
        
        # Check that there are no NaN values in the SMA column (fillna should handle them)
        self.assertFalse(result_df['SMA'].isnull().any())
        
        # Log the result
        logger.info(f"SMA values with NaN in data:\n{result_df['SMA'].head(15)}")
        
if __name__ == '__main__':
    unittest.main() 