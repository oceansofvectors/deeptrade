"""
Data debugging utility to analyze CSV files and diagnose issues.
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_csv(filepath='data/nq.csv'):
    """
    Analyze a CSV file and print diagnostic information.
    
    Args:
        filepath: Path to the CSV file
    """
    logger.info(f"Analyzing CSV file: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return
    
    # Check file size
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
    logger.info(f"File size: {file_size:.2f} MB")
    
    # Try to read the file
    try:
        # First just count the lines
        with open(filepath, 'r') as f:
            line_count = sum(1 for _ in f)
        logger.info(f"Line count: {line_count}")
        
        # Try to read with pandas
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded CSV with pandas")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check for missing values
        missing_values = df.isna().sum()
        missing_pct = (missing_values / len(df)) * 100
        
        logger.info("\nMissing values by column:")
        for col, missing in zip(missing_values.index, missing_values):
            if missing > 0:
                logger.info(f"  {col}: {missing} ({missing_pct[col]:.2f}%)")
        
        # Check column types
        logger.info("\nColumn data types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Check for infinite values
        if (df.select_dtypes(include=['float']).isin([float('inf'), float('-inf')])).any().any():
            logger.warning("DataFrame contains infinite values")
        
        # Print sample rows
        logger.info("\nFirst 5 rows:")
        print(df.head())
        
        if len(df) > 5:
            logger.info("\nLast 5 rows:")
            print(df.tail())
            
        # Check if 'time' column exists and can be converted to datetime
        if 'time' in df.columns:
            try:
                # Try to parse time column - check if it's unix timestamp or string
                if pd.api.types.is_numeric_dtype(df['time']):
                    time_parsed = pd.to_datetime(df['time'], unit='s')
                    logger.info("\nTime column appears to be unix timestamps")
                    logger.info(f"Time range: {time_parsed.min()} to {time_parsed.max()}")
                else:
                    time_parsed = pd.to_datetime(df['time'])
                    logger.info("\nTime column appears to be datetime strings")
                    logger.info(f"Time range: {time_parsed.min()} to {time_parsed.max()}")
            except Exception as e:
                logger.error(f"Error parsing time column: {e}")
        else:
            logger.warning("No 'time' column found in the data")
            
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"DataFrame contains {duplicate_count} duplicate rows")
            
        return df
        
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file - it may be malformed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading the CSV: {e}")

if __name__ == "__main__":
    analyze_csv() 