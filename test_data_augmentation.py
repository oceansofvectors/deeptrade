#!/usr/bin/env python3
"""
Test script for data augmentation functionality.
This script demonstrates how the data augmentation works and validates the output.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from data_augmentation import DataAugmenter
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create a sample dataset for testing data augmentation."""
    
    # Create synthetic price data that resembles trading data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic time series with trend and noise
    time_index = pd.date_range('2024-01-01', periods=n_samples, freq='5T')
    
    # Generate price data with trend
    trend = np.linspace(4000, 4200, n_samples)  # Upward trend
    noise = np.random.normal(0, 5, n_samples)   # Price noise
    close_prices = trend + noise
    
    # Create OHLC data
    high_noise = np.abs(np.random.normal(0, 2, n_samples))
    low_noise = np.abs(np.random.normal(0, 2, n_samples))
    
    data = pd.DataFrame({
        'open': np.roll(close_prices, 1),  # Previous close as open
        'high': close_prices + high_noise,
        'low': close_prices - low_noise,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, n_samples),
        'close_norm': close_prices / close_prices[0],  # Normalized close
        'RSI': np.random.uniform(20, 80, n_samples),   # Random RSI values
        'MACD': np.random.normal(0, 1, n_samples),     # Random MACD
        'position': np.zeros(n_samples)                # No position initially
    }, index=time_index)
    
    # Fix OHLC relationships
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_jittering(augmenter, sample_data):
    """Test jittering functionality."""
    logger.info("Testing jittering functionality...")
    
    jitter_config = {
        'price_noise_std': 0.001,
        'indicator_noise_std': 0.02,
        'volume_noise_std': 0.05
    }
    
    # Apply jittering
    jittered_data = augmenter._apply_jittering(sample_data.copy(), jitter_config)
    
    # Validate that data was modified but structure preserved
    assert jittered_data.shape == sample_data.shape, "Shape should be preserved"
    assert list(jittered_data.columns) == list(sample_data.columns), "Columns should be preserved"
    
    # Check that prices were modified (but not too much)
    price_diff = np.abs(jittered_data['close'] - sample_data['close'])
    assert price_diff.mean() > 0, "Prices should be modified"
    assert price_diff.mean() < sample_data['close'].std() * 0.1, "Changes should be reasonable"
    
    # Validate OHLC relationships are maintained
    assert (jittered_data['high'] >= jittered_data['close']).all(), "High should be >= Close"
    assert (jittered_data['high'] >= jittered_data['open']).all(), "High should be >= Open"
    assert (jittered_data['low'] <= jittered_data['close']).all(), "Low should be <= Close"
    assert (jittered_data['low'] <= jittered_data['open']).all(), "Low should be <= Open"
    
    logger.info("✓ Jittering test passed")
    return jittered_data

def test_cutpaste(augmenter, sample_data):
    """Test cut-and-paste functionality."""
    logger.info("Testing cut-and-paste functionality...")
    
    cutpaste_config = {
        'segment_size_range': (50, 100),
        'num_operations': 2,
        'preserve_trend': True
    }
    
    # Apply cut-paste
    cutpaste_data = augmenter._apply_cutpaste(sample_data.copy(), cutpaste_config)
    
    # Validate that data structure is preserved
    assert cutpaste_data.shape == sample_data.shape, "Shape should be preserved"
    assert list(cutpaste_data.columns) == list(sample_data.columns), "Columns should be preserved"
    
    # Data should be different from original
    assert not cutpaste_data.equals(sample_data), "Data should be modified"
    
    logger.info("✓ Cut-and-paste test passed")
    return cutpaste_data

def test_bootstrap(augmenter, sample_data):
    """Test bootstrap functionality."""
    logger.info("Testing bootstrap functionality...")
    
    # Apply bootstrap sampling
    bootstrap_data = augmenter.create_bootstrapped_dataset(sample_data, sample_ratio=0.8)
    
    # Validate bootstrap properties
    expected_size = int(len(sample_data) * 0.8)
    assert len(bootstrap_data) == expected_size, f"Bootstrap size should be {expected_size}"
    assert list(bootstrap_data.columns) == list(sample_data.columns), "Columns should be preserved"
    
    logger.info("✓ Bootstrap test passed")
    return bootstrap_data

def test_multiple_strategies(augmenter, sample_data):
    """Test multiple augmentation strategies together."""
    logger.info("Testing multiple augmentation strategies...")
    
    augmentation_config = {
        'jittering': {
            'enabled': True,
            'num_datasets': 2,
            'config': {
                'price_noise_std': 0.0005,
                'indicator_noise_std': 0.01,
                'volume_noise_std': 0.03
            }
        },
        'cutpaste': {
            'enabled': True,
            'num_datasets': 1,
            'config': {
                'segment_size_range': (30, 80),
                'num_operations': 1,
                'preserve_trend': True
            }
        },
        'bootstrap': {
            'enabled': True,
            'num_datasets': 1,
            'sample_ratio': 0.9
        }
    }
    
    # Apply multiple strategies
    augmented_datasets = augmenter.augment_with_multiple_strategies(sample_data, augmentation_config)
    
    # Should have original + 2 jittered + 1 cutpaste + 1 bootstrap = 5 datasets
    expected_count = 1 + 2 + 1 + 1  # original + jittered + cutpaste + bootstrap
    assert len(augmented_datasets) == expected_count, f"Should have {expected_count} datasets"
    
    # All datasets should have same structure
    for i, dataset in enumerate(augmented_datasets):
        assert dataset.shape[1] == sample_data.shape[1], f"Dataset {i} should have same number of columns"
        assert list(dataset.columns) == list(sample_data.columns), f"Dataset {i} should have same columns"
    
    logger.info(f"✓ Multiple strategies test passed - Created {len(augmented_datasets)} datasets")
    return augmented_datasets

def visualize_results(original_data, augmented_datasets):
    """Create visualizations of the augmentation results."""
    logger.info("Creating visualizations...")
    
    os.makedirs('plots', exist_ok=True)
    
    # Plot price comparisons
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data
    axes[0, 0].plot(original_data['close'], label='Original', alpha=0.8)
    axes[0, 0].set_title('Original Close Prices')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    
    # Augmented data overlays
    axes[0, 1].plot(original_data['close'], label='Original', alpha=0.8, linewidth=2)
    for i, dataset in enumerate(augmented_datasets[1:4]):  # Show first 3 augmented
        axes[0, 1].plot(dataset['close'], label=f'Augmented {i+1}', alpha=0.6)
    axes[0, 1].set_title('Price Comparison: Original vs Augmented')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    
    # Volume comparison
    axes[1, 0].plot(original_data['volume'], label='Original', alpha=0.8)
    for i, dataset in enumerate(augmented_datasets[1:3]):  # Show first 2 augmented
        axes[1, 0].plot(dataset['volume'], label=f'Augmented {i+1}', alpha=0.6)
    axes[1, 0].set_title('Volume Comparison')
    axes[1, 0].set_ylabel('Volume')
    axes[1, 0].legend()
    
    # RSI comparison
    axes[1, 1].plot(original_data['RSI'], label='Original', alpha=0.8)
    for i, dataset in enumerate(augmented_datasets[1:3]):  # Show first 2 augmented
        axes[1, 1].plot(dataset['RSI'], label=f'Augmented {i+1}', alpha=0.6)
    axes[1, 1].set_title('RSI Comparison')
    axes[1, 1].set_ylabel('RSI')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/data_augmentation_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved visualization to plots/data_augmentation_comparison.png")
    
    # Statistics comparison
    stats_comparison = pd.DataFrame({
        'Original': [
            original_data['close'].mean(),
            original_data['close'].std(),
            original_data['volume'].mean(),
            original_data['RSI'].mean()
        ]
    }, index=['Close Mean', 'Close Std', 'Volume Mean', 'RSI Mean'])
    
    for i, dataset in enumerate(augmented_datasets[1:]):
        stats_comparison[f'Augmented_{i+1}'] = [
            dataset['close'].mean(),
            dataset['close'].std(),
            dataset['volume'].mean(),
            dataset['RSI'].mean()
        ]
    
    print("\nStatistics Comparison:")
    print(stats_comparison.round(4))

def main():
    """Main testing function."""
    logger.info("="*60)
    logger.info("TESTING DATA AUGMENTATION FUNCTIONALITY")
    logger.info("="*60)
    
    # Initialize augmenter
    augmenter = DataAugmenter(random_seed=42)
    
    # Create sample data
    logger.info("Creating sample data...")
    sample_data = create_sample_data()
    logger.info(f"Created sample data with shape: {sample_data.shape}")
    logger.info(f"Sample data columns: {list(sample_data.columns)}")
    
    # Test individual methods
    jittered_data = test_jittering(augmenter, sample_data)
    cutpaste_data = test_cutpaste(augmenter, sample_data)
    bootstrap_data = test_bootstrap(augmenter, sample_data)
    
    # Test multiple strategies
    augmented_datasets = test_multiple_strategies(augmenter, sample_data)
    
    # Create visualizations
    visualize_results(sample_data, augmented_datasets)
    
    # Test with config from yaml
    logger.info("Testing with configuration from config.yaml...")
    training_config = config.get("training", {})
    if training_config.get("data_augmentation", {}).get("enabled", False):
        yaml_config = training_config["data_augmentation"]["config"]
        yaml_datasets = augmenter.augment_with_multiple_strategies(sample_data, yaml_config)
        logger.info(f"✓ YAML config test passed - Created {len(yaml_datasets)} datasets")
    else:
        logger.info("Data augmentation not enabled in config.yaml")
    
    logger.info("="*60)
    logger.info("ALL TESTS PASSED! ✓")
    logger.info("="*60)
    logger.info(f"Data augmentation is ready to use.")
    logger.info(f"To enable in walk-forward testing, set training.data_augmentation.enabled=true in config.yaml")

if __name__ == "__main__":
    main() 