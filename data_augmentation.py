import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from copy import deepcopy

logger = logging.getLogger(__name__)

class DataAugmenter:
    """Handles data augmentation for creating multiple synthetic training datasets."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data augmenter with a random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def augment_training_data(self, 
                            train_data: pd.DataFrame, 
                            num_synthetic_datasets: int = 3,
                            jitter_config: Dict = None,
                            cutpaste_config: Dict = None) -> List[pd.DataFrame]:
        """
        Create multiple synthetic training datasets from the original training data.
        
        Args:
            train_data: Original training dataset
            num_synthetic_datasets: Number of synthetic datasets to create
            jitter_config: Configuration for jittering (noise addition)
            cutpaste_config: Configuration for cut-and-paste operations
            
        Returns:
            List of augmented datasets including the original
        """
        if jitter_config is None:
            jitter_config = {
                'price_noise_std': 0.001,  # 0.1% noise for price columns
                'indicator_noise_std': 0.02,  # 2% noise for indicators
                'volume_noise_std': 0.05   # 5% noise for volume
            }
        
        if cutpaste_config is None:
            cutpaste_config = {
                'segment_size_range': (50, 200),  # Size range for segments to cut/paste
                'num_operations': 2,  # Number of cut-paste operations per dataset
                'preserve_trend': True  # Try to preserve overall trend
            }
        
        logger.info(f"Creating {num_synthetic_datasets} synthetic datasets from training data of shape {train_data.shape}")
        
        augmented_datasets = [train_data.copy()]  # Include original dataset
        
        for i in range(num_synthetic_datasets):
            logger.info(f"Creating synthetic dataset {i+1}/{num_synthetic_datasets}")
            
            # Start with original data
            synthetic_data = train_data.copy()
            
            # Apply jittering
            if jitter_config:
                synthetic_data = self._apply_jittering(synthetic_data, jitter_config)
            
            # Apply cut-and-paste operations
            if cutpaste_config:
                synthetic_data = self._apply_cutpaste(synthetic_data, cutpaste_config)
            
            augmented_datasets.append(synthetic_data)
        
        logger.info(f"Created {len(augmented_datasets)} total datasets (including original)")
        return augmented_datasets
    
    def _apply_jittering(self, data: pd.DataFrame, jitter_config: Dict) -> pd.DataFrame:
        """Apply noise-based jittering to the data."""
        augmented_data = data.copy()
        
        # Identify different types of columns
        price_columns = ['open', 'high', 'low', 'close', 'close_norm']
        volume_columns = ['volume', 'VOLUME_NORM', 'OBV']
        
        # Get all indicator columns (exclude price, volume, and non-numeric columns)
        excluded_cols = price_columns + volume_columns + ['position', 'timestamp']
        indicator_columns = [col for col in augmented_data.columns 
                           if col not in excluded_cols and 
                           augmented_data[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        # Apply noise to price columns
        for col in price_columns:
            if col in augmented_data.columns:
                noise_std = jitter_config.get('price_noise_std', 0.001)
                noise = np.random.normal(0, noise_std * augmented_data[col].std(), len(augmented_data))
                augmented_data[col] = augmented_data[col] * (1 + noise)
        
        # Apply noise to volume columns
        for col in volume_columns:
            if col in augmented_data.columns:
                noise_std = jitter_config.get('volume_noise_std', 0.05)
                noise = np.random.normal(0, noise_std * augmented_data[col].std(), len(augmented_data))
                augmented_data[col] = augmented_data[col] * (1 + noise)
                # Ensure volume stays positive
                augmented_data[col] = np.maximum(augmented_data[col], 0)
        
        # Apply noise to indicator columns
        for col in indicator_columns:
            if col in augmented_data.columns and not col.startswith('DOW_') and not col.startswith('MSO_'):
                noise_std = jitter_config.get('indicator_noise_std', 0.02)
                noise = np.random.normal(0, noise_std * augmented_data[col].std(), len(augmented_data))
                augmented_data[col] = augmented_data[col] + noise
        
        # Maintain OHLC relationships (ensure high >= max(open, close) and low <= min(open, close))
        if all(col in augmented_data.columns for col in ['open', 'high', 'low', 'close']):
            augmented_data['high'] = np.maximum(augmented_data['high'], 
                                              np.maximum(augmented_data['open'], augmented_data['close']))
            augmented_data['low'] = np.minimum(augmented_data['low'], 
                                             np.minimum(augmented_data['open'], augmented_data['close']))
        
        return augmented_data
    
    def _apply_cutpaste(self, data: pd.DataFrame, cutpaste_config: Dict) -> pd.DataFrame:
        """Apply cut-and-paste operations to rearrange data segments."""
        augmented_data = data.copy()
        data_length = len(augmented_data)
        
        if data_length < 100:  # Skip if data is too small
            return augmented_data
        
        segment_size_range = cutpaste_config.get('segment_size_range', (50, 200))
        num_operations = cutpaste_config.get('num_operations', 2)
        preserve_trend = cutpaste_config.get('preserve_trend', True)
        
        for operation in range(num_operations):
            # Choose random segment size
            segment_size = random.randint(
                max(10, min(segment_size_range[0], data_length // 10)),
                min(segment_size_range[1], data_length // 5)
            )
            
            # Choose source and destination positions
            max_source_start = data_length - segment_size
            source_start = random.randint(0, max_source_start)
            source_end = source_start + segment_size
            
            # Choose destination (avoid overlap with source)
            max_dest_start = data_length - segment_size
            dest_start = random.randint(0, max_dest_start)
            
            # Avoid overlap
            while (dest_start < source_end and dest_start + segment_size > source_start):
                dest_start = random.randint(0, max_dest_start)
            
            dest_end = dest_start + segment_size
            
            # Extract source segment
            source_segment = augmented_data.iloc[source_start:source_end].copy()
            
            # If preserve_trend is True, adjust the segment to match destination context
            if preserve_trend and 'close' in augmented_data.columns:
                # Get the price level at destination
                if dest_start > 0:
                    dest_base_price = augmented_data.iloc[dest_start - 1]['close']
                    source_base_price = source_segment.iloc[0]['close']
                    
                    # Calculate adjustment factor
                    adjustment_factor = dest_base_price / source_base_price if source_base_price != 0 else 1
                    
                    # Apply adjustment to price columns
                    price_cols = ['open', 'high', 'low', 'close', 'close_norm']
                    for col in price_cols:
                        if col in source_segment.columns:
                            source_segment[col] = source_segment[col] * adjustment_factor
            
            # Paste the segment at destination
            augmented_data.iloc[dest_start:dest_end] = source_segment.values
        
        return augmented_data
    
    def create_bootstrapped_dataset(self, data: pd.DataFrame, sample_ratio: float = 0.8) -> pd.DataFrame:
        """Create a bootstrapped dataset by sampling with replacement."""
        sample_size = int(len(data) * sample_ratio)
        sampled_indices = np.random.choice(len(data), size=sample_size, replace=True)
        sampled_indices = np.sort(sampled_indices)  # Maintain temporal order somewhat
        
        bootstrapped_data = data.iloc[sampled_indices].copy()
        # Reset index to avoid duplicate indices
        bootstrapped_data = bootstrapped_data.reset_index(drop=True)
        
        return bootstrapped_data
    
    def create_sliding_window_datasets(self, 
                                     data: pd.DataFrame, 
                                     window_size: int = None,
                                     step_size: int = None,
                                     num_windows: int = 3) -> List[pd.DataFrame]:
        """Create multiple datasets using sliding windows."""
        if window_size is None:
            window_size = len(data) // 2
        
        if step_size is None:
            step_size = window_size // 4  # 25% overlap
        
        datasets = []
        data_length = len(data)
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > data_length:
                # Wrap around or adjust
                if start_idx < data_length:
                    # Take what we can and pad with beginning
                    remaining = end_idx - data_length
                    window_data = pd.concat([
                        data.iloc[start_idx:],
                        data.iloc[:remaining]
                    ]).reset_index(drop=True)
                else:
                    break
            else:
                window_data = data.iloc[start_idx:end_idx].copy()
            
            datasets.append(window_data)
        
        return datasets
    
    def augment_with_multiple_strategies(self, 
                                       train_data: pd.DataFrame,
                                       augmentation_config: Dict = None) -> List[pd.DataFrame]:
        """Apply multiple augmentation strategies to create diverse datasets."""
        if augmentation_config is None:
            augmentation_config = {
                'jittering': {
                    'enabled': True,
                    'num_datasets': 2,
                    'config': {
                        'price_noise_std': 0.001,
                        'indicator_noise_std': 0.02,
                        'volume_noise_std': 0.05
                    }
                },
                'cutpaste': {
                    'enabled': True,
                    'num_datasets': 2,
                    'config': {
                        'segment_size_range': (50, 200),
                        'num_operations': 2,
                        'preserve_trend': True
                    }
                },
                'bootstrap': {
                    'enabled': True,
                    'num_datasets': 1,
                    'sample_ratio': 0.85
                },
                'sliding_window': {
                    'enabled': False,  # Can be resource intensive
                    'num_datasets': 2,
                    'window_size': None,
                    'step_size': None
                }
            }
        
        all_datasets = [train_data.copy()]  # Start with original
        
        # Apply jittering
        if augmentation_config.get('jittering', {}).get('enabled', False):
            jitter_config = augmentation_config['jittering']['config']
            num_jitter = augmentation_config['jittering']['num_datasets']
            
            for i in range(num_jitter):
                jittered_data = self._apply_jittering(train_data.copy(), jitter_config)
                all_datasets.append(jittered_data)
        
        # Apply cut-paste
        if augmentation_config.get('cutpaste', {}).get('enabled', False):
            cutpaste_config = augmentation_config['cutpaste']['config']
            num_cutpaste = augmentation_config['cutpaste']['num_datasets']
            
            for i in range(num_cutpaste):
                cutpaste_data = self._apply_cutpaste(train_data.copy(), cutpaste_config)
                all_datasets.append(cutpaste_data)
        
        # Apply bootstrapping
        if augmentation_config.get('bootstrap', {}).get('enabled', False):
            num_bootstrap = augmentation_config['bootstrap']['num_datasets']
            sample_ratio = augmentation_config['bootstrap']['sample_ratio']
            
            for i in range(num_bootstrap):
                bootstrap_data = self.create_bootstrapped_dataset(train_data, sample_ratio)
                all_datasets.append(bootstrap_data)
        
        # Apply sliding window
        if augmentation_config.get('sliding_window', {}).get('enabled', False):
            sliding_config = augmentation_config['sliding_window']
            sliding_datasets = self.create_sliding_window_datasets(
                train_data,
                window_size=sliding_config.get('window_size'),
                step_size=sliding_config.get('step_size'),
                num_windows=sliding_config['num_datasets']
            )
            all_datasets.extend(sliding_datasets)
        
        logger.info(f"Created {len(all_datasets)} total datasets using multiple augmentation strategies")
        return all_datasets 