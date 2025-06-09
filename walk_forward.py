import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, time
import pytz
from typing import List, Dict, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

from stable_baselines3 import PPO
import optuna

from environment import TradingEnv
from data import get_data
from train import ModelTrainer, ModelEvaluator, save_trade_history
from data_augmentation import DataAugmenter
from config import config
import money
from utils.seeding import seed_worker
from normalization import scale_window, get_standardized_column_names
from plotting_utils import (plot_walk_forward_results, plot_cumulative_performance, 
                           create_summary_report, plot_training_progress)

# Setup logging
os.makedirs('models/logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'models/logs/walk_forward_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WalkForwardTester:
    """Handles walk-forward testing with simplified interface."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = self._filter_market_hours(data)
        self.trading_days = self._get_trading_days()
        self.config = config
        
    def _filter_market_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to NYSE market hours only."""
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data index must be DatetimeIndex")
            return data
            
        filtered_data = data.copy()
        eastern = pytz.timezone('US/Eastern')
        
        # Ensure timezone awareness
        if filtered_data.index.tz is None:
            filtered_data.index = filtered_data.index.tz_localize('UTC')
        
        filtered_data.index = filtered_data.index.tz_convert(eastern)
        
        # Filter for weekdays and market hours (9:30 AM to 4:00 PM ET)
        weekday_mask = (filtered_data.index.dayofweek >= 0) & (filtered_data.index.dayofweek <= 4)
        hours_mask = (filtered_data.index.time >= time(9, 30)) & (filtered_data.index.time <= time(16, 0))
        
        filtered_data = filtered_data[weekday_mask & hours_mask]
        filtered_data.index = filtered_data.index.tz_convert('UTC')
        
        logger.info(f"Filtered to market hours: {len(filtered_data)} / {len(data)} rows")
        return filtered_data
        
    def _get_trading_days(self) -> List[str]:
        """Get unique trading days from filtered data."""
        eastern = pytz.timezone('US/Eastern')
        data_index = self.data.index.tz_convert(eastern)
        unique_days = sorted(set(data_index.date.astype(str)))
        logger.info(f"Found {len(unique_days)} trading days")
        return unique_days
    
    def run_walk_forward(self, 
                        window_size: int,
                        step_size: int,
                        train_ratio: float = 0.7,
                        validation_ratio: float = 0.15,
                        use_parallel: bool = False) -> Dict:
        """Run walk-forward testing."""
        
        # Validate inputs
        if len(self.trading_days) < window_size:
            raise ValueError(f"Not enough trading days ({len(self.trading_days)}) for window size ({window_size})")
        
        num_windows = max(1, (len(self.trading_days) - window_size) // step_size + 1)
        logger.info(f"Running {num_windows} walk-forward windows")
        
        # Create session folder
        session_folder = f'models/session_{timestamp}'
        os.makedirs(f'{session_folder}/models', exist_ok=True)
        os.makedirs(f'{session_folder}/reports', exist_ok=True)
        
        # Prepare window data
        window_configs = self._prepare_windows(num_windows, window_size, step_size, 
                                             train_ratio, validation_ratio, session_folder)
        
        # Process windows
        if use_parallel and num_windows > 1:
            results = self._process_windows_parallel(window_configs)
        else:
            results = self._process_windows_sequential(window_configs)
        
        # Generate plots and reports
        if results:
            plot_walk_forward_results(results, f'{session_folder}/reports')
            plot_cumulative_performance(results, f'{session_folder}/reports')
            create_summary_report(self._aggregate_results(results, session_folder), f'{session_folder}/reports')
        
        # Aggregate results
        summary = self._aggregate_results(results, session_folder)
        return summary
    
    def _prepare_windows(self, num_windows: int, window_size: int, step_size: int,
                        train_ratio: float, validation_ratio: float, session_folder: str) -> List[Dict]:
        """Prepare configuration for each window."""
        window_configs = []
        
        for i in range(num_windows):
            # Calculate window boundaries
            start_day_idx = i * step_size
            end_day_idx = start_day_idx + window_size
            if end_day_idx > len(self.trading_days):
                end_day_idx = len(self.trading_days)
            
            start_day = self.trading_days[start_day_idx]
            end_day = self.trading_days[end_day_idx - 1]
            
            # Extract window data
            window_data = self._extract_window_data(start_day, end_day)
            
            # Split into train/validation/test
            train_idx = int(len(window_data) * train_ratio)
            validation_idx = train_idx + int(len(window_data) * validation_ratio)
            
            train_data = window_data.iloc[:train_idx].copy()
            validation_data = window_data.iloc[train_idx:validation_idx].copy()
            test_data = window_data.iloc[validation_idx:].copy()
            
            window_folder = f'{session_folder}/models/window_{i+1}'
            os.makedirs(window_folder, exist_ok=True)
            
            window_configs.append({
                'window_idx': i + 1,
                'train_data': train_data,
                'validation_data': validation_data,
                'test_data': test_data,
                'window_folder': window_folder
            })
            
        return window_configs
    
    def _extract_window_data(self, start_day: str, end_day: str) -> pd.DataFrame:
        """Extract data for a specific window."""
        eastern = pytz.timezone('US/Eastern')
        data_eastern = self.data.copy()
        data_eastern.index = data_eastern.index.tz_convert(eastern)
        
        window_mask = (data_eastern.index.date.astype(str) >= start_day) & (data_eastern.index.date.astype(str) <= end_day)
        window_data = data_eastern[window_mask].copy()
        window_data.index = window_data.index.tz_convert('UTC')
        
        return window_data
    
    def _process_windows_sequential(self, window_configs: List[Dict]) -> List[Dict]:
        """Process windows sequentially."""
        results = []
        for config in window_configs:
            result = self._process_single_window(config)
            results.append(result)
            logger.info(f"Completed window {config['window_idx']}")
        return results
    
    def _process_windows_parallel(self, window_configs: List[Dict]) -> List[Dict]:
        """Process windows in parallel."""
        max_workers = min(multiprocessing.cpu_count(), len(window_configs))
        logger.info(f"Processing windows with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers, initializer=seed_worker, 
                                initargs=(self.config.get("seed", 42),)) as executor:
            futures = [executor.submit(self._process_single_window, config) for config in window_configs]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing window: {e}")
        
        # Sort by window index
        results.sort(key=lambda x: x.get('window', 0))
        return results
    
    def _process_single_window(self, config: Dict) -> Dict:
        """Process a single walk-forward window."""
        window_idx = config['window_idx']
        train_data = config['train_data']
        validation_data = config['validation_data']
        test_data = config['test_data']
        window_folder = config['window_folder']
        
        logger.info(f"Processing window {window_idx}")
        
        # Apply sigmoid transformation to prevent leakage
        cols_to_scale = get_standardized_column_names(train_data)
        sigmoid_params, train_data, validation_data, test_data = scale_window(
            train_data, validation_data, test_data, cols_to_scale, 
            window_folder=window_folder,
            sigmoid_k=2.0  # Use sigmoid transformation only
        )
        
        # Train model
        trainer = ModelTrainer(train_data, validation_data)
        training_config = self.config.get("training", {})
        
        # Check if data augmentation is enabled
        use_augmentation = training_config.get("data_augmentation", {}).get("enabled", False)
        
        if use_augmentation:
            logger.info(f"Training with data augmentation for window {window_idx}")
            augmentation_config = training_config.get("data_augmentation", {}).get("config", None)
            
            model, training_info = trainer.train_with_augmented_data(
                initial_timesteps=training_config.get("total_timesteps", 20000),
                additional_timesteps=training_config.get("additional_timesteps", 5000),
                max_iterations=training_config.get("max_iterations", 10),
                patience=training_config.get("n_stagnant_loops", 3),
                improvement_threshold=training_config.get("improvement_threshold", 0.1),
                save_path=window_folder,
                use_data_augmentation=True,
                augmentation_config=augmentation_config
            )
        else:
            logger.info(f"Training without data augmentation for window {window_idx}")
            model, training_info = trainer.train_with_validation(
                initial_timesteps=training_config.get("total_timesteps", 20000),
                additional_timesteps=training_config.get("additional_timesteps", 5000),
                max_iterations=training_config.get("max_iterations", 10),
                patience=training_config.get("n_stagnant_loops", 3),
                improvement_threshold=training_config.get("improvement_threshold", 0.1),
                save_path=window_folder
            )
        
        # Plot training progress
        if training_info and 'history' in training_info:
            plot_training_progress(training_info['history'], window_folder)
        
        # Evaluate on test data
        evaluator = ModelEvaluator()
        test_results = evaluator.evaluate(model, test_data, verbose=0)
        
        # Save results
        with open(f'{window_folder}/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4, default=str)
        
        return {
            'window': window_idx,
            'return': test_results['total_return_pct'],
            'sharpe_ratio': test_results.get('sharpe_ratio', 0),
            'portfolio_value': test_results['final_portfolio_value'],
            'trade_count': test_results['trade_count'],
            'hit_rate': test_results.get('hit_rate', 0),
            'prediction_accuracy': test_results.get('prediction_accuracy', 0),
            'portfolio_bankrupted': test_results.get('portfolio_bankrupted', False),
            'window_folder': window_folder
        }
    
    def _aggregate_results(self, results: List[Dict], session_folder: str) -> Dict:
        """Aggregate results from all windows."""
        if not results:
            return {'error': 'No results to aggregate'}
        
        returns = [r['return'] for r in results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        portfolio_values = [r['portfolio_value'] for r in results]
        trade_counts = [r['trade_count'] for r in results]
        hit_rates = [r.get('hit_rate', 0) for r in results]
        prediction_accuracies = [r.get('prediction_accuracy', 0) for r in results]
        bankruptcies = [r.get('portfolio_bankrupted', False) for r in results]
        
        # Count bankrupted windows
        bankrupted_windows = sum(bankruptcies)
        
        summary = {
            'num_windows': len(results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'avg_portfolio': np.mean(portfolio_values),
            'avg_trades': np.mean(trade_counts),
            'avg_hit_rate': np.mean(hit_rates),
            'avg_prediction_accuracy': np.mean(prediction_accuracies),
            'bankrupted_windows': bankrupted_windows,
            'bankruptcy_rate': (bankrupted_windows / len(results)) * 100 if results else 0,
            'all_window_results': results,
            'timestamp': timestamp
        }
        
        # Save summary
        with open(f'{session_folder}/reports/summary.json', 'w') as f:
            # Don't include detailed results in summary JSON
            summary_for_json = {k: v for k, v in summary.items() if k != 'all_window_results'}
            json.dump(summary_for_json, f, indent=4, default=str)
        
        logger.info(f"Walk-forward testing complete. Average return: {summary['avg_return']:.2f}%")
        logger.info(f"Average Sharpe ratio: {summary['avg_sharpe_ratio']:.2f}")
        if summary['bankrupted_windows'] > 0:
            logger.warning(f"Bankruptcy Alert: {summary['bankrupted_windows']} windows ({summary['bankruptcy_rate']:.1f}%) resulted in portfolio bankruptcy")
        return summary

def load_tradingview_data(csv_filepath: str) -> pd.DataFrame:
    """Load and process TradingView CSV data."""
    logger.info(f"Loading TradingView data from {csv_filepath}")
    
    try:
        df = pd.read_csv(csv_filepath)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        time_cols = ['time', 'timestamp']
        
        available_cols = [col.lower() for col in df.columns]
        
        if not all(col in available_cols for col in required_cols):
            raise ValueError(f"Missing required columns. Available: {df.columns.tolist()}")
        
        if not any(col in available_cols for col in time_cols):
            raise ValueError(f"Missing time column. Available: {df.columns.tolist()}")
        
        # Standardize column names
        col_mapping = {}
        for req_col in required_cols:
            for avail_col in df.columns:
                if avail_col.lower() == req_col:
                    col_mapping[avail_col] = req_col
        
        # Handle time column
        time_col_found = None
        for time_col in time_cols:
            for avail_col in df.columns:
                if avail_col.lower() == time_col:
                    time_col_found = avail_col
                    col_mapping[avail_col] = 'time'
                    break
            if time_col_found:
                break
        
        df = df.rename(columns=col_mapping)
        
        # Convert time to datetime index
        if pd.api.types.is_numeric_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], unit='s')
        else:
            df['time'] = pd.to_datetime(df['time'])
        
        df = df.set_index('time')
        
        # Process technical indicators
        from data import process_technical_indicators
        df = process_technical_indicators(df)
        
        logger.info(f"Loaded TradingView data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading TradingView data: {e}")
        return None

def main():
    """Main walk-forward testing function."""
    
    # Load data
    data = load_tradingview_data("data/NQ_2024_unix.csv")
    if data is None or len(data) == 0:
        logger.error("Failed to load data")
        return
    
    # Get configuration
    wf_config = config.get("walk_forward", {})
    window_size = wf_config.get("window_size", 14)
    step_size = wf_config.get("step_size", 7)
    use_parallel = wf_config.get("parallel_processing", {}).get("enabled", False)
    
    logger.info(f"Window size: {window_size} trading days")
    logger.info(f"Step size: {step_size} trading days") 
    logger.info(f"Parallel processing: {use_parallel}")
    
    # Run walk-forward testing
    tester = WalkForwardTester(data)
    results = tester.run_walk_forward(
        window_size=window_size,
        step_size=step_size,
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        use_parallel=use_parallel
    )
    
    # Print summary
    if "error" not in results:
        print(f"\nWalk-Forward Testing Complete!")
        print(f"Number of windows: {results['num_windows']}")
        print(f"Average return: {results['avg_return']:.2f}% ± {results['std_return']:.2f}%")
        print(f"Average Sharpe ratio: {results['avg_sharpe_ratio']:.2f} ± {results['std_sharpe_ratio']:.2f}")
        print(f"Average portfolio value: ${results['avg_portfolio']:.2f}")
        print(f"Average trades per window: {results['avg_trades']:.1f}")
        if results.get('bankrupted_windows', 0) > 0:
            print(f"⚠️  BANKRUPTCY WARNING: {results['bankrupted_windows']} windows ({results['bankruptcy_rate']:.1f}%) went bankrupt (-100% loss)")
        print(f"Results saved to: models/session_{results['timestamp']}")
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()

