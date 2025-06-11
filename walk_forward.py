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
                           create_summary_report, plot_training_progress, plot_sharpe_comparison,
                           plot_hyperparameter_tuning_summary)
from hyperparameter_tuning import objective_func

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
        
        # Aggregate results first
        summary = self._aggregate_results(results, session_folder)
        
        # Generate plots and reports
        if results:
            plot_walk_forward_results(results, f'{session_folder}/reports')
            plot_cumulative_performance(results, f'{session_folder}/reports')
            plot_sharpe_comparison(summary, f'{session_folder}/reports')
            plot_hyperparameter_tuning_summary(summary, f'{session_folder}/reports')
            create_summary_report(summary, f'{session_folder}/reports')
        
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
        """Process a single walk-forward window with hyperparameter tuning."""
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
        
        # Check if hyperparameter tuning is enabled for walk-forward
        hp_config = self.config.get("hyperparameter_tuning", {})
        wf_hp_enabled = hp_config.get("walk_forward_enabled", True)
        
        best_params = None
        if wf_hp_enabled:
            logger.info(f"Starting hyperparameter tuning for window {window_idx}")
            
            # Run hyperparameter optimization for this window
            tuning_results = self._run_window_hyperparameter_tuning(
                train_data, validation_data, window_folder, window_idx
            )
            
            if tuning_results and 'best_params' in tuning_results:
                best_params = tuning_results['best_params']
                logger.info(f"Window {window_idx} - Best hyperparameters: {best_params}")
                
                # Save hyperparameter results for this window
                with open(f'{window_folder}/hyperparameter_results.json', 'w') as f:
                    json.dump({
                        'best_params': best_params,
                        'best_value': tuning_results.get('best_value', 0),
                        'n_trials': tuning_results.get('n_trials', 0)
                    }, f, indent=4, default=str)
            else:
                logger.warning(f"Window {window_idx} - Hyperparameter tuning failed, using default parameters")
        else:
            logger.info(f"Window {window_idx} - Hyperparameter tuning disabled, using default parameters")
        
        # Train model with best parameters (or defaults if tuning disabled/failed)
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
                model_params=best_params,  # Use optimized hyperparameters
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
                model_params=best_params,  # Use optimized hyperparameters
                save_path=window_folder
            )
        
        # Plot training progress
        if training_info and 'history' in training_info:
            plot_training_progress(training_info['history'], window_folder)
        
        # Evaluate on test data
        evaluator = ModelEvaluator()
        
        # Get reward type from training config
        reward_type = self.config.get("training", {}).get("evaluation", {}).get("reward_type", "hybrid")
        test_results = evaluator.evaluate(model, test_data, verbose=0, reward_type=reward_type)
        
        # Save results
        with open(f'{window_folder}/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4, default=str)
        
        return {
            'window': window_idx,
            'return': test_results['total_return_pct'],
            'sharpe_ratio': test_results.get('sharpe_ratio', 0),
            'enhanced_sharpe_ratio': test_results.get('enhanced_sharpe_ratio', 0),
            'volatility': test_results.get('volatility', 0),
            'max_drawdown': test_results.get('max_drawdown', 0),
            'calmar_ratio': test_results.get('calmar_ratio', 0),
            'portfolio_value': test_results['final_portfolio_value'],
            'trade_count': test_results['trade_count'],
            'hit_rate': test_results.get('hit_rate', 0),
            'prediction_accuracy': test_results.get('prediction_accuracy', 0),
            'portfolio_bankrupted': test_results.get('portfolio_bankrupted', False),
            'portfolio_history': test_results.get('portfolio_history', []),  # Include portfolio history for pooled Sharpe calculation
            'reward_type': test_results.get('reward_type', reward_type),
            'window_folder': window_folder,
            'best_hyperparameters': best_params  # Include the optimized hyperparameters
        }
    
    def _run_window_hyperparameter_tuning(self, train_data: pd.DataFrame, validation_data: pd.DataFrame, 
                                         window_folder: str, window_idx: int) -> Dict:
        """Run hyperparameter tuning for a single window with early stopping."""
        hp_config = self.config.get("hyperparameter_tuning", {})
        
        # Get tuning parameters
        n_trials = hp_config.get("n_trials", 30)
        eval_metric = hp_config.get("eval_metric", "return")
        
        # Configure pruning for early stopping of underperforming trials
        pruner_config = hp_config.get("pruning", {})
        use_pruning = pruner_config.get("enabled", True)
        
        if use_pruning:
            # Set up pruner for early stopping
            pruner_type = pruner_config.get("type", "median")
            
            if pruner_type == "median":
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=pruner_config.get("n_startup_trials", 5),
                    n_warmup_steps=pruner_config.get("n_warmup_steps", 3),
                    interval_steps=pruner_config.get("interval_steps", 1)
                )
            elif pruner_type == "percentile":
                pruner = optuna.pruners.PercentilePruner(
                    percentile=pruner_config.get("percentile", 25.0),
                    n_startup_trials=pruner_config.get("n_startup_trials", 5),
                    n_warmup_steps=pruner_config.get("n_warmup_steps", 3),
                    interval_steps=pruner_config.get("interval_steps", 1)
                )
            elif pruner_type == "successive_halving":
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=pruner_config.get("min_resource", 1),
                    reduction_factor=pruner_config.get("reduction_factor", 4),
                    min_early_stopping_rate=pruner_config.get("min_early_stopping_rate", 0)
                )
            else:
                # Default to median pruner
                pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()  # No pruning
        
        try:
            # Create study with pruning
            study_name = f"window_{window_idx}_hp_tuning"
            sampler = optuna.samplers.TPESampler(seed=self.config.get("seed", 42))
            
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )
            
            # Create objective function with intermediate reporting for pruning
            def pruning_objective(trial):
                try:
                    # Call the main objective function
                    result = objective_func(
                        trial=trial,
                        train_data=train_data,
                        validation_data=validation_data,
                        eval_metric=eval_metric,
                        hit_rate_min_trades=hp_config.get("hit_rate_min_trades", 5),
                        min_predictions=hp_config.get("min_predictions", 10)
                    )
                    
                    # Report intermediate result for pruning (if applicable)
                    if use_pruning:
                        trial.report(result, step=0)
                        
                        # Check if trial should be pruned
                        if trial.should_prune():
                            logger.info(f"Window {window_idx} - Trial {trial.number} pruned with result {result:.2f}")
                            raise optuna.TrialPruned()
                    
                    return result
                    
                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    logger.error(f"Window {window_idx} - Trial {trial.number} failed: {e}")
                    # Return a very bad score for failed trials
                    return -1000.0
            
            # Run optimization
            logger.info(f"Window {window_idx} - Starting {n_trials} trials with {pruner_type} pruning")
            study.optimize(pruning_objective, n_trials=n_trials, timeout=hp_config.get("timeout", None))
            
            # Get results
            best_params = study.best_params
            best_value = study.best_value
            n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            n_pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            
            logger.info(f"Window {window_idx} - Hyperparameter tuning completed:")
            logger.info(f"  Best {eval_metric}: {best_value:.2f}")
            logger.info(f"  Completed trials: {n_completed_trials}/{n_trials}")
            logger.info(f"  Pruned trials: {n_pruned_trials}")
            logger.info(f"  Best parameters: {best_params}")
            
            # Save optimization plots if possible
            try:
                tuning_plots_folder = f'{window_folder}/tuning_plots'
                os.makedirs(tuning_plots_folder, exist_ok=True)
                
                # Save optimization history
                fig1 = optuna.visualization.plot_optimization_history(study)
                fig1.write_image(f'{tuning_plots_folder}/optimization_history.png')
                
                # Save parameter importance (if we have enough trials)
                if n_completed_trials >= 3:
                    fig2 = optuna.visualization.plot_param_importances(study)
                    fig2.write_image(f'{tuning_plots_folder}/param_importances.png')
                
            except ImportError:
                logger.warning("Plotly not available for saving optimization plots")
            except Exception as e:
                logger.warning(f"Could not save optimization plots: {e}")
            
            return {
                "best_params": best_params,
                "best_value": best_value,
                "study": study,
                "n_trials": n_trials,
                "n_completed_trials": n_completed_trials,
                "n_pruned_trials": n_pruned_trials
            }
            
        except Exception as e:
            logger.error(f"Window {window_idx} - Hyperparameter tuning failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_pooled_sharpe_ratio(self, results: List[Dict], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from pooled returns across all windows."""
        all_returns = []
        
        # Pool all period returns from all windows
        for result in results:
            portfolio_history = result.get('portfolio_history', [])
            if len(portfolio_history) < 2:
                continue
                
            # Calculate period returns for this window
            window_returns = []
            for i in range(1, len(portfolio_history)):
                if portfolio_history[i-1] > 0:
                    period_return = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                    window_returns.append(period_return)
            
            all_returns.extend(window_returns)
        
        if len(all_returns) == 0:
            return 0.0
        
        # Convert to pandas Series for calculations
        returns_series = pd.Series(all_returns)
        
        # Remove any NaN or infinite values
        returns_series = returns_series[np.isfinite(returns_series)]
        
        if len(returns_series) == 0:
            return 0.0
        
        # Calculate mean return and standard deviation
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        
        # Handle edge cases
        if pd.isna(mean_return) or pd.isna(std_return) or std_return == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        # Annualize the Sharpe ratio (assuming the data frequency)
        # For intraday data, this might need adjustment based on your data frequency
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)  # 252 trading days per year
        
        logger.info(f"Pooled Sharpe calculation: {len(all_returns)} total returns from {len(results)} windows")
        logger.info(f"Mean return: {mean_return:.6f}, Std return: {std_return:.6f}")
        
        return sharpe_ratio_annualized

    def _aggregate_results(self, results: List[Dict], session_folder: str) -> Dict:
        """Aggregate results from all windows."""
        if not results:
            return {'error': 'No results to aggregate'}
        
        returns = [r['return'] for r in results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        enhanced_sharpe_ratios = [r.get('enhanced_sharpe_ratio', 0) for r in results]
        volatilities = [r.get('volatility', 0) for r in results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results]
        calmar_ratios = [r.get('calmar_ratio', 0) for r in results if np.isfinite(r.get('calmar_ratio', 0))]
        portfolio_values = [r['portfolio_value'] for r in results]
        trade_counts = [r['trade_count'] for r in results]
        hit_rates = [r.get('hit_rate', 0) for r in results]
        prediction_accuracies = [r.get('prediction_accuracy', 0) for r in results]
        bankruptcies = [r.get('portfolio_bankrupted', False) for r in results]
        
        # Count bankrupted windows
        bankrupted_windows = sum(bankruptcies)
        
        # Calculate pooled Sharpe ratio from all windows
        pooled_sharpe_ratio = self._calculate_pooled_sharpe_ratio(results)
        
        summary = {
            'num_windows': len(results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),  # Keep individual average for comparison
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'avg_enhanced_sharpe_ratio': np.mean(enhanced_sharpe_ratios),
            'std_enhanced_sharpe_ratio': np.std(enhanced_sharpe_ratios),
            'pooled_sharpe_ratio': pooled_sharpe_ratio,  # Add pooled Sharpe ratio
            'avg_volatility': np.mean(volatilities),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_calmar_ratio': np.mean(calmar_ratios) if calmar_ratios else 0,
            'avg_portfolio': np.mean(portfolio_values),
            'avg_trades': np.mean(trade_counts),
            'avg_hit_rate': np.mean(hit_rates),
            'avg_prediction_accuracy': np.mean(prediction_accuracies),
            'bankrupted_windows': bankrupted_windows,
            'bankruptcy_rate': (bankrupted_windows / len(results)) * 100 if results else 0,
            'reward_type': results[0].get('reward_type', 'unknown') if results else 'unknown',
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
        logger.info(f"Average Enhanced Sharpe ratio: {summary['avg_enhanced_sharpe_ratio']:.2f}")
        logger.info(f"Pooled Sharpe ratio: {summary['pooled_sharpe_ratio']:.2f}")
        logger.info(f"Average Volatility: {summary['avg_volatility']:.2f}%")
        logger.info(f"Average Max Drawdown: {summary['avg_max_drawdown']:.2f}%")
        logger.info(f"Reward Type: {summary['reward_type']}")
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
    
    # Force CPU usage - disable CUDA
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_device("cpu")
    logger.info("Forcing CPU usage - CUDA disabled")
    
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
        print(f"Pooled Sharpe ratio: {results['pooled_sharpe_ratio']:.2f} (computed from all {results['num_windows']} windows)")
        print(f"Average portfolio value: ${results['avg_portfolio']:.2f}")
        print(f"Average trades per window: {results['avg_trades']:.1f}")
        if results.get('bankrupted_windows', 0) > 0:
            print(f"⚠️  BANKRUPTCY WARNING: {results['bankrupted_windows']} windows ({results['bankruptcy_rate']:.1f}%) went bankrupt (-100% loss)")
        print(f"Results saved to: models/session_{results['timestamp']}")
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()

