import logging
import pandas as pd
import os
import json
from typing import Dict, List, Tuple
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_linear_fn

from environment import TradingEnv
from data import get_data
from config import config
import money
from utils.seeding import set_global_seed
from data_augmentation import DataAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles PPO model training with configurable parameters."""
    
    def __init__(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None):
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config
        self.data_augmenter = DataAugmenter(random_seed=self.config.get('seed', 42))
        
    def create_model(self, model_params: Dict = None) -> PPO:
        """Create a PPO model with specified parameters."""
        env = TradingEnv(
            self.train_data,
            initial_balance=self.config["environment"]["initial_balance"],
            position_size=self.config["environment"].get("position_size", 1),
            returns_window=self.config["environment"].get("returns_window", 30)
        )
        check_env(env, skip_render_check=True)
        
        # Use provided params or defaults from config
        if model_params is None:
            model_params = self._get_default_params()
            
        # Handle learning rate decay
        learning_rate = self._setup_learning_rate(model_params)
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=self.config["training"].get("verbose", 1),
            learning_rate=learning_rate,
            ent_coef=model_params.get("ent_coef", 0.01),
            n_steps=model_params.get("n_steps", 2048),
            batch_size=model_params.get("batch_size", 64),
            gamma=model_params.get("gamma", 0.99),
            gae_lambda=model_params.get("gae_lambda", 0.95),
            seed=self.config.get('seed'),
            device="cpu"  # Force CPU usage
        )
        
        return model
    
    def _get_default_params(self) -> Dict:
        """Get default model parameters from config."""
        return {
            "ent_coef": self.config["model"].get("ent_coef", 0.01),
            "learning_rate": self.config["model"].get("learning_rate", 0.0003),
            "n_steps": self.config["model"].get("n_steps", 2048),
            "batch_size": self.config["model"].get("batch_size", 64),
            "gamma": self.config["model"].get("gamma", 0.99),
            "gae_lambda": self.config["model"].get("gae_lambda", 0.95),
        }
    
    def _setup_learning_rate(self, model_params: Dict):
        """Setup learning rate schedule if decay is enabled."""
        use_lr_decay = self.config["model"].get("use_lr_decay", False)
        
        if use_lr_decay:
            initial_lr = model_params.get("learning_rate", 0.0003)
            final_lr = self.config["model"].get("final_learning_rate", 1e-5)
            total_timesteps = self.config["training"].get("total_timesteps", 50000)
            
            learning_rate = get_linear_fn(initial_lr, final_lr, total_timesteps)
            logger.info(f"Using learning rate decay from {initial_lr} to {final_lr}")
        else:
            learning_rate = model_params.get("learning_rate", 0.0003)
            logger.info(f"Using constant learning rate: {learning_rate}")
            
        return learning_rate
    
    def train_simple(self, total_timesteps: int, model_params: Dict = None) -> PPO:
        """Simple training without iteration."""
        model = self.create_model(model_params)
        logger.info(f"Starting training for {total_timesteps} timesteps")
        model.learn(total_timesteps=total_timesteps)
        logger.info("Training completed")
        return model
    
    def train_with_augmented_data(self,
                                initial_timesteps: int = 20000,
                                additional_timesteps: int = 10000,
                                max_iterations: int = 20,
                                patience: int = 3,
                                improvement_threshold: float = 0.1,
                                model_params: Dict = None,
                                save_path: str = None,
                                use_data_augmentation: bool = True,
                                augmentation_config: Dict = None) -> Tuple[PPO, Dict]:
        """Train model with data augmentation and validation-based early stopping."""
        
        if self.validation_data is None:
            raise ValueError("Validation data required for iterative training")
        
        # Create augmented datasets if enabled
        if use_data_augmentation:
            logger.info("Creating augmented training datasets")
            
            if augmentation_config is None:
                # Default augmentation configuration
                augmentation_config = {
                    'jittering': {
                        'enabled': True,
                        'num_datasets': 2,
                        'config': {
                            'price_noise_std': 0.0005,  # Smaller noise for stability
                            'indicator_noise_std': 0.01,
                            'volume_noise_std': 0.03
                        }
                    },
                    'cutpaste': {
                        'enabled': True,
                        'num_datasets': 1,
                        'config': {
                            'segment_size_range': (30, 100),
                            'num_operations': 1,
                            'preserve_trend': True
                        }
                    },
                    'bootstrap': {
                        'enabled': False,  # Disable bootstrap for now
                        'num_datasets': 1,
                        'sample_ratio': 0.9
                    }
                }
            
            augmented_datasets = self.data_augmenter.augment_with_multiple_strategies(
                self.train_data, augmentation_config
            )
            logger.info(f"Created {len(augmented_datasets)} training datasets for multi-episode training")
        else:
            augmented_datasets = [self.train_data]
        
        # Initialize model with first dataset
        model = self.create_model(model_params)
        evaluator = ModelEvaluator()
        
        # Track training across all datasets
        best_results = None
        best_model = None
        best_metric = float('-inf')
        patience_counter = 0
        training_history = []
        
        # Initial training on first dataset
        logger.info(f"Initial training on dataset 1/{len(augmented_datasets)} for {initial_timesteps} timesteps")
        model.learn(total_timesteps=initial_timesteps)
        
        # Initial evaluation
        best_results = evaluator.evaluate(model, self.validation_data)
        best_model = model
        best_metric = self._get_metric_value(best_results)
        training_history.append(best_results)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save(os.path.join(save_path, "best_model"))
        
        # Iterative training with multiple datasets
        for iteration in range(1, max_iterations + 1):
            # Cycle through augmented datasets
            dataset_idx = iteration % len(augmented_datasets)
            current_dataset = augmented_datasets[dataset_idx]
            
            logger.info(f"Iteration {iteration}: training on dataset {dataset_idx+1}/{len(augmented_datasets)} for {additional_timesteps} timesteps")
            
            # Create new environment with current dataset
            # Each dataset gets a fresh environment to ensure proper episode boundaries
            env = TradingEnv(
                current_dataset,
                initial_balance=self.config["environment"]["initial_balance"],
                position_size=self.config["environment"].get("position_size", 1),
                returns_window=self.config["environment"].get("returns_window", 30)
            )
            
            # Update model environment and force episode reset for new dataset
            model.set_env(env)
            
            # Force environment reset to start fresh episode with new dataset
            env.reset()
            
            # Train on the new dataset with episode reset
            # reset_num_timesteps=True ensures proper episode boundaries between datasets
            model.learn(total_timesteps=additional_timesteps, reset_num_timesteps=True)
            
            # Evaluate on validation data
            results = evaluator.evaluate(model, self.validation_data)
            current_metric = self._get_metric_value(results)
            training_history.append(results)
            
            improvement = current_metric - best_metric
            logger.info(f"Iteration {iteration}: metric = {current_metric:.2f}%, improvement = {improvement:.2f}%")
            
            if improvement > improvement_threshold:
                best_metric = current_metric
                best_model = model
                best_results = results
                patience_counter = 0
                
                if save_path:
                    model.save(os.path.join(save_path, "best_model"))
                logger.info(f"New best model! Metric: {best_metric:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {patience} iterations without improvement")
                    break
        
        return best_model, {"best_results": best_results, "history": training_history}

    def train_with_validation(self, 
                            initial_timesteps: int = 20000,
                            additional_timesteps: int = 10000,
                            max_iterations: int = 20,
                            patience: int = 3,
                            improvement_threshold: float = 0.1,
                            model_params: Dict = None,
                            save_path: str = None) -> Tuple[PPO, Dict]:
        """Train model iteratively with validation-based early stopping."""
        
        if self.validation_data is None:
            raise ValueError("Validation data required for iterative training")
            
        model = self.create_model(model_params)
        evaluator = ModelEvaluator()
        
        # Initial training
        logger.info(f"Initial training for {initial_timesteps} timesteps")
        model.learn(total_timesteps=initial_timesteps)
        
        # Initial evaluation
        best_results = evaluator.evaluate(model, self.validation_data)
        best_model = model
        best_metric = self._get_metric_value(best_results)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save(os.path.join(save_path, "best_model"))
        
        # Iterative training
        patience_counter = 0
        training_history = [best_results]
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Iteration {iteration}: training for {additional_timesteps} timesteps")
            model.learn(total_timesteps=additional_timesteps)
            
            results = evaluator.evaluate(model, self.validation_data)
            current_metric = self._get_metric_value(results)
            training_history.append(results)
            
            improvement = current_metric - best_metric
            logger.info(f"Iteration {iteration}: metric = {current_metric:.2f}%, improvement = {improvement:.2f}%")
            
            if improvement > improvement_threshold:
                best_metric = current_metric
                best_model = model
                best_results = results
                patience_counter = 0
                
                if save_path:
                    model.save(os.path.join(save_path, "best_model"))
                logger.info(f"New best model! Metric: {best_metric:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {patience} iterations without improvement")
                    break
        
        return best_model, {"best_results": best_results, "history": training_history}
    
    def _get_metric_value(self, results: Dict) -> float:
        """Extract the metric value based on config.
        
        Enhanced to support composite metrics that balance returns and risk.
        """
        metric = self.config.get("training", {}).get("evaluation", {}).get("metric", "composite_score")
        
        if metric == "hit_rate":
            return results.get("hit_rate", 0)
        elif metric == "prediction_accuracy":
            return results.get("prediction_accuracy", 0)
        elif metric == "sharpe_ratio":
            return results.get("sharpe_ratio", 0)
        elif metric == "total_return_pct":
            return results.get("total_return_pct", 0)
        elif metric == "composite_score":
            return self._calculate_composite_score(results)
        elif metric == "risk_adjusted_score":
            return self._calculate_risk_adjusted_score(results)
        else:
            return results.get("total_return_pct", 0)
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate composite score balancing returns, Sharpe ratio, and other metrics."""
        total_return = results.get("total_return_pct", 0)
        sharpe_ratio = results.get("sharpe_ratio", 0)
        hit_rate = results.get("hit_rate", 0) / 100.0  # Convert to decimal
        portfolio_bankrupted = results.get("portfolio_bankrupted", False)
        
        # Severe penalty for bankruptcy
        if portfolio_bankrupted:
            return -1000.0
        
        # Weights for different components
        return_weight = 0.4
        sharpe_weight = 0.4
        hit_rate_weight = 0.2
        
        # Normalize Sharpe ratio to similar scale as returns
        sharpe_normalized = np.tanh(sharpe_ratio / 3.0) * 20  # Scale to ~±20 range
        
        # Calculate composite score
        composite = (return_weight * total_return + 
                    sharpe_weight * sharpe_normalized + 
                    hit_rate_weight * (hit_rate - 0.5) * 40)  # Hit rate bonus/penalty
        
        return float(composite)
    
    def _calculate_risk_adjusted_score(self, results: Dict) -> float:
        """Calculate risk-adjusted score emphasizing consistent performance."""
        total_return = results.get("total_return_pct", 0)
        sharpe_ratio = results.get("sharpe_ratio", 0)
        portfolio_bankrupted = results.get("portfolio_bankrupted", False)
        
        # Severe penalty for bankruptcy
        if portfolio_bankrupted:
            return -1000.0
        
        # For risk-adjusted scoring, prioritize Sharpe ratio over raw returns
        if sharpe_ratio > 0:
            # Reward positive Sharpe with bonus for high values
            risk_adjusted = total_return * (1 + min(sharpe_ratio / 2.0, 2.0))
        else:
            # Penalize negative Sharpe by reducing the return score
            risk_adjusted = total_return * max(0.1, 1 + sharpe_ratio / 5.0)
        
        return float(risk_adjusted)

class ModelEvaluator:
    """Handles model evaluation with different metrics."""
    
    def evaluate(self, model: PPO, test_data: pd.DataFrame, 
                 deterministic: bool = True, verbose: int = 0, reward_type: str = "hybrid") -> Dict:
        """Evaluate model and return comprehensive results."""
        
        # Ensure close_norm column exists
        if 'close_norm' not in test_data.columns:
            close_col = self._get_close_column(test_data)
            test_data['close_norm'] = test_data[close_col].pct_change().fillna(0)
        
        env = TradingEnv(
            test_data,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=config["environment"].get("transaction_cost", 0.0),
            position_size=config["environment"].get("position_size", 1),
            returns_window=config["environment"].get("returns_window", 30),
            reward_type=reward_type
        )
        
        # Run evaluation
        obs, _ = env.reset()
        initial_balance = env.net_worth
        
        portfolio_history = [float(env.net_worth)]
        action_history = []
        reward_history = []
        sharpe_history = []
        portfolio_bankrupted = False
        
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_history.append(int(action))
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            # Track additional metrics
            reward_history.append(reward)
            sharpe_history.append(info.get("rolling_sharpe_ratio", 0))
            
            # Check if portfolio was bankrupted
            if info.get("portfolio_bankrupted", False):
                portfolio_bankrupted = True
                logger.warning(f"Portfolio bankrupted during evaluation at step {step_count}")
            
            portfolio_history.append(float(env.net_worth))
            step_count += 1
        
        # Calculate metrics
        final_balance = env.net_worth
        
        # If portfolio was bankrupted, return exactly -100%
        if portfolio_bankrupted:
            total_return_pct = -100.0
            sharpe_ratio = -999.0  # Very negative Sharpe for bankruptcy
            enhanced_sharpe_ratio = -999.0
        else:
            total_return_pct = money.calculate_return_pct(final_balance, initial_balance)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_history)
            enhanced_sharpe_ratio = self._calculate_enhanced_sharpe_ratio(portfolio_history)
        
        # Calculate additional performance metrics
        volatility = self._calculate_volatility(portfolio_history)
        max_drawdown = self._calculate_max_drawdown(portfolio_history)
        calmar_ratio = self._calculate_calmar_ratio(total_return_pct, max_drawdown)
        
        results = {
            "final_portfolio_value": float(final_balance),
            "total_return_pct": float(total_return_pct),
            "sharpe_ratio": float(sharpe_ratio),
            "enhanced_sharpe_ratio": float(enhanced_sharpe_ratio),
            "volatility": float(volatility),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "trade_count": step_count,
            "hit_rate": self._calculate_hit_rate(action_history, test_data),
            "prediction_accuracy": self._calculate_prediction_accuracy(action_history, test_data),
            "portfolio_history": portfolio_history,
            "action_history": action_history,
            "reward_history": reward_history,
            "sharpe_history": sharpe_history,
            "final_position": env.position,
            "portfolio_bankrupted": portfolio_bankrupted,
            "reward_type": reward_type
        }
        
        return results
    
    def _get_close_column(self, data: pd.DataFrame) -> str:
        """Get the appropriate close price column name."""
        return 'close'
    
    def _calculate_hit_rate(self, actions: List[int], data: pd.DataFrame) -> float:
        """Calculate hit rate based on actions and price movements."""
        if len(actions) == 0:
            return 0.0
        
        close_col = self._get_close_column(data)
        price_changes = data[close_col].pct_change().fillna(0)
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, action in enumerate(actions[:-1]):
            if i < len(price_changes) - 1:
                next_change = price_changes.iloc[i + 1]
                
                if action == 0 and next_change > 0:  # Buy and price goes up
                    correct_predictions += 1
                elif action == 1 and next_change < 0:  # Sell and price goes down
                    correct_predictions += 1
                
                total_predictions += 1
        
        return (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    def _calculate_prediction_accuracy(self, actions: List[int], data: pd.DataFrame) -> float:
        """Calculate prediction accuracy - same as hit rate for now."""
        return self._calculate_hit_rate(actions, data)
    
    def _calculate_sharpe_ratio(self, portfolio_history: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from portfolio history."""
        if len(portfolio_history) < 2:
            return 0.0
        
        # Calculate period returns
        returns = []
        for i in range(1, len(portfolio_history)):
            if portfolio_history[i-1] != 0:
                period_return = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                returns.append(period_return)
        
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns)
        
        # Calculate mean return and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Handle edge cases
        if pd.isna(mean_return) or pd.isna(std_return) or std_return == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        # Annualize the Sharpe ratio (assuming daily data)
        # For intraday data, this might need adjustment based on your data frequency
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)  # 252 trading days per year
        
        return sharpe_ratio_annualized
    
    def _calculate_enhanced_sharpe_ratio(self, portfolio_history: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate enhanced Sharpe ratio with better handling of edge cases."""
        if len(portfolio_history) < 2:
            return 0.0
        
        # Calculate period returns using log returns for better properties
        returns = []
        for i in range(1, len(portfolio_history)):
            if portfolio_history[i-1] > 0:
                log_return = np.log(portfolio_history[i] / portfolio_history[i-1])
                returns.append(log_return)
        
        if len(returns) < 2:
            return 0.0
        
        returns = pd.Series(returns)
        
        # Remove outliers using IQR method for more robust calculation
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        filtered_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
        
        if len(filtered_returns) < 2:
            filtered_returns = returns  # Use original if too many outliers
        
        # Calculate mean return and standard deviation
        mean_return = filtered_returns.mean()
        std_return = filtered_returns.std(ddof=1)
        
        # Handle edge cases
        if pd.isna(mean_return) or pd.isna(std_return) or std_return == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        # Annualize the Sharpe ratio
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)
        
        return sharpe_ratio_annualized
    
    def _calculate_volatility(self, portfolio_history: List[float]) -> float:
        """Calculate annualized volatility of portfolio returns."""
        if len(portfolio_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(portfolio_history)):
            if portfolio_history[i-1] > 0:
                period_return = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                returns.append(period_return)
        
        if len(returns) < 2:
            return 0.0
        
        returns_series = pd.Series(returns)
        volatility = returns_series.std(ddof=1) * np.sqrt(252)  # Annualized
        
        return volatility if pd.notna(volatility) else 0.0
    
    def _calculate_max_drawdown(self, portfolio_history: List[float]) -> float:
        """Calculate maximum drawdown as a percentage."""
        if len(portfolio_history) < 2:
            return 0.0
        
        portfolio_series = pd.Series(portfolio_history)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown) if pd.notna(max_drawdown) else 0.0
    
    def _calculate_calmar_ratio(self, total_return_pct: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if max_drawdown == 0:
            return 0.0 if total_return_pct <= 0 else float('inf')
        
        # Annualize the return (assuming the evaluation period represents the annualized return)
        calmar = total_return_pct / max_drawdown
        
        return calmar if pd.notna(calmar) else 0.0

def save_trade_history(trade_history: List[Dict], filename: str = "trade_history.csv"):
    """Save trade history to CSV file."""
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        trade_df.to_csv(filename, index=False)
        logger.info(f"Trade history saved to {filename}")

def main():
    """Main training function."""
    # Force CPU usage - disable CUDA
    import torch
    torch.device("cpu")
    torch.set_default_device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Set up deterministic training
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_global_seed(config["seed"])
    
    logger.info("Forcing CPU usage - CUDA disabled")
    
    # Load data
    train_data, validation_data, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        use_yfinance=True
    )
    
    # Initialize trainer
    trainer = ModelTrainer(train_data, validation_data)
    evaluator = ModelEvaluator()
    
    # Train model
    training_config = config.get("training", {})
    use_validation = training_config.get("use_validation", True)
    
    if use_validation and validation_data is not None:
        logger.info("Training with validation-based early stopping")
        model, training_info = trainer.train_with_validation(
            initial_timesteps=training_config.get("total_timesteps", 50000),
            additional_timesteps=training_config.get("additional_timesteps", 10000),
            max_iterations=training_config.get("max_iterations", 20),
            patience=training_config.get("n_stagnant_loops", 3),
            improvement_threshold=training_config.get("improvement_threshold", 0.1),
            save_path="models"
        )
        best_results = training_info["best_results"]
    else:
        logger.info("Simple training without validation")
        model = trainer.train_simple(training_config.get("total_timesteps", 50000))
        best_results = evaluator.evaluate(model, validation_data or test_data)
    
    # Final evaluation on test data
    logger.info("Evaluating on test data")
    reward_type = training_config.get("evaluation", {}).get("reward_type", "hybrid")
    test_results = evaluator.evaluate(model, test_data, verbose=1, reward_type=reward_type)
    
    # Log results
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Final Portfolio Value: ${test_results['final_portfolio_value']:.2f}")
    logger.info(f"Total Return: {test_results['total_return_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    logger.info(f"Enhanced Sharpe Ratio: {test_results['enhanced_sharpe_ratio']:.2f}")
    logger.info(f"Volatility: {test_results['volatility']:.2f}%")
    logger.info(f"Max Drawdown: {test_results['max_drawdown']:.2f}%")
    logger.info(f"Calmar Ratio: {test_results['calmar_ratio']:.2f}")
    logger.info(f"Hit Rate: {test_results['hit_rate']:.2f}%")
    logger.info(f"Prediction Accuracy: {test_results['prediction_accuracy']:.2f}%")
    logger.info(f"Reward Type Used: {test_results['reward_type']}")
    
    # Calculate and log composite scores
    trainer_instance = ModelTrainer(train_data, validation_data)
    composite_score = trainer_instance._calculate_composite_score(test_results)
    risk_adjusted_score = trainer_instance._calculate_risk_adjusted_score(test_results)
    logger.info(f"Composite Score: {composite_score:.2f}")
    logger.info(f"Risk-Adjusted Score: {risk_adjusted_score:.2f}")
    
    # Save model and results
    os.makedirs("models", exist_ok=True)
    model.save("models/final_model")
    
    with open("models/final_results.json", "w") as f:
        json.dump({
            "test_results": test_results,
            "validation_results": best_results if use_validation else None,
            "composite_score": composite_score,
            "risk_adjusted_score": risk_adjusted_score
        }, f, indent=4, default=str)
    
    logger.info("Model and results saved to models/")

if __name__ == "__main__":
    main()

