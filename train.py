import logging
import pandas as pd
import os
import json
from typing import Dict, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_linear_fn

from environment import TradingEnv
from data import get_data
from config import config
import money
from utils.seeding import set_global_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles PPO model training with configurable parameters."""
    
    def __init__(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None):
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config
        
    def create_model(self, model_params: Dict = None) -> PPO:
        """Create a PPO model with specified parameters."""
        env = TradingEnv(
            self.train_data,
            initial_balance=self.config["environment"]["initial_balance"],
            position_size=self.config["environment"].get("position_size", 1)
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
            seed=self.config.get('seed')
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
        """Extract the metric value based on config."""
        metric = self.config.get("training", {}).get("evaluation", {}).get("metric", "return")
        
        if metric == "hit_rate":
            return results.get("hit_rate", 0)
        elif metric == "prediction_accuracy":
            return results.get("prediction_accuracy", 0)
        else:
            return results.get("total_return_pct", 0)

class ModelEvaluator:
    """Handles model evaluation with different metrics."""
    
    def evaluate(self, model: PPO, test_data: pd.DataFrame, 
                 deterministic: bool = True, verbose: int = 0) -> Dict:
        """Evaluate model and return comprehensive results."""
        
        # Ensure close_norm column exists
        if 'close_norm' not in test_data.columns:
            close_col = self._get_close_column(test_data)
            test_data['close_norm'] = test_data[close_col].pct_change().fillna(0)
        
        env = TradingEnv(
            test_data,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=config["environment"].get("transaction_cost", 0.0),
            position_size=config["environment"].get("position_size", 1)
        )
        
        # Run evaluation
        obs, _ = env.reset()
        initial_balance = env.net_worth
        
        portfolio_history = [float(env.net_worth)]
        action_history = []
        
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_history.append(int(action))
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            portfolio_history.append(float(env.net_worth))
            step_count += 1
        
        # Calculate metrics
        final_balance = env.net_worth
        total_return_pct = money.calculate_return_pct(final_balance, initial_balance)
        
        return {
            "final_portfolio_value": float(final_balance),
            "total_return_pct": float(total_return_pct),
            "trade_count": step_count,
            "hit_rate": self._calculate_hit_rate(action_history, test_data),
            "prediction_accuracy": self._calculate_prediction_accuracy(action_history, test_data),
            "portfolio_history": portfolio_history,
            "action_history": action_history,
            "final_position": env.position
        }
    
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

def save_trade_history(trade_history: List[Dict], filename: str = "trade_history.csv"):
    """Save trade history to CSV file."""
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        trade_df.to_csv(filename, index=False)
        logger.info(f"Trade history saved to {filename}")

def main():
    """Main training function."""
    # Set up deterministic training
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_global_seed(config["seed"])
    
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
    test_results = evaluator.evaluate(model, test_data, verbose=1)
    
    # Log results
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Final Portfolio Value: ${test_results['final_portfolio_value']:.2f}")
    logger.info(f"Total Return: {test_results['total_return_pct']:.2f}%")
    logger.info(f"Hit Rate: {test_results['hit_rate']:.2f}%")
    logger.info(f"Prediction Accuracy: {test_results['prediction_accuracy']:.2f}%")
    
    # Save model and results
    os.makedirs("models", exist_ok=True)
    model.save("models/final_model")
    
    with open("models/final_results.json", "w") as f:
        json.dump({
            "test_results": test_results,
            "validation_results": best_results if use_validation else None
        }, f, indent=4, default=str)
    
    logger.info("Model and results saved to models/")

if __name__ == "__main__":
    main()

