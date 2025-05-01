import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment import TradingEnv
from config import config

# Make sure config has necessary structure
if 'data' not in config:
    config['data'] = {}
if 'symbol' not in config['data']:
    config['data']['symbol'] = 'NQ'  # Default to NQ
if 'period' not in config['data']:
    config['data']['period'] = '60d'  # Default to 60 days
if 'interval' not in config['data']:
    config['data']['interval'] = '5m'  # Default to 5 minutes

class FeatureImportanceCallback(BaseCallback):
    """
    Callback for collecting feature activations and analyzing importance.
    """
    def __init__(self, eval_env, verbose=0):
        super(FeatureImportanceCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.feature_activations = []
        self.rewards = []
        self.last_reward = 0  # Keep track of the last reward received
        
    def _on_step(self):
        # Get the current features being used by the model
        features = self.eval_env._get_obs()
        # Store feature values
        self.feature_activations.append(features)
        # Store the reward directly instead of trying to access model.ep_info_buffer
        self.rewards.append(self.last_reward)
        return True
    
    def update_reward(self, reward):
        """Update the last reward received"""
        self.last_reward = reward
    
    def get_feature_importance(self):
        """
        Calculate feature importance by correlation with rewards.
        """
        # Convert to numpy arrays
        features = np.array(self.feature_activations)
        rewards = np.array(self.rewards)
        
        # Match lengths if needed
        min_len = min(len(features), len(rewards))
        features = features[:min_len]
        rewards = rewards[:min_len]
        
        # Calculate correlation for each feature
        importance = {}
        for i in range(features.shape[1]):
            if len(features) > 0 and len(rewards) > 0:
                corr = np.corrcoef(features[:, i], rewards)[0, 1]
                importance[i] = abs(corr) if not np.isnan(corr) else 0
            else:
                importance[i] = 0
                
        return importance

def analyze_feature_importance(model_path, data, indicator_names):
    """
    Analyze which features are most important for the model's decisions.
    
    Args:
        model_path: Path to the saved model
        data: DataFrame with trading data
        indicator_names: List of indicator names in the same order as in observation space
    
    Returns:
        DataFrame of sorted feature importance
    """
    # Ensure environment config is available
    if 'environment' not in config:
        config['environment'] = {'initial_balance': 10000.0, 'position_size': 1}
    elif 'initial_balance' not in config['environment']:
        config['environment']['initial_balance'] = 10000.0
        
    # Load the model first to get observation space info
    model = PPO.load(model_path)
    expected_obs_dim = model.observation_space.shape[0]
    print(f"Model expects observation space with dimension: {expected_obs_dim}")
    
    # Adjust indicator_names if needed to match model's expected dimensions
    # Expected obs space is: close_norm + indicators + position (1 + len(indicators) + 1)
    expected_indicator_count = expected_obs_dim - 2  # Subtract 1 for close_norm and 1 for position
    
    if len(indicator_names) != expected_indicator_count:
        print(f"Warning: Model expects {expected_indicator_count} indicators but {len(indicator_names)} were provided")
        # If we have too many indicators, trim the list
        if len(indicator_names) > expected_indicator_count:
            print(f"Trimming indicator list to match model's expected dimensions")
            indicator_names = indicator_names[:expected_indicator_count]
        # If we have too few indicators, we might need to pad (though this is not ideal)
        elif len(indicator_names) < expected_indicator_count:
            padding_needed = expected_indicator_count - len(indicator_names)
            print(f"Not enough indicators provided. Adding {padding_needed} placeholder indicators to match dimensions.")
            if len(indicator_names) > 0:
                # Use the first indicator repeatedly as padding
                padding = [indicator_names[0]] * padding_needed
                indicator_names = indicator_names + padding
            else:
                print("Error: No indicators available for padding. Cannot proceed.")
                return pd.DataFrame()
    
    print(f"Using indicators: {indicator_names}")
    
    # Create environment for evaluation with the adjusted indicators
    env = TradingEnv(
        data,
        initial_balance=config["environment"].get("initial_balance", 10000.0),
        position_size=config["environment"].get("position_size", 1),
        enabled_indicators=indicator_names
    )
    
    # Verify observation space dimensions match
    test_obs = env.reset()[0]
    print(f"Environment observation space dimension: {len(test_obs)}")
    
    if len(test_obs) != expected_obs_dim:
        raise ValueError(f"Environment observation space dimension ({len(test_obs)}) " 
                         f"doesn't match model's expected dimension ({expected_obs_dim})")
    
    # Create callback
    callback = FeatureImportanceCallback(eval_env=env)
    
    # Run model with callback
    obs = env.reset()[0]
    done = False
    while not done:
        action, _ = model.predict(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Update the reward in the callback
        callback.update_reward(reward)
        # Call _on_step to record data
        callback._on_step()
        obs = next_obs
        done = terminated or truncated
    
    # Get feature importance
    importance = callback.get_feature_importance()
    
    # Create DataFrame for display
    feature_names = ["close_norm"] + indicator_names + ["position"]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": [importance.get(i, 0) for i in range(len(feature_names))]
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values("Importance", ascending=False)
    
    return importance_df

def permutation_importance(model_path, data, indicator_names, n_repeats=5):
    """
    Calculate feature importance using permutation importance method.
    
    Args:
        model_path: Path to the saved model
        data: DataFrame with trading data
        indicator_names: List of indicator names
        n_repeats: Number of times to repeat permutation for each feature
    
    Returns:
        DataFrame of sorted feature importance
    """
    # Load model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame()
        
    expected_obs_dim = model.observation_space.shape[0]
    print(f"Model expects observation space with dimension: {expected_obs_dim}")
    
    # Adjust indicator_names if needed to match model's expected dimensions
    expected_indicator_count = expected_obs_dim - 2  # Subtract 1 for close_norm and 1 for position
    
    if len(indicator_names) != expected_indicator_count:
        print(f"Warning: Model expects {expected_indicator_count} indicators but {len(indicator_names)} were provided")
        # If we have too many indicators, trim the list
        if len(indicator_names) > expected_indicator_count:
            print(f"Trimming indicator list to match model's expected dimensions")
            indicator_names = indicator_names[:expected_indicator_count]
        # If we have too few indicators, we might need to pad (though this is not ideal)
        elif len(indicator_names) < expected_indicator_count:
            padding_needed = expected_indicator_count - len(indicator_names)
            print(f"Not enough indicators provided. Adding {padding_needed} placeholder indicators to match dimensions.")
            if len(indicator_names) > 0:
                # Use the first indicator repeatedly as padding
                padding = [indicator_names[0]] * padding_needed
                indicator_names = indicator_names + padding
            else:
                print("Error: No indicators available for padding. Cannot proceed.")
                return pd.DataFrame()
    
    print(f"Using indicators: {indicator_names}")
    
    # Ensure environment config is available
    if 'environment' not in config:
        config['environment'] = {'initial_balance': 10000.0, 'position_size': 1}
    elif 'initial_balance' not in config['environment']:
        config['environment']['initial_balance'] = 10000.0
    
    # Baseline performance
    baseline_env = TradingEnv(
        data,
        initial_balance=config["environment"].get("initial_balance", 10000.0),
        position_size=config["environment"].get("position_size", 1),
        enabled_indicators=indicator_names
    )
    
    # Verify observation space dimensions match
    try:
        test_obs = baseline_env.reset()[0]
        print(f"Environment observation space dimension: {len(test_obs)}")
        
        if len(test_obs) != expected_obs_dim:
            raise ValueError(f"Environment observation space dimension ({len(test_obs)}) " 
                            f"doesn't match model's expected dimension ({expected_obs_dim})")
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return pd.DataFrame()
    
    # Get baseline performance
    obs, _ = baseline_env.reset()
    done = False
    baseline_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = baseline_env.step(action)
        baseline_reward += reward
        done = terminated or truncated
    
    print(f"Baseline reward: {baseline_reward}")
    
    # Calculate importance for each feature
    importance = {}
    feature_names = ["close_norm"] + indicator_names + ["position"]
    
    # Only proceed if the observation space matches expected dimensions
    if len(feature_names) != expected_obs_dim:
        print(f"Error: Feature names length ({len(feature_names)}) doesn't match expected observation space ({expected_obs_dim})")
        return pd.DataFrame()
    
    for idx, feature in enumerate(feature_names):
        print(f"Testing importance of feature {idx+1}/{len(feature_names)}: {feature}")
        feature_importance = []
        
        for _ in range(n_repeats):
            # Create environment with same data
            env = TradingEnv(
                data.copy(),
                initial_balance=config["environment"].get("initial_balance", 10000.0),
                position_size=config["environment"].get("position_size", 1),
                enabled_indicators=indicator_names
            )
            
            # Run episode with permuted feature
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Permute the specific feature
                if idx < len(obs):  # Make sure the index is valid
                    obs_copy = obs.copy()
                    # Randomly shuffle the feature value
                    obs_copy[idx] = np.random.uniform(
                        env.observation_space.low[idx],
                        env.observation_space.high[idx]
                    )
                    action, _ = model.predict(obs_copy)
                else:
                    action, _ = model.predict(obs)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            # Calculate importance as drop in performance
            feature_importance.append(baseline_reward - total_reward)
        
        # Average importance over repeats
        importance[feature] = np.mean(feature_importance)
        print(f"  Average importance: {importance[feature]:.4f}")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Importance": list(importance.values())
    })
    
    # Sort by importance (absolute value)
    importance_df["AbsImportance"] = importance_df["Importance"].abs()
    importance_df = importance_df.sort_values("AbsImportance", ascending=False)
    importance_df = importance_df.drop("AbsImportance", axis=1)
    
    return importance_df

def plot_feature_importance(importance_df, title="Feature Importance"):
    """
    Plot feature importance.
    """
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

def run_feature_pruning_analysis(model_path, data):
    """
    Run feature pruning analysis and return pruning recommendations.
    
    Args:
        model_path: Path to the saved model
        data: DataFrame with trading data
    
    Returns:
        Dictionary with pruning recommendations
    """
    # Load the model to check its expected observation space
    try:
        model = PPO.load(model_path)
        expected_obs_dim = model.observation_space.shape[0]
        print(f"Model expects observation space with dimension: {expected_obs_dim}")
        
        # Expected indicator count (obs_dim - close_norm - position)
        expected_indicator_count = expected_obs_dim - 2
        print(f"Expected indicator count: {expected_indicator_count}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using default indicator set without dimension check")
        expected_indicator_count = None
    
    # Get all available indicators
    all_indicators = []
    if 'supertrend' in data.columns:
        all_indicators.append('supertrend')
    if 'RSI' in data.columns:
        all_indicators.append('RSI')
    if 'CCI' in data.columns:
        all_indicators.append('CCI')
    if 'ADX' in data.columns:
        all_indicators.append('ADX')
    if 'ADX_POS' in data.columns:
        all_indicators.append('ADX_POS')
    if 'ADX_NEG' in data.columns:
        all_indicators.append('ADX_NEG')
    if 'STOCH_K' in data.columns:
        all_indicators.append('STOCH_K')
    if 'STOCH_D' in data.columns:
        all_indicators.append('STOCH_D')
    if 'MACD' in data.columns:
        all_indicators.append('MACD')
    if 'MACD_SIGNAL' in data.columns:
        all_indicators.append('MACD_SIGNAL')
    if 'MACD_HIST' in data.columns:
        all_indicators.append('MACD_HIST')
    if 'ROC' in data.columns:
        all_indicators.append('ROC')
    if 'WILLIAMS_R' in data.columns:
        all_indicators.append('WILLIAMS_R')
    if 'SMA_NORM' in data.columns:
        all_indicators.append('SMA_NORM')
    if 'EMA_NORM' in data.columns:
        all_indicators.append('EMA_NORM')
    if 'DISPARITY' in data.columns:
        all_indicators.append('DISPARITY')
    if 'ATR' in data.columns:
        all_indicators.append('ATR')
    if 'OBV_NORM' in data.columns:
        all_indicators.append('OBV_NORM')
    if 'CMF' in data.columns:
        all_indicators.append('CMF')
    if 'PSAR_NORM' in data.columns:
        all_indicators.append('PSAR_NORM')
    if 'PSAR_DIR' in data.columns:
        all_indicators.append('PSAR_DIR')
    if 'VOLUME_MA' in data.columns:
        all_indicators.append('VOLUME_MA')
    if 'VWAP_NORM' in data.columns:
        all_indicators.append('VWAP_NORM')
    if 'DOW_SIN' in data.columns:
        all_indicators.append('DOW_SIN')
    if 'DOW_COS' in data.columns:
        all_indicators.append('DOW_COS')
    if 'MSO_SIN' in data.columns:
        all_indicators.append('MSO_SIN')
    if 'MSO_COS' in data.columns:
        all_indicators.append('MSO_COS')
    
    # If we know how many indicators the model expects, adjust our list
    if expected_indicator_count is not None and len(all_indicators) != expected_indicator_count:
        print(f"Adjusting indicator list: have {len(all_indicators)}, need {expected_indicator_count}")
        if len(all_indicators) > expected_indicator_count:
            # Keep only the first expected_indicator_count indicators
            all_indicators = all_indicators[:expected_indicator_count]
        else:
            padding_needed = expected_indicator_count - len(all_indicators)
            print(f"Not enough indicators available. Adding {padding_needed} placeholder indicators to match dimensions.")
            if len(all_indicators) > 0:
                # Use the first indicator repeatedly as padding
                padding = [all_indicators[0]] * padding_needed
                all_indicators = all_indicators + padding
            else:
                print("Error: No indicators available for padding. Cannot proceed.")
                return {}, pd.DataFrame(), pd.DataFrame()
    
    print(f"Analyzing importance for {len(all_indicators)} indicators...")
    
    # Method 1: Correlation-based importance
    importance_df = analyze_feature_importance(model_path, data, all_indicators)
    print("\nFeature Importance (Correlation):")
    print(importance_df)
    plot_feature_importance(importance_df, "Feature Importance (Correlation)")
    
    # Method 2: Permutation importance
    perm_importance_df = permutation_importance(model_path, data, all_indicators, n_repeats=5)
    print("\nFeature Importance (Permutation):")
    print(perm_importance_df)
    plot_feature_importance(perm_importance_df, "Feature Importance (Permutation)")
    
    # Identify least important features (bottom 25%)
    threshold = importance_df["Importance"].quantile(0.25)
    to_prune = importance_df[importance_df["Importance"] <= threshold]["Feature"].tolist()
    
    # Make pruning recommendations
    recommendations = {
        "least_important_features": to_prune,
        "recommended_features": importance_df[importance_df["Importance"] > threshold]["Feature"].tolist(),
        "importance_threshold": float(threshold)
    }
    
    return recommendations, importance_df, perm_importance_df

if __name__ == "__main__":
    from get_data import get_data
    import os
    
    # Add default data configuration for testing if missing
    if 'data' not in config:
        config['data'] = {}
        
    # Load data with default parameter values if needed
    train_data, validation_data, test_data = get_data(
        symbol=config["data"].get("symbol", "NQ"),
        period=config["data"].get("period", "60d"),
        interval=config["data"].get("interval", "5m"),
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        use_yfinance=True
    )
    
    # Check if best_model exists, if not, warn the user
    if not os.path.exists("best_model"):
        print("Warning: 'best_model' not found. Please train a model first using train.py")
        print("Try using a different model file or create a simple model for testing")
        model_path = input("Enter path to model file or press Enter to attempt to use 'best_model' anyway: ")
        if not model_path:
            model_path = "best_model"
    else:
        model_path = "best_model"
    
    try:
        # Run analysis on the model
        recommendations, imp_df, perm_df = run_feature_pruning_analysis(model_path, validation_data)
        
        print("\nPruning Recommendations:")
        if not recommendations or "least_important_features" not in recommendations:
            print("No pruning recommendations available.")
        else:
            print(f"Features to consider pruning: {recommendations['least_important_features']}")
            print(f"Recommended features to keep: {recommendations['recommended_features']}")
            
            # Save results
            if not imp_df.empty:
                imp_df.to_csv("feature_importance_correlation.csv")
                print("Saved correlation-based feature importance to feature_importance_correlation.csv")
                
            if not perm_df.empty:
                perm_df.to_csv("feature_importance_permutation.csv")
                print("Saved permutation-based feature importance to feature_importance_permutation.csv")
            
            # Save recommendations
            import json
            with open("pruning_recommendations.json", "w") as f:
                json.dump(recommendations, f, indent=4)
            print("Saved pruning recommendations to pruning_recommendations.json")
    except Exception as e:
        print(f"\nError during feature importance analysis: {e}")
        print("Please ensure you have a trained model with the correct observation space dimensions.") 