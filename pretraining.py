import logging
import glob
import os
from functools import partial
from typing import Dict, Optional, List, Callable

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import config
from environment import TradingEnv
from train import evaluate_agent
from utils.seeding import set_global_seed

logger = logging.getLogger(__name__)

# --- Utilities ---------------------------------------------------------------

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load CSV, parse timestamp, drop 'position'."""
    try:
        df = pd.read_csv(filepath)
        if 'timestamp' not in df:
            logger.warning("%s missing timestamp", filepath)
            return None
        df.timestamp = pd.to_datetime(df.timestamp)
        df.set_index('timestamp', inplace=True)
        return df.drop(columns=['position'], errors='ignore')
    except Exception:
        logger.exception("Failed to load %s", filepath)
        return None


def list_csv(directory: str, max_files: Optional[int] = None) -> List[str]:
    files = sorted(f for f in glob.glob(f"{directory}/*.csv") if not f.endswith('_scaler.pkl'))
    return files[:max_files] if max_files else files


def make_schedule(name: str, initial: float) -> Callable[[float], float]:
    """Return learning rate schedule by name: linear, exp, cosine."""
    cfg = config['model']
    if name == 'exponential':
        decay = cfg.get('lr_decay_rate', .96)
        return lambda p: initial * (decay ** (1 - p))
    if name == 'cosine':
        minv = cfg.get('lr_min_value', 1e-6)
        return lambda p: minv + (initial - minv) * 0.5 * (1 + np.cos(np.pi * p))
    # linear
    return lambda p: p * initial


def init_mlflow_run(name: str, nested: bool = True, **params):
    run_id = mlflow.active_run() is not None
    mlflow.start_run(run_name=name, nested=nested and run_id)
    for k, v in params.items():
        mlflow.log_param(k, v)


# --- Pretraining -------------------------------------------------------------

def pretrain(
    data_dir: str,
    timesteps: int = 5_000,
    params: Optional[Dict] = None,
    max_files: Optional[int] = None
) -> PPO:
    """Pretrain PPO on synthetic CSVs."""
    files = list_csv(data_dir, max_files)
    if not files:
        logger.error("No data for pretraining")
        return None
    
    default = config['model']
    params = params or {
        'ent_coef': default.get('ent_coef', .01),
        'learning_rate': default.get('learning_rate', 3e-4),
        'n_steps': default.get('n_steps', 2048),
        'batch_size': default.get('batch_size', 64),
        'gamma': default.get('gamma', .99),
        'gae_lambda': default.get('gae_lambda', .95),
    }

    schedule = make_schedule(config['model'].get('lr_schedule', 'linear'), params['learning_rate'])
    init_mlflow_run('Pretraining',
                    initial_timesteps=timesteps,
                    total_files=len(files),
                    **{f'model_{k}': v for k, v in params.items()})

    # Initialize
    set_global_seed(config.get('seed', 42))
    first = load_data(files[0])
    
    # Get environment config and override transaction_cost
    env_config = config['environment'].copy()
    env_config['transaction_cost'] = 0.0
    
    env = TradingEnv(first, **env_config)
    model = PPO('MlpPolicy', env, verbose=0,
                ent_coef=params['ent_coef'], learning_rate=schedule,
                n_steps=params['n_steps'], batch_size=params['batch_size'],
                gamma=params['gamma'], gae_lambda=params['gae_lambda'],
                seed=config['seed'])

    stats = {'returns': [], 'portfolios': [], 'trades': [], 'hit_rates': []}

    for idx, fpath in enumerate(files, 1):
        set_global_seed(config['seed'] + idx)
        data = load_data(fpath)
        if data is None:
            continue
        env = TradingEnv(data, **env_config)
        model.set_env(env)
        model.learn(total_timesteps=timesteps)

        # Evaluate
        metrics = evaluate_agent(model, data, verbose=0, deterministic=True) or {}
        r = metrics.get('total_return_pct', 0)
        p = metrics.get('final_portfolio_value', config['environment']['initial_balance'])
        t = metrics.get('trade_count', 0)
        h = metrics.get('hit_rate', 0)

        for name, val in zip(stats, [r, p, t, h]):
            stats[name].append(val)
            mlflow.log_metric(name, val, step=idx * timesteps)

    # Final summary
    for key, arr in stats.items():
        arr = np.array(arr)
        mlflow.log_metric(f'{key}_mean', float(np.nanmean(arr)))
        mlflow.log_metric(f'{key}_std', float(np.nanstd(arr)))

    mlflow.end_run()
    return model


# --- Evaluation Accuracy ----------------------------------------------------

def evaluate_accuracy(
    model: PPO,
    data: pd.DataFrame,
    verbose: int = 0,
    deterministic: bool = True
) -> Dict:
    """Measure prediction accuracy, returns history & metrics."""
    cols = [c for c in ['Close', 'CLOSE', 'close'] if c in data.columns]
    price_col = cols[0] if cols else 'close'

    env = TradingEnv(data, **config['environment'])
    obs, _ = env.reset()
    hist = {'actions': [], 'portfolio': [config['environment']['initial_balance']]}
    correct = total = 0
    pos = 0

    while True:
        step = env.current_step
        price = data.iloc[step][price_col]
        action, _ = model.predict(obs, deterministic=deterministic)
        hist['actions'].append(int(action))

        # Determine correctness if not last
        if step < len(data) - 1:
            change = data.iloc[step+1][price_col] - price
            pred = (action==0 and change>0) or (action==1 and change<0)
            correct += pred
            total += 1
        obs, reward, done, truncated, _ = env.step(action)
        hist['portfolio'].append(env.net_worth)
        if done or truncated:
            break

    acc = 100 * correct/total if total else 0
    ret = 100 * (hist['portfolio'][-1] - hist['portfolio'][0]) / hist['portfolio'][0]
    return {
        'prediction_accuracy': acc,
        'total_return_pct': ret,
        'final_portfolio_value': hist['portfolio'][-1],
        'trade_history': hist,
    }

# --- Hyperparameter Tuning --------------------------------------------------

def tune_pretraining(
    data_dir: str,
    trials: int = 20,
    metric: str = 'return',
    files: int = 20,
    timesteps: int = 500
) -> Dict:
    import optuna
    all_files = list_csv(data_dir, files)
    datasets = [load_data(f) for f in all_files]
    datasets = [d for d in datasets if d is not None]
    if not datasets:
        logger.error("No valid data for tuning")
        return {}

    init_mlflow_run('Tuning', n_trials=trials, metric=metric)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config['seed']))
    
    # Get hyperparameter ranges from config
    hp_config = config.get('hyperparameter_tuning', {}).get('parameters', {})
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float(
                'learning_rate', 
                hp_config.get('learning_rate', {}).get('min', 1e-5),
                hp_config.get('learning_rate', {}).get('max', 1e-2),
                log=True
            ),
            'n_steps': trial.suggest_int(
                'n_steps',
                hp_config.get('n_steps', {}).get('min', 128),
                hp_config.get('n_steps', {}).get('max', 2048),
                log=True
            ),
            'ent_coef': trial.suggest_float(
                'ent_coef',
                hp_config.get('ent_coef', {}).get('min', 0.00001),
                hp_config.get('ent_coef', {}).get('max', 0.5),
                log=True
            ),
            'batch_size': trial.suggest_int(
                'batch_size',
                hp_config.get('batch_size', {}).get('min', 8),
                hp_config.get('batch_size', {}).get('max', 128),
                log=True
            ),
            'gamma': config['model'].get('gamma', 0.99),
            'gae_lambda': config['model'].get('gae_lambda', 0.95),
        }
        
        model = pretrain(data_dir, timesteps=timesteps, params=params, max_files=files)
        if model is None:
            return float('-inf')
            
        results = evaluate_accuracy(model, datasets[-1]) if metric=='prediction_accuracy' else evaluate_agent(model, datasets[-1])
        return results.get('prediction_accuracy' if metric=='prediction_accuracy' else 'total_return_pct', 0)

    study.optimize(objective, n_trials=trials)
    best = study.best_params
    mlflow.log_params({f'best_{k}': v for k, v in best.items()})
    mlflow.log_metric('best_value', study.best_value)
    mlflow.end_run()
    return {'best_params': best, 'best_value': study.best_value}

# --- Main --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Pretrain PPO with optional tuning")
    parser.add_argument('--dir', default='./synthetic_normalized')
    parser.add_argument('--skip-tuning', action='store_true')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--files', type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    set_global_seed(config['seed'])

    if not args.skip_tuning:
        tune_pretraining(args.dir, trials=args.trials, files=args.files or 20, timesteps=args.timesteps)

    model = pretrain(args.dir,
                     timesteps=args.timesteps * 10,
                     max_files=args.files)
    model.save("pretrained_model.zip")
    logger.info("Pretraining complete - model saved as pretrained_model.zip")
