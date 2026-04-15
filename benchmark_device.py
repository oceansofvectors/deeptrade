"""
Benchmark RecurrentPPO on CUDA vs CPU.
"""
import time
import torch
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import RecurrentPPO
from config import config
from environment import TradingEnv


def create_synthetic_data(n_rows: int = 5000) -> pd.DataFrame:
    """Create synthetic trading data for benchmarking."""
    np.random.seed(42)

    # Generate random walk price data
    returns = np.random.randn(n_rows) * 0.001
    close = 20000 * np.exp(np.cumsum(returns))  # NQ-like prices

    # Create DataFrame with required columns
    data = pd.DataFrame({
        'close': close,
        'close_norm': (close - close.min()) / (close.max() - close.min()),
        'high': close * (1 + np.abs(np.random.randn(n_rows) * 0.001)),
        'low': close * (1 - np.abs(np.random.randn(n_rows) * 0.001)),
        'volume': np.random.randint(1000, 10000, n_rows),

        # Technical indicators (normalized)
        'RSI': np.random.rand(n_rows) * 100,  # 0-100
        'ADX': np.random.rand(n_rows) * 50,   # 0-50
        'MACD': np.random.randn(n_rows) * 0.1,
        'MACD_SIGNAL': np.random.randn(n_rows) * 0.1,
        'MACD_HIST': np.random.randn(n_rows) * 0.05,
        'ATR': np.random.rand(n_rows) * 50 + 10,
        'VOLUME_MA': np.random.rand(n_rows),
        'DOW_SIN': np.sin(np.linspace(0, 4*np.pi, n_rows)),
        'DOW_COS': np.cos(np.linspace(0, 4*np.pi, n_rows)),
        'MSO_SIN': np.sin(np.linspace(0, 20*np.pi, n_rows)),
        'MSO_COS': np.cos(np.linspace(0, 20*np.pi, n_rows)),
        'ROLLING_DD': np.random.rand(n_rows) * 0.1,
        'VOL_PERCENTILE': np.random.rand(n_rows),
    })

    # Add LSTM features
    for i in range(8):
        data[f'LSTM_F{i}'] = np.random.randn(n_rows) * 0.5

    return data


def create_env(data):
    """Create a vectorized environment."""
    env_kwargs = dict(
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1),
    )
    return DummyVecEnv([lambda: TradingEnv(data.copy(), **env_kwargs)])


def benchmark_device(device: str, data, timesteps: int = 5000) -> dict:
    """Benchmark RecurrentPPO on a specific device."""
    print(f"\n{'='*60}")
    print(f"Benchmarking on: {device.upper()}")
    print(f"{'='*60}")

    env = create_env(data)

    seq_config = config.get("sequence_model", {})

    # Create model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=0,
        device=device,
        learning_rate=3e-4,
        n_steps=128,  # Smaller for faster benchmark
        batch_size=64,
        policy_kwargs={
            "lstm_hidden_size": seq_config.get("lstm_hidden_size", 256),
            "n_lstm_layers": seq_config.get("n_lstm_layers", 1),
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
        }
    )

    # Warmup run (first run can be slower due to JIT compilation)
    print("Warmup run...")
    warmup_start = time.perf_counter()
    model.learn(total_timesteps=500, progress_bar=False)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s")

    # Benchmark run
    print(f"Benchmark run ({timesteps} timesteps)...")
    start_time = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed_time = time.perf_counter() - start_time

    steps_per_second = timesteps / elapsed_time

    # Memory usage (CUDA only)
    memory_mb = None
    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    results = {
        "device": device,
        "timesteps": timesteps,
        "elapsed_time": elapsed_time,
        "steps_per_second": steps_per_second,
        "memory_mb": memory_mb,
    }

    print(f"\nResults for {device.upper()}:")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Steps/sec: {steps_per_second:.1f}")
    if memory_mb:
        print(f"  GPU Memory: {memory_mb:.1f} MB")

    env.close()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    print("RecurrentPPO Device Benchmark")
    print("=" * 60)

    # Check available devices
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Create synthetic data (no pandas_ta dependency)
    print("\nCreating synthetic benchmark data...")
    data = create_synthetic_data(n_rows=5000)
    print(f"Data shape: {data.shape}")

    # Benchmark settings
    timesteps = 5000  # Limited steps for quick benchmark

    results = {}

    # Benchmark CPU
    results["cpu"] = benchmark_device("cpu", data, timesteps)

    # Benchmark CUDA if available
    if cuda_available:
        results["cuda"] = benchmark_device("cuda", data, timesteps)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\n{'Device':<10} {'Time (s)':<12} {'Steps/sec':<12} {'Memory (MB)':<12}")
    print("-" * 50)

    for device, r in results.items():
        mem_str = f"{r['memory_mb']:.1f}" if r['memory_mb'] else "N/A"
        print(f"{device.upper():<10} {r['elapsed_time']:<12.2f} {r['steps_per_second']:<12.1f} {mem_str:<12}")

    # Calculate speedup
    if cuda_available:
        speedup = results["cpu"]["elapsed_time"] / results["cuda"]["elapsed_time"]
        print(f"\n{'='*60}")
        print(f"CUDA SPEEDUP: {speedup:.2f}x faster than CPU")
        print(f"{'='*60}")

        if speedup > 1:
            print(f"\nCUDA is {speedup:.2f}x faster ({results['cuda']['steps_per_second']:.1f} vs {results['cpu']['steps_per_second']:.1f} steps/sec)")
        else:
            print(f"\nCPU is {1/speedup:.2f}x faster ({results['cpu']['steps_per_second']:.1f} vs {results['cuda']['steps_per_second']:.1f} steps/sec)")

    return results


if __name__ == "__main__":
    main()
