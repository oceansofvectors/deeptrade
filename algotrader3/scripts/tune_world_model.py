#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for World Model.

Tunes VAE and MDN-RNN hyperparameters using Optuna.
Controller uses default CMA-ES parameters.

Usage:
    # Full VAE + RNN tuning (recommended)
    python -m algotrader3.scripts.tune_world_model --vae-trials 10 --rnn-trials 10

    # Quick tuning (fewer trials)
    python -m algotrader3.scripts.tune_world_model --vae-trials 5 --rnn-trials 5

    # VAE only
    python -m algotrader3.scripts.tune_world_model --phase vae --trials 10

    # RNN only (requires existing best_vae.pt)
    python -m algotrader3.scripts.tune_world_model --phase rnn --trials 10

    # Then train controller with tuned VAE/RNN
    python -m algotrader3.scripts.train_world_model --skip-vae --skip-rnn
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algotrader3.tuning import WorldModelTuner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Tune World Model hyperparameters with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full tuning (VAE + RNN)
  python -m algotrader3.scripts.tune_world_model --vae-trials 10 --rnn-trials 10

  # Quick test
  python -m algotrader3.scripts.tune_world_model --vae-trials 3 --rnn-trials 3

  # After tuning, train controller with tuned models:
  python -m algotrader3.scripts.train_world_model --skip-vae --skip-rnn
        """
    )

    parser.add_argument(
        '--data', type=str, default='../data/NQ_2024_unix.csv',
        help='Path to data file (default: ../data/NQ_2024_unix.csv)'
    )
    parser.add_argument(
        '--output', type=str, default='./checkpoints/tuning',
        help='Output directory for tuning results (default: ./checkpoints/tuning)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for training (default: auto)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    # Tuning options
    parser.add_argument(
        '--phase', type=str, default='all',
        choices=['all', 'vae', 'rnn'],
        help='Which phase to tune (default: all)'
    )
    parser.add_argument(
        '--vae-trials', type=int, default=10,
        help='Number of VAE tuning trials (default: 10)'
    )
    parser.add_argument(
        '--rnn-trials', type=int, default=10,
        help='Number of RNN tuning trials (default: 10)'
    )
    parser.add_argument(
        '--trials', type=int, default=None,
        help='Number of trials (overrides --vae-trials and --rnn-trials)'
    )
    parser.add_argument(
        '--timeout', type=int, default=None,
        help='Timeout per phase in seconds (optional)'
    )

    args = parser.parse_args()

    # Handle --trials override
    vae_trials = args.trials if args.trials else args.vae_trials
    rnn_trials = args.trials if args.trials else args.rnn_trials

    logger.info("=" * 60)
    logger.info("WORLD MODEL HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Phase: {args.phase}")
    if args.phase in ['all', 'vae']:
        logger.info(f"VAE trials: {vae_trials}")
    if args.phase in ['all', 'rnn']:
        logger.info(f"RNN trials: {rnn_trials}")
    logger.info("=" * 60)

    # Create tuner
    tuner = WorldModelTuner(
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        seed=args.seed
    )

    # Prepare data
    tuner.prepare_data()

    # Run tuning based on phase
    if args.phase == 'all':
        best_params = tuner.run_full_tuning(
            vae_trials=vae_trials,
            rnn_trials=rnn_trials,
            timeout_per_phase=args.timeout
        )
        logger.info("\nBest hyperparameters:")
        logger.info(f"  VAE: {best_params['vae']}")
        logger.info(f"  RNN: {best_params['rnn']}")

    elif args.phase == 'vae':
        vae_params, vae_loss = tuner.tune_vae(
            n_trials=vae_trials,
            timeout=args.timeout
        )
        logger.info(f"\nBest VAE params: {vae_params}")
        logger.info(f"Best VAE loss: {vae_loss:.6f}")

    elif args.phase == 'rnn':
        # For RNN-only tuning, need to load existing VAE
        vae_path = Path(args.output) / 'best_vae.pt'
        if not vae_path.exists():
            logger.error(f"VAE not found at {vae_path}")
            logger.error("Run VAE tuning first: --phase vae")
            sys.exit(1)

        # Load existing VAE and params
        import torch
        import yaml

        checkpoint = torch.load(vae_path, map_location='cpu', weights_only=False)
        tuner.best_vae_params = {
            'latent_dim': checkpoint['config']['latent_dim'],
            'hidden_dims': '_'.join(str(x) for x in checkpoint['config']['hidden_dims']),
            'beta': checkpoint['config']['beta']
        }

        from algotrader3.models.vae import create_vae
        tuner.best_vae = create_vae(
            obs_dim=tuner.obs_dim,
            latent_dim=checkpoint['config']['latent_dim'],
            hidden_dims=checkpoint['config']['hidden_dims'],
            beta=checkpoint['config']['beta']
        )
        tuner.best_vae.load_state_dict(checkpoint['model_state_dict'])
        tuner.best_vae.to(tuner.device)
        tuner.best_vae.eval()

        logger.info(f"Loaded VAE from {vae_path}")

        rnn_params, rnn_loss = tuner.tune_rnn(
            n_trials=rnn_trials,
            timeout=args.timeout
        )
        logger.info(f"\nBest RNN params: {rnn_params}")
        logger.info(f"Best RNN NLL: {rnn_loss:.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output}")
    logger.info("\nNext steps:")
    logger.info("  1. Copy best_vae.pt and best_rnn.pt to ./checkpoints/")
    logger.info("  2. Run: python -m algotrader3.scripts.train_world_model --skip-vae --skip-rnn")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
