"""
MDN-RNN Hyperparameter Tuning with Optuna.

Objective: Minimize validation negative log-likelihood (NLL).
Assumes VAE is already trained and frozen.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from ..models.mdn_rnn import create_mdn_rnn, MDNRNN
from ..training.train_rnn import SequenceDataset

logger = logging.getLogger(__name__)


def rnn_objective(
    trial: optuna.Trial,
    train_latents: np.ndarray,
    train_actions: np.ndarray,
    val_latents: np.ndarray,
    val_actions: np.ndarray,
    latent_dim: int,
    action_dim: int = 3,
    device: str = 'cpu'
) -> float:
    """
    Optuna objective function for MDN-RNN hyperparameter tuning.

    Args:
        trial: Optuna trial object
        train_latents: Training latent vectors [N_train, latent_dim]
        train_actions: Training actions [N_train]
        val_latents: Validation latent vectors [N_val, latent_dim]
        val_actions: Validation actions [N_val]
        latent_dim: Dimension of latent space (from VAE)
        action_dim: Number of discrete actions
        device: Device for training

    Returns:
        Validation NLL (negative log-likelihood)
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    num_mixtures = trial.suggest_int('num_mixtures', 3, 8)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    sequence_length = trial.suggest_categorical('sequence_length', [25, 50, 100])
    epochs = trial.suggest_int('epochs', 10, 30)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    logger.info(f"Trial {trial.number}: hidden_dim={hidden_dim}, num_layers={num_layers}, "
                f"num_mixtures={num_mixtures}, lr={lr:.6f}, seq_len={sequence_length}, "
                f"epochs={epochs}, batch_size={batch_size}")

    device = torch.device(device)

    # Create MDN-RNN
    mdn_rnn = create_mdn_rnn(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_mixtures=num_mixtures
    ).to(device)

    # Create datasets
    train_dataset = SequenceDataset(train_latents, train_actions, sequence_length)
    val_dataset = SequenceDataset(val_latents, val_actions, sequence_length)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.warning(f"Trial {trial.number}: Dataset too small for seq_len={sequence_length}")
        return float('inf')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Optimizer
    optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        mdn_rnn.train()
        for z_seq, a_seq, z_next in train_loader:
            z_seq = z_seq.to(device)
            a_seq = a_seq.to(device)
            z_next = z_next.to(device)

            # Initialize hidden state
            hidden = mdn_rnn.initial_hidden(z_seq.shape[0], device)

            # Forward pass
            pi, mu, sigma, _ = mdn_rnn(z_seq, a_seq, hidden)

            # Compute NLL loss
            loss = mdn_rnn.loss(z_next, pi, mu, sigma)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), 1.0)
            optimizer.step()

        # Evaluate on validation set
        val_loss = evaluate_rnn(mdn_rnn, val_loader, device)

        # Report intermediate value for pruning
        trial.report(val_loss, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Final validation loss
    final_val_loss = evaluate_rnn(mdn_rnn, val_loader, device)
    logger.info(f"Trial {trial.number} completed: val_nll={final_val_loss:.6f}")

    return final_val_loss


def evaluate_rnn(mdn_rnn: MDNRNN, dataloader: DataLoader, device) -> float:
    """Evaluate MDN-RNN on a dataset and return average NLL."""
    mdn_rnn.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for z_seq, a_seq, z_next in dataloader:
            z_seq = z_seq.to(device)
            a_seq = a_seq.to(device)
            z_next = z_next.to(device)

            hidden = mdn_rnn.initial_hidden(z_seq.shape[0], device)
            pi, mu, sigma, _ = mdn_rnn(z_seq, a_seq, hidden)
            loss = mdn_rnn.loss(z_next, pi, mu, sigma)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_rnn_with_params(
    params: dict,
    train_latents: np.ndarray,
    train_actions: np.ndarray,
    latent_dim: int,
    action_dim: int = 3,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> MDNRNN:
    """
    Train MDN-RNN with specific hyperparameters (for final training after tuning).

    Args:
        params: Dictionary of hyperparameters from Optuna
        train_latents: Training latent vectors
        train_actions: Training actions
        latent_dim: Dimension of latent space
        action_dim: Number of discrete actions
        device: Device for training
        save_path: Optional path to save model

    Returns:
        Trained MDN-RNN model
    """
    import os

    hidden_dim = params['hidden_dim']
    num_layers = params['num_layers']
    num_mixtures = params['num_mixtures']
    lr = params['lr']
    sequence_length = params['sequence_length']
    epochs = params['epochs']
    batch_size = params['batch_size']

    device = torch.device(device)

    # Create MDN-RNN
    mdn_rnn = create_mdn_rnn(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_mixtures=num_mixtures
    ).to(device)

    # Create dataset
    train_dataset = SequenceDataset(train_latents, train_actions, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=lr)

    # Training loop
    logger.info(f"Training MDN-RNN with best params: {params}")

    for epoch in range(epochs):
        mdn_rnn.train()
        epoch_loss = 0
        num_batches = 0

        for z_seq, a_seq, z_next in train_loader:
            z_seq = z_seq.to(device)
            a_seq = a_seq.to(device)
            z_next = z_next.to(device)

            hidden = mdn_rnn.initial_hidden(z_seq.shape[0], device)
            pi, mu, sigma, _ = mdn_rnn(z_seq, a_seq, hidden)
            loss = mdn_rnn.loss(z_next, pi, mu, sigma)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} - NLL: {avg_loss:.4f}")

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': mdn_rnn.state_dict(),
            'config': {
                'latent_dim': latent_dim,
                'action_dim': action_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_mixtures': num_mixtures
            }
        }, save_path)
        logger.info(f"MDN-RNN saved to {save_path}")

    return mdn_rnn
