"""
VAE Hyperparameter Tuning with Optuna.

Objective: Minimize validation reconstruction loss + beta * KL divergence.
"""

import logging
from typing import Optional

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..models.vae import create_vae

logger = logging.getLogger(__name__)


def vae_objective(
    trial: optuna.Trial,
    train_obs: np.ndarray,
    val_obs: np.ndarray,
    obs_dim: int,
    device: str = 'cpu'
) -> float:
    """
    Optuna objective function for VAE hyperparameter tuning.

    Args:
        trial: Optuna trial object
        train_obs: Training observations [N_train, obs_dim]
        val_obs: Validation observations [N_val, obs_dim]
        obs_dim: Observation dimension
        device: Device for training

    Returns:
        Validation loss (reconstruction + beta * KL)
    """
    # Suggest hyperparameters
    latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 48, 64])
    hidden_dims_str = trial.suggest_categorical('hidden_dims', ['128_64', '256_128', '512_256'])
    beta = trial.suggest_float('beta', 0.0001, 0.01, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 5, 20)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    # Parse hidden_dims
    hidden_dims = [int(x) for x in hidden_dims_str.split('_')]

    logger.info(f"Trial {trial.number}: latent_dim={latent_dim}, hidden_dims={hidden_dims}, "
                f"beta={beta:.6f}, lr={lr:.6f}, epochs={epochs}, batch_size={batch_size}")

    device = torch.device(device)

    # Create VAE
    vae = create_vae(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta
    ).to(device)

    # Create dataloaders
    train_dataset = TensorDataset(torch.tensor(train_obs, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(val_obs, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        vae.train()
        for batch in train_loader:
            x = batch[0].to(device)

            # Forward pass
            x_recon, z, mu, logvar = vae(x)

            # Compute loss
            total_loss, _, _ = vae.loss_function(x, x_recon, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Evaluate on validation set
        val_loss = evaluate_vae(vae, val_loader, device)

        # Report intermediate value for pruning
        trial.report(val_loss, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Final validation loss
    final_val_loss = evaluate_vae(vae, val_loader, device)
    logger.info(f"Trial {trial.number} completed: val_loss={final_val_loss:.6f}")

    return final_val_loss


def evaluate_vae(vae, dataloader, device) -> float:
    """Evaluate VAE on a dataset and return average loss."""
    vae.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            x_recon, z, mu, logvar = vae(x)
            loss, _, _ = vae.loss_function(x, x_recon, mu, logvar)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_vae_with_params(
    params: dict,
    train_obs: np.ndarray,
    val_obs: np.ndarray,
    obs_dim: int,
    device: str = 'cpu',
    save_path: Optional[str] = None
):
    """
    Train VAE with specific hyperparameters (for final training after tuning).

    Args:
        params: Dictionary of hyperparameters from Optuna
        train_obs: Training observations
        val_obs: Validation observations
        obs_dim: Observation dimension
        device: Device for training
        save_path: Optional path to save model

    Returns:
        Trained VAE model
    """
    import os

    latent_dim = params['latent_dim']
    hidden_dims = [int(x) for x in params['hidden_dims'].split('_')]
    beta = params['beta']
    lr = params['lr']
    epochs = params['epochs']
    batch_size = params['batch_size']

    device = torch.device(device)

    # Create VAE
    vae = create_vae(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta
    ).to(device)

    # Create dataloaders
    train_dataset = TensorDataset(torch.tensor(train_obs, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    logger.info(f"Training VAE with best params: {params}")

    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)
            x_recon, z, mu, logvar = vae(x)
            total_loss, _, _ = vae.loss_function(x, x_recon, mu, logvar)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': vae.state_dict(),
            'config': {
                'obs_dim': obs_dim,
                'latent_dim': latent_dim,
                'hidden_dims': hidden_dims,
                'beta': beta
            }
        }, save_path)
        logger.info(f"VAE saved to {save_path}")

    return vae
