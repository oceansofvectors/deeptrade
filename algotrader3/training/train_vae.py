"""
VAE Training Script.

Phase 2 of World Model training:
Train the VAE to compress observations into latent space.
"""

import logging
import os
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models.vae import VAE, create_vae

logger = logging.getLogger(__name__)


def train_vae(
    observations: np.ndarray,
    obs_dim: int = 30,
    latent_dim: int = 32,
    hidden_dims: list = [256, 128],
    beta: float = 0.001,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 10,
    device: str = 'auto',
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[VAE, Dict[str, list]]:
    """
    Train VAE on collected observations.

    Args:
        observations: Observation data [N, obs_dim]
        obs_dim: Observation dimension
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        beta: KL divergence weight
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of training epochs
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps')
        save_path: Path to save trained model
        verbose: Whether to show progress

    Returns:
        Tuple of (trained VAE, training history)
    """
    # Setup device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    device = torch.device(device)
    logger.info(f"Training VAE on {device}")

    # Create VAE
    vae = create_vae(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta
    ).to(device)

    # Create dataloader
    dataset = TensorDataset(torch.tensor(observations, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }

    # Training loop
    logger.info(f"Starting VAE training: {epochs} epochs, {len(observations)} samples")

    for epoch in range(epochs):
        vae.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        num_batches = 0

        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader

        for batch in iterator:
            x = batch[0].to(device)

            # Forward pass
            x_recon, z, mu, logvar = vae(x)

            # Compute loss
            total_loss, recon_loss, kl_loss = vae.loss_function(x, x_recon, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            num_batches += 1

            if verbose:
                iterator.set_postfix({
                    'loss': total_loss.item(),
                    'recon': recon_loss.item(),
                    'kl': kl_loss.item()
                })

        # Average losses
        avg_total = epoch_losses['total'] / num_batches
        avg_recon = epoch_losses['recon'] / num_batches
        avg_kl = epoch_losses['kl'] / num_batches

        history['total_loss'].append(avg_total)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)

        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_total:.4f}, "
                   f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': vae.state_dict(),
            'config': {
                'obs_dim': obs_dim,
                'latent_dim': latent_dim,
                'hidden_dims': hidden_dims,
                'beta': beta
            },
            'history': history
        }, save_path)
        logger.info(f"VAE saved to {save_path}")

    return vae, history


def load_vae(path: str, device: str = 'auto') -> VAE:
    """
    Load a trained VAE from disk.

    Args:
        path: Path to saved model
        device: Device to load on

    Returns:
        Loaded VAE
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint['config']
    vae = create_vae(
        obs_dim=config['obs_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        beta=config['beta']
    )
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()

    logger.info(f"Loaded VAE from {path}")
    return vae


def encode_observations(
    vae: VAE,
    observations: np.ndarray,
    batch_size: int = 1024,
    device: str = 'auto',
    deterministic: bool = True
) -> np.ndarray:
    """
    Encode all observations to latent vectors.

    Args:
        vae: Trained VAE
        observations: Observations [N, obs_dim]
        batch_size: Batch size for encoding
        device: Device
        deterministic: If True, use mu instead of sampling

    Returns:
        Latent vectors [N, latent_dim]
    """
    if device == 'auto':
        device = next(vae.parameters()).device

    vae.eval()
    latents = []

    with torch.no_grad():
        for i in range(0, len(observations), batch_size):
            batch = torch.tensor(
                observations[i:i+batch_size],
                dtype=torch.float32,
                device=device
            )
            z = vae.encode(batch, deterministic=deterministic)
            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)
