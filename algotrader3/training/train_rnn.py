"""
MDN-RNN Training Script.

Phase 3 of World Model training:
Train the MDN-RNN to predict next latent state distribution.
"""

import logging
import os
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models.mdn_rnn import MDNRNN, create_mdn_rnn

logger = logging.getLogger(__name__)


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for sequences of (z, action) pairs."""

    def __init__(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        sequence_length: int = 50
    ):
        """
        Initialize sequence dataset.

        Args:
            latents: Latent vectors [N, latent_dim]
            actions: Actions [N]
            sequence_length: Length of each sequence
        """
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.seq_len = sequence_length

        # Calculate number of valid sequences
        self.num_sequences = len(latents) - sequence_length

    def __len__(self):
        return max(0, self.num_sequences)

    def __getitem__(self, idx):
        z_seq = self.latents[idx:idx + self.seq_len]
        a_seq = self.actions[idx:idx + self.seq_len]
        z_next = self.latents[idx + 1:idx + self.seq_len + 1]

        return z_seq, a_seq, z_next


def train_mdnrnn(
    latents: np.ndarray,
    actions: np.ndarray,
    latent_dim: int = 32,
    action_dim: int = 3,
    hidden_dim: int = 256,
    num_layers: int = 1,
    num_mixtures: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    sequence_length: int = 50,
    epochs: int = 20,
    grad_clip: float = 1.0,
    device: str = 'auto',
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[MDNRNN, Dict[str, list]]:
    """
    Train MDN-RNN on latent sequences.

    Args:
        latents: Encoded latent vectors [N, latent_dim]
        actions: Actions taken [N]
        latent_dim: Latent space dimension
        action_dim: Number of actions
        hidden_dim: RNN hidden dimension
        num_layers: Number of RNN layers
        num_mixtures: Number of Gaussian mixtures
        learning_rate: Learning rate
        batch_size: Batch size
        sequence_length: Sequence length for training
        epochs: Number of training epochs
        grad_clip: Gradient clipping value
        device: Device to train on
        save_path: Path to save model
        verbose: Show progress

    Returns:
        Tuple of (trained MDN-RNN, training history)
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
    logger.info(f"Training MDN-RNN on {device}")

    # Create MDN-RNN
    mdn_rnn = create_mdn_rnn(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_mixtures=num_mixtures
    ).to(device)

    # Create dataset and dataloader
    dataset = SequenceDataset(latents, actions, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty. Need at least {sequence_length} samples.")

    # Optimizer
    optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=learning_rate)

    # Training history
    history = {'loss': []}

    # Training loop
    logger.info(f"Starting MDN-RNN training: {epochs} epochs, {len(dataset)} sequences")

    for epoch in range(epochs):
        mdn_rnn.train()
        epoch_loss = 0
        num_batches = 0

        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader

        for z_seq, a_seq, z_next in iterator:
            z_seq = z_seq.to(device)
            a_seq = a_seq.to(device)
            z_next = z_next.to(device)

            # Initialize hidden state
            hidden = mdn_rnn.initial_hidden(batch_size, device)

            # Forward pass
            pi, mu, sigma, _ = mdn_rnn(z_seq, a_seq, hidden)

            # Compute loss (predict z_next from z_seq)
            # We want to predict z_{t+1} from z_t, a_t
            # So z_next[t] should match prediction from (z_seq[t], a_seq[t])
            loss = mdn_rnn.loss(z_next, pi, mu, sigma)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if verbose:
                iterator.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save model
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
            },
            'history': history
        }, save_path)
        logger.info(f"MDN-RNN saved to {save_path}")

    return mdn_rnn, history


def load_mdnrnn(path: str, device: str = 'auto') -> MDNRNN:
    """
    Load a trained MDN-RNN from disk.

    Args:
        path: Path to saved model
        device: Device to load on

    Returns:
        Loaded MDN-RNN
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
    mdn_rnn = create_mdn_rnn(
        latent_dim=config['latent_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_mixtures=config['num_mixtures']
    )
    mdn_rnn.load_state_dict(checkpoint['model_state_dict'])
    mdn_rnn.to(device)
    mdn_rnn.eval()

    logger.info(f"Loaded MDN-RNN from {path}")
    return mdn_rnn
