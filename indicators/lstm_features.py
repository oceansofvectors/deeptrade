"""
LSTM Feature Extractor for noise filtering and pattern extraction.

Based on the cascaded LSTM approach from:
"A Novel Deep Reinforcement Learning Based Automated Stock Trading System
Using Cascaded LSTM Networks" (arXiv:2212.02721)

This module creates an LSTM encoder that:
1. Takes raw OHLCV data as input
2. PRE-TRAINS using an autoencoder to learn meaningful representations
3. Filters noise from the low signal-to-noise ratio financial data
4. Outputs learned features that complement hand-crafted indicators
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, Optional
import logging
import pickle

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for pre-training feature extraction.

    Architecture:
    - Encoder: LSTM that compresses sequence into latent representation
    - Decoder: LSTM that reconstructs the original sequence

    The encoder's latent representation becomes our learned features.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 32,
        num_layers: int = 1,
        latent_size: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Latent projection (encoder output)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Decoder input projection
        self.decoder_fc = nn.Linear(latent_size, hidden_size)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_fc = nn.Linear(hidden_size, input_size)

        # Activation for bounded latent space
        self.tanh = nn.Tanh()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Latent tensor of shape (batch, latent_size)
        """
        # Encode sequence
        _, (h_n, _) = self.encoder_lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Project to latent space
        latent = self.encoder_fc(last_hidden)

        # Bound to [-1, 1]
        latent = self.tanh(latent)

        return latent

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation back to sequence.

        Args:
            latent: Latent tensor of shape (batch, latent_size)
            seq_len: Length of sequence to reconstruct

        Returns:
            Reconstructed tensor of shape (batch, seq_len, input_size)
        """
        batch_size = latent.shape[0]

        # Project latent to decoder input
        decoder_input = self.decoder_fc(latent)  # (batch, hidden_size)

        # Repeat for each timestep
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)

        # Decode sequence
        decoder_out, _ = self.decoder_lstm(decoder_input)

        # Project to output
        output = self.output_fc(decoder_out)

        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Tuple of (reconstructed, latent)
        """
        seq_len = x.shape[1]
        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed, latent


class LSTMFeatureExtractor(nn.Module):
    """
    Wrapper that uses the encoder part of a trained autoencoder.
    """

    def __init__(self, autoencoder: LSTMAutoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)


class LSTMFeatureGenerator:
    """
    Wrapper class to generate LSTM features for a DataFrame.

    Handles:
    - Sliding window creation
    - Autoencoder pre-training on training data
    - Feature extraction using trained encoder
    - Saving/loading models
    """

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 8,
        device: str = "auto",
        pretrain_epochs: int = 50,
        pretrain_lr: float = 0.001,
        pretrain_batch_size: int = 64,
        pretrain_patience: int = 10
    ):
        """
        Initialize the LSTM feature generator.

        Args:
            lookback: Number of bars to look back for LSTM input
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            output_size: Number of output features (latent size)
            device: Device to use ("auto", "cuda", "mps", "cpu")
            pretrain_epochs: Number of epochs for autoencoder pre-training
            pretrain_lr: Learning rate for pre-training
            pretrain_batch_size: Batch size for pre-training
            pretrain_patience: Early stopping patience (epochs without improvement)
        """
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_batch_size = pretrain_batch_size
        self.pretrain_patience = pretrain_patience

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                # MPS support for LSTMs improved in PyTorch 2.0+
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) for LSTM training")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Input features: returns, high-low range, close-open, volume change, volatility
        self.input_size = 5

        # Initialize autoencoder
        self.autoencoder = LSTMAutoencoder(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_size=output_size
        ).to(self.device)

        # Statistics for normalization
        self.input_mean = None
        self.input_std = None
        self.is_pretrained = False

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare input features from OHLCV data.

        Converts raw OHLCV to normalized features that are more suitable
        for LSTM processing.
        """
        # Get column names (handle different capitalizations)
        close_col = 'close' if 'close' in df.columns else 'Close'
        open_col = 'open' if 'open' in df.columns else 'Open'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'

        # Calculate normalized features
        returns = df[close_col].pct_change().fillna(0).values
        hl_range = ((df[high_col] - df[low_col]) / df[close_col]).fillna(0).values
        co_diff = ((df[close_col] - df[open_col]) / df[close_col]).fillna(0).values
        vol_change = df[vol_col].pct_change().fillna(0).values

        # Rolling volatility (std of returns over short window)
        volatility = pd.Series(returns).rolling(5, min_periods=1).std().fillna(0).values

        # Stack features
        features = np.column_stack([returns, hl_range, co_diff, vol_change, volatility])

        return features

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """
        Create sliding window sequences for LSTM input.

        Args:
            features: Array of shape (n_samples, n_features)

        Returns:
            Sequences of shape (n_samples - lookback + 1, lookback, n_features)
        """
        n_samples = len(features)
        sequences = []

        for i in range(self.lookback - 1, n_samples):
            seq = features[i - self.lookback + 1:i + 1]
            sequences.append(seq)

        return np.array(sequences)

    def _pretrain_autoencoder(self, sequences: np.ndarray, checkpoint_path: str = None):
        """
        Pre-train the autoencoder to learn meaningful representations.

        Uses reconstruction loss to train the encoder to capture
        important patterns in the data.

        Args:
            sequences: Training sequences
            checkpoint_path: Path to save best model checkpoint
        """
        logger.info(f"Pre-training LSTM autoencoder for up to {self.pretrain_epochs} epochs...")

        # Create dataset
        x_tensor = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.pretrain_batch_size,
            shuffle=True
        )

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.pretrain_lr)
        criterion = nn.MSELoss()

        # Training loop
        self.autoencoder.train()
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        patience = self.pretrain_patience
        avg_loss = 0.0

        for epoch in range(self.pretrain_epochs):
            total_loss = 0
            num_batches = len(dataloader)

            # Progress bar for batches within epoch
            batch_pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.pretrain_epochs}",
                unit="batch",
                leave=False
            )

            for (batch_x,) in batch_pbar:
                optimizer.zero_grad()

                # Forward pass
                reconstructed, latent = self.autoencoder(batch_x)

                # Reconstruction loss
                loss = criterion(reconstructed, batch_x)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update batch progress bar
                batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

            batch_pbar.close()
            avg_loss = total_loss / num_batches

            # Log epoch results
            logger.info(f"  Epoch {epoch + 1}/{self.pretrain_epochs} - Loss: {avg_loss:.6f} (best: {best_loss:.6f})")

            # Early stopping check and checkpointing
            if avg_loss < best_loss - 0.0001:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in self.autoencoder.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            self.autoencoder.load_state_dict(best_state)
            logger.info(f"Restored best model with loss: {best_loss:.6f}")

        # Save checkpoint if path provided
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, best_loss)

        logger.info(f"Pre-training complete. Best loss: {best_loss:.6f}")
        self.is_pretrained = True

    def _save_checkpoint(self, path: str, loss: float):
        """Save a training checkpoint."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'autoencoder_state': self.autoencoder.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'lookback': self.lookback,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'input_size': self.input_size,
            'is_pretrained': self.is_pretrained,
            'best_loss': loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (loss: {loss:.6f})")

    def fit(self, df: pd.DataFrame, checkpoint_path: str = None):
        """
        Fit the feature extractor on training data.

        This includes:
        1. Computing normalization statistics
        2. Pre-training the autoencoder
        3. Saving checkpoint if path provided

        Args:
            df: Training DataFrame with OHLCV data
            checkpoint_path: Optional path to save the trained model checkpoint
        """
        # Prepare input features
        features = self._prepare_input(df)

        # Compute normalization statistics
        self.input_mean = np.mean(features, axis=0)
        self.input_std = np.std(features, axis=0) + 1e-8

        # Normalize
        features_norm = (features - self.input_mean) / self.input_std
        features_norm = np.clip(features_norm, -5, 5)

        # Create sequences
        sequences = self._create_sequences(features_norm)

        # Pre-train the autoencoder
        self._pretrain_autoencoder(sequences, checkpoint_path=checkpoint_path)

        logger.info(f"LSTM feature extractor fitted and pre-trained on {len(df)} samples")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate LSTM features for the given DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional LSTM_F0, LSTM_F1, ... columns
        """
        if self.input_mean is None:
            raise ValueError("Must call fit() before transform()")

        if not self.is_pretrained:
            logger.warning("Autoencoder was not pre-trained! Features may be meaningless.")

        # Prepare and normalize input
        features = self._prepare_input(df)
        features = (features - self.input_mean) / self.input_std

        # Clip extreme values
        features = np.clip(features, -5, 5)

        # Create sequences
        sequences = self._create_sequences(features)

        # Convert to tensor
        x = torch.FloatTensor(sequences).to(self.device)

        # Generate features using trained encoder
        self.autoencoder.eval()
        with torch.no_grad():
            lstm_features = self.autoencoder.encode(x).cpu().numpy()

        # Create output DataFrame
        result = df.copy()

        # Add NaN padding for the lookback period
        padding = np.full((self.lookback - 1, self.output_size), np.nan)
        all_features = np.vstack([padding, lstm_features])

        # Add columns
        for i in range(self.output_size):
            result[f'LSTM_F{i}'] = all_features[:, i]

        # Backward fill NaN values (use first valid value)
        for i in range(self.output_size):
            result[f'LSTM_F{i}'] = result[f'LSTM_F{i}'].bfill()

        logger.debug(f"Generated {self.output_size} LSTM features for {len(df)} samples")

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        """Save the model and normalizer statistics."""
        state = {
            'autoencoder_state': self.autoencoder.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'lookback': self.lookback,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'input_size': self.input_size,
            'is_pretrained': self.is_pretrained
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"LSTM feature extractor saved to {path}")

    def load(self, path: str):
        """Load a saved model and normalizer statistics."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.input_mean = state['input_mean']
        self.input_std = state['input_std']
        self.lookback = state['lookback']
        self.hidden_size = state['hidden_size']
        self.num_layers = state['num_layers']
        self.output_size = state['output_size']
        self.input_size = state['input_size']
        self.is_pretrained = state.get('is_pretrained', False)

        # Recreate autoencoder with correct dimensions
        self.autoencoder = LSTMAutoencoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            latent_size=self.output_size
        ).to(self.device)

        self.autoencoder.load_state_dict(state['autoencoder_state'])
        logger.info(f"LSTM feature extractor loaded from {path}")


def calculate_lstm_features(
    df: pd.DataFrame,
    lookback: int = 20,
    hidden_size: int = 32,
    num_layers: int = 1,
    output_size: int = 8,
    generator: Optional[LSTMFeatureGenerator] = None
) -> Tuple[pd.DataFrame, LSTMFeatureGenerator]:
    """
    Calculate LSTM features for a DataFrame.

    This is the main entry point for adding LSTM features to data.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look back
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        output_size: Number of output features
        generator: Optional pre-fitted generator (for val/test data)

    Returns:
        Tuple of (DataFrame with LSTM features, fitted generator)
    """
    if generator is None:
        generator = LSTMFeatureGenerator(
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        result = generator.fit_transform(df)
    else:
        result = generator.transform(df)

    return result, generator
