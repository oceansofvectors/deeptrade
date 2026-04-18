"""LSTM/VAE feature extractor for market-feature compression."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, Optional
import logging
import pickle
from config import config

logger = logging.getLogger(__name__)


MODEL_FORMAT_VERSION = "lstm_vae_v3"


def _clear_cuda_cache() -> None:
    """Best-effort CUDA cache cleanup after large trial allocations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed_lstm_trial(seed: int) -> None:
    """Reset RNG state before each LSTM tuning trial."""
    from utils.seeding import enable_full_determinism, set_global_seed

    deterministic_mode = bool(config.get("reproducibility", {}).get("deterministic_mode", True))
    if deterministic_mode:
        enable_full_determinism(seed)
    else:
        set_global_seed(seed)


class LSTMVAE(nn.Module):
    """LSTM variational autoencoder used for pre-training feature extraction."""

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

        # Latent posterior heads
        self.encoder_mu = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)

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

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to posterior parameters."""
        _, (h_n, _) = self.encoder_lstm(x)
        last_hidden = h_n[-1]
        mu = self.encoder_mu(last_hidden)
        logvar = torch.clamp(self.encoder_logvar(last_hidden), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vector using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence deterministically to posterior mean."""
        mu, _ = self.encode_stats(x)
        return mu

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass returning reconstruction and posterior stats."""
        seq_len = x.shape[1]
        mu, logvar = self.encode_stats(x)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed, mu, logvar


class LSTMFeatureExtractor(nn.Module):
    """
    Wrapper that uses the encoder part of a trained autoencoder.
    """

    def __init__(self, autoencoder: LSTMVAE):
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
        feature_columns: Optional[list[str]] = None,
        feature_prefix: str = "LATENT_F",
        device: str = "auto",
        pretrain_epochs: int = 50,
        pretrain_lr: float = 0.001,
        pretrain_batch_size: int = 64,
        pretrain_patience: int = 10,
        pretrain_min_delta: float = 0.0001,
        pretrain_verbose: bool = True,
        beta: float = 0.001,
        kl_warmup_epochs: int = 10,
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
            pretrain_min_delta: Minimum loss improvement required to reset patience
            pretrain_verbose: Whether to log per-epoch autoencoder progress
            beta: KL divergence weight for Beta-VAE pretraining
            kl_warmup_epochs: Number of epochs to ramp KL weight from 0 to beta
        """
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.feature_prefix = str(feature_prefix)
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_batch_size = pretrain_batch_size
        self.pretrain_patience = pretrain_patience
        self.pretrain_min_delta = pretrain_min_delta
        self.pretrain_verbose = pretrain_verbose
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs

        # Determine device (use centralized logic, avoid MPS for LSTM)
        from utils.device import get_device
        resolved = get_device(device, for_recurrent=True)
        self.device = torch.device(resolved)

        self.input_size = len(self.feature_columns) if self.feature_columns is not None else 0
        self.autoencoder = None
        if self.input_size > 0:
            self._rebuild_autoencoder()

        # Statistics for normalization
        self.input_mean = None
        self.input_std = None
        self.is_pretrained = False

    def _rebuild_autoencoder(self) -> None:
        """Instantiate the autoencoder for the current input width."""
        if self.input_size <= 0:
            raise ValueError("LSTMFeatureGenerator input_size must be positive before building the autoencoder")
        self.autoencoder = LSTMVAE(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            latent_size=self.output_size,
        ).to(self.device)

    def _resolve_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Resolve and persist the ordered encoder feature columns."""
        if self.feature_columns is None:
            inferred = []
            for col in df.columns:
                if str(col).endswith("_RAW"):
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    inferred.append(col)
            if not inferred:
                raise ValueError("No numeric feature columns available for LSTM/VAE input")
            self.feature_columns = inferred
        return list(self.feature_columns)

    def _validation_batch_size(self) -> int:
        """Use smaller batches for eval to avoid single-pass CUDA spikes."""
        return max(1, min(self.pretrain_batch_size, 256))

    def _evaluate_sequences_loss(self, sequences: np.ndarray, kl_weight: float) -> float:
        """Evaluate VAE loss over sequences in small batches."""
        if sequences is None or len(sequences) == 0:
            raise ValueError("Validation sequences are empty")

        dataset = TensorDataset(torch.FloatTensor(sequences))
        loader = DataLoader(dataset, batch_size=self._validation_batch_size(), shuffle=False)
        total_loss = 0.0
        total_items = 0

        self.autoencoder.eval()
        with torch.no_grad():
            for (batch_x_cpu,) in loader:
                batch_x = batch_x_cpu.to(self.device)
                reconstructed, mu, logvar = self.autoencoder(batch_x)
                loss = self._vae_loss(reconstructed, batch_x, mu, logvar, kl_weight)[0]
                batch_size = batch_x.shape[0]
                total_loss += loss.item() * batch_size
                total_items += batch_size

        return total_loss / max(1, total_items)

    def _encode_sequences_batched(self, sequences: np.ndarray) -> np.ndarray:
        """Encode sequences in batches to control GPU memory usage."""
        dataset = TensorDataset(torch.FloatTensor(sequences))
        loader = DataLoader(dataset, batch_size=self._validation_batch_size(), shuffle=False)
        outputs = []

        self.autoencoder.eval()
        with torch.no_grad():
            for (batch_x_cpu,) in loader:
                batch_x = batch_x_cpu.to(self.device)
                outputs.append(self.autoencoder.encode(batch_x).cpu().numpy())

        if not outputs:
            return np.empty((0, self.output_size), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _current_kl_weight(self, epoch_idx: int) -> float:
        """Return the epoch-specific KL weight."""
        if self.kl_warmup_epochs <= 0:
            return self.beta
        progress = min(1.0, float(epoch_idx + 1) / float(self.kl_warmup_epochs))
        return self.beta * progress

    @staticmethod
    def _vae_loss(
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, reconstruction, and KL losses."""
        recon_loss = nn.functional.mse_loss(reconstructed, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (kl_weight * kl_loss)
        return total_loss, recon_loss, kl_loss

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare the ordered market-feature matrix for VAE training/inference."""
        feature_columns = self._resolve_feature_columns(df)
        features_df = df.reindex(columns=feature_columns, fill_value=0.0).copy()
        for col in feature_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features = features_df.to_numpy(dtype=np.float32, copy=False)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

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

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize raw features using train-only statistics."""
        if self.input_mean is None or self.input_std is None:
            raise ValueError("Normalization statistics are not initialized")

        features_norm = (features - self.input_mean) / self.input_std
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(features_norm, -5, 5)

    def _prepare_sequence_splits(
        self,
        train_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        fallback_validation_ratio: float = 0.2
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build train/validation sequences using normalization statistics fit on train only.

        If validation_df is not provided, keep a tail slice of the training sequences for
        validation so callers that only provide one frame still use a standard train/val flow.
        """
        train_features = self._prepare_input(train_df)
        self.input_size = int(train_features.shape[1])
        self._rebuild_autoencoder()
        self.input_mean = np.nan_to_num(np.mean(train_features, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        self.input_std = np.nan_to_num(np.std(train_features, axis=0), nan=1.0, posinf=1.0, neginf=1.0)
        self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)

        train_sequences = self._create_sequences(self._normalize_features(train_features))
        validation_sequences = None

        if validation_df is not None and len(validation_df) >= self.lookback:
            validation_features = self._prepare_input(validation_df)
            validation_sequences = self._create_sequences(self._normalize_features(validation_features))
        elif len(train_sequences) >= 10:
            split_idx = max(1, int(len(train_sequences) * (1 - fallback_validation_ratio)))
            split_idx = min(split_idx, len(train_sequences) - 1)
            validation_sequences = train_sequences[split_idx:]
            train_sequences = train_sequences[:split_idx]

        return train_sequences, validation_sequences

    def _pretrain_autoencoder(
        self,
        train_sequences: np.ndarray,
        validation_sequences: Optional[np.ndarray] = None,
        checkpoint_path: str = None
    ) -> float:
        """
        Pre-train the autoencoder to learn meaningful representations.

        Uses reconstruction loss to train the encoder to capture
        important patterns in the data.

        Args:
            train_sequences: Training sequences
            validation_sequences: Optional validation sequences for early stopping
            checkpoint_path: Path to save best model checkpoint
        """
        logger.info(f"Pre-training LSTM VAE for up to {self.pretrain_epochs} epochs...")

        if len(train_sequences) == 0:
            raise ValueError("Not enough data to build LSTM training sequences")

        train_tensor = torch.FloatTensor(train_sequences).to(self.device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.pretrain_batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.pretrain_lr)
        # Training loop
        self.autoencoder.train()
        best_state = None
        patience_counter = 0
        patience = self.pretrain_patience
        min_delta = self.pretrain_min_delta
        best_metric = float('inf')

        for epoch in range(self.pretrain_epochs):
            total_train_loss = 0.0
            num_batches = len(train_loader)

            # Progress bar for batches within epoch
            batch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.pretrain_epochs}",
                unit="batch",
                leave=False,
                disable=not self.pretrain_verbose,
            )

            for (batch_x,) in batch_pbar:
                optimizer.zero_grad()

                # Forward pass
                reconstructed, mu, logvar = self.autoencoder(batch_x)
                kl_weight = self._current_kl_weight(epoch)
                loss, _, _ = self._vae_loss(reconstructed, batch_x, mu, logvar, kl_weight)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                # Update batch progress bar
                if self.pretrain_verbose:
                    batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

            batch_pbar.close()
            avg_train_loss = total_train_loss / num_batches

            val_loss = None
            if validation_sequences is not None and len(validation_sequences) > 0:
                val_loss = self._evaluate_sequences_loss(validation_sequences, self.beta)
                self.autoencoder.train()

            monitor_loss = val_loss if val_loss is not None else avg_train_loss

            # Log epoch results
            if self.pretrain_verbose:
                if val_loss is not None:
                    logger.info(
                        f"  Epoch {epoch + 1}/{self.pretrain_epochs} - "
                        f"Train VAE Loss: {avg_train_loss:.6f}, Val VAE Loss: {val_loss:.6f} "
                        f"(best val: {best_metric:.6f})"
                    )
                else:
                    logger.info(
                        f"  Epoch {epoch + 1}/{self.pretrain_epochs} - "
                        f"Train VAE Loss: {avg_train_loss:.6f} (best: {best_metric:.6f})"
                    )

            # Early stopping check and checkpointing
            if monitor_loss < best_metric - min_delta:
                best_metric = monitor_loss
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
            logger.info(f"Restored best model with monitored loss: {best_metric:.6f}")

        # Save checkpoint if path provided
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, best_metric)

        logger.info(f"Pre-training complete. Best monitored VAE loss: {best_metric:.6f}")
        self.is_pretrained = True
        return best_metric

    def _save_checkpoint(self, path: str, loss: float):
        """Save a training checkpoint."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'model_format': MODEL_FORMAT_VERSION,
            'autoencoder_state': self.autoencoder.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'lookback': self.lookback,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'input_size': self.input_size,
            'is_pretrained': self.is_pretrained,
            'pretrain_min_delta': self.pretrain_min_delta,
            'beta': self.beta,
            'kl_warmup_epochs': self.kl_warmup_epochs,
            'best_loss': loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (loss: {loss:.6f})")

    def fit(
        self,
        train_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        checkpoint_path: str = None
    ):
        """
        Fit the feature extractor on training data.

        This includes:
        1. Computing normalization statistics from train data only
        2. Pre-training the autoencoder from scratch on train sequences
        3. Early stopping/model selection on validation sequences
        4. Saving checkpoint if path provided

        Args:
            train_df: Training DataFrame with OHLCV data
            validation_df: Optional validation DataFrame for monitored pretraining
            checkpoint_path: Optional path to save the trained model checkpoint
        """
        train_sequences, validation_sequences = self._prepare_sequence_splits(train_df, validation_df)
        self._pretrain_autoencoder(
            train_sequences=train_sequences,
            validation_sequences=validation_sequences,
            checkpoint_path=checkpoint_path
        )
        logger.info(
            "LSTM feature extractor fitted on %d train samples%s",
            len(train_df),
            f" with {len(validation_df)} validation samples" if validation_df is not None else ""
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate LSTM features for the given DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional LATENT_F0, LATENT_F1, ... columns
        """
        if self.input_mean is None:
            raise ValueError("Must call fit() before transform()")

        if not self.is_pretrained:
            logger.warning("Autoencoder was not pre-trained! Features may be meaningless.")

        # Prepare and normalize input
        features = self._normalize_features(self._prepare_input(df))

        # Create sequences
        sequences = self._create_sequences(features)

        # Convert to tensor
        # Generate features using trained encoder in batches to avoid large eval spikes.
        lstm_features = self._encode_sequences_batched(sequences)

        # Create output DataFrame
        result = df.copy()

        # Add NaN padding for the lookback period
        padding = np.full((self.lookback - 1, self.output_size), np.nan)
        all_features = np.vstack([padding, lstm_features])

        # Add columns
        for i in range(self.output_size):
            result[f'{self.feature_prefix}{i}'] = all_features[:, i]

        # Fill NaN values with 0.0 (neutral value since LSTM features are tanh-bounded to [-1,1])
        for i in range(self.output_size):
            result[f'{self.feature_prefix}{i}'] = result[f'{self.feature_prefix}{i}'].fillna(0.0)

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
            'model_format': MODEL_FORMAT_VERSION,
            'autoencoder_state': self.autoencoder.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'lookback': self.lookback,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'input_size': self.input_size,
            'is_pretrained': self.is_pretrained,
            'pretrain_min_delta': self.pretrain_min_delta,
            'beta': self.beta,
            'kl_warmup_epochs': self.kl_warmup_epochs,
            'feature_columns': self.feature_columns,
            'feature_prefix': self.feature_prefix,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"LSTM feature extractor saved to {path}")

    def load(self, path: str):
        """Load a saved model and normalizer statistics."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        if state.get('model_format') != MODEL_FORMAT_VERSION:
            raise ValueError(
                "Unsupported LSTM generator bundle format. Regenerate LSTM artifacts with the new VAE feature generator."
            )

        self.input_mean = state['input_mean']
        self.input_std = state['input_std']
        self.lookback = state['lookback']
        self.hidden_size = state['hidden_size']
        self.num_layers = state['num_layers']
        self.output_size = state['output_size']
        self.input_size = state['input_size']
        self.is_pretrained = state.get('is_pretrained', False)
        self.pretrain_min_delta = state.get('pretrain_min_delta', 0.0001)
        self.beta = state.get('beta', 0.001)
        self.kl_warmup_epochs = state.get('kl_warmup_epochs', 10)
        self.feature_columns = state.get('feature_columns')
        self.feature_prefix = state.get('feature_prefix', 'LATENT_F')

        # Recreate VAE with correct dimensions
        self._rebuild_autoencoder()

        self.autoencoder.load_state_dict(state['autoencoder_state'])
        logger.info(f"LSTM feature extractor loaded from {path}")


def calculate_lstm_features(
    df: pd.DataFrame,
    lookback: int = 20,
    hidden_size: int = 32,
    num_layers: int = 1,
    output_size: int = 8,
    feature_columns: Optional[list[str]] = None,
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
            output_size=output_size,
            feature_columns=feature_columns,
        )
        result = generator.fit_transform(df)
    else:
        result = generator.transform(df)

    return result, generator


def tune_lstm_hyperparameters(
    train_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame],
    tuning_config: dict,
    base_config: dict,
    feature_columns: Optional[list[str]] = None,
    window_folder: str = None
) -> dict:
    """
    Tune LSTM autoencoder hyperparameters using Optuna.

    Args:
        train_data: Training DataFrame with OHLCV data
        validation_data: Validation DataFrame with OHLCV data
        tuning_config: Tuning configuration with parameters to search
        base_config: Base LSTM config for defaults
        window_folder: Optional folder to save study results

    Returns:
        Dictionary of best hyperparameters
    """
    import optuna
    from optuna.samplers import TPESampler

    n_trials = tuning_config.get("n_trials", 20)
    params_config = tuning_config.get("parameters", {})

    # Determine device once (use centralized logic, avoid MPS for LSTM)
    from utils.device import get_device
    device = torch.device(get_device("auto", for_recurrent=True))

    logger.info(f"Starting LSTM hyperparameter tuning with {n_trials} trials on {device}")
    base_seed = int(config.get("seed", 42))

    def objective(trial: optuna.Trial) -> float:
        _seed_lstm_trial(base_seed)

        # Sample hyperparameters
        hidden_size_config = params_config.get("hidden_size", {})
        if "choices" in hidden_size_config:
            hidden_size = trial.suggest_categorical("hidden_size", hidden_size_config["choices"])
        else:
            hidden_size = base_config.get("hidden_size", 32)

        num_layers_config = params_config.get("num_layers", {})
        if "min" in num_layers_config:
            num_layers = trial.suggest_int("num_layers", num_layers_config["min"], num_layers_config["max"])
        else:
            num_layers = base_config.get("num_layers", 1)

        output_size_config = params_config.get("output_size", {})
        if "choices" in output_size_config:
            output_size = trial.suggest_categorical("output_size", output_size_config["choices"])
        else:
            output_size = base_config.get("output_size", 8)

        lookback_config = params_config.get("lookback", {})
        if "choices" in lookback_config:
            lookback = trial.suggest_categorical("lookback", lookback_config["choices"])
        else:
            lookback = base_config.get("lookback", 20)

        lr_config = params_config.get("pretrain_lr", {})
        if "min" in lr_config:
            pretrain_lr = trial.suggest_float(
                "pretrain_lr",
                lr_config["min"],
                lr_config["max"],
                log=lr_config.get("log", True)
            )
        else:
            pretrain_lr = base_config.get("pretrain_lr", 0.001)

        beta_config = params_config.get("beta", {})
        if "min" in beta_config:
            beta = trial.suggest_float(
                "beta",
                beta_config["min"],
                beta_config["max"],
                log=beta_config.get("log", True)
            )
        else:
            beta = base_config.get("beta", 0.001)

        # Create generator with trial parameters
        generator = LSTMFeatureGenerator(
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            feature_columns=feature_columns,
            device=str(device),
            pretrain_epochs=base_config.get("pretrain_epochs", 50),
            pretrain_lr=pretrain_lr,
            pretrain_batch_size=base_config.get("pretrain_batch_size", 64),
            pretrain_patience=base_config.get("pretrain_patience", 10),
            pretrain_min_delta=base_config.get("pretrain_min_delta", 0.0001),
            pretrain_verbose=False,
            beta=beta,
            kl_warmup_epochs=base_config.get("kl_warmup_epochs", 10),
        )

        # Train with reduced epochs for tuning
        generator.pretrain_epochs = min(20, base_config.get("pretrain_epochs", 50))
        generator.pretrain_patience = min(5, base_config.get("pretrain_patience", 10))

        train_sequences, validation_sequences = generator._prepare_sequence_splits(
            train_df=train_data,
            validation_df=validation_data
        )

        try:
            best_val_loss = generator._pretrain_autoencoder(
                train_sequences=train_sequences,
                validation_sequences=validation_sequences,
                checkpoint_path=None
            )
        except torch.OutOfMemoryError as exc:
            logger.warning("LSTM tuning trial %s hit CUDA OOM and will be pruned: %s", trial.number, exc)
            _clear_cuda_cache()
            raise optuna.TrialPruned("cuda_oom") from exc
        finally:
            _clear_cuda_cache()

        trial.report(best_val_loss, generator.pretrain_epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return best_val_loss

    # Create study
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    previous_optuna_verbosity = optuna.logging.get_verbosity()
    try:
        # Suppress noisy per-trial LSTM Optuna logs without leaking this setting
        # into the later PPO walk-forward tuning stages.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    finally:
        optuna.logging.set_verbosity(previous_optuna_verbosity)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"LSTM tuning complete. Best validation loss: {best_value:.6f}")
    logger.info(f"Best parameters: {best_params}")

    # Save study results if folder provided
    if window_folder:
        import json
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_trials
        }
        with open(f"{window_folder}/lstm_tuning_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Merge with base config (use base values for non-tuned params)
    final_params = {
        "lookback": best_params.get("lookback", base_config.get("lookback", 20)),
        "hidden_size": best_params.get("hidden_size", base_config.get("hidden_size", 32)),
        "num_layers": best_params.get("num_layers", base_config.get("num_layers", 1)),
        "output_size": best_params.get("output_size", base_config.get("output_size", 8)),
        "pretrain_lr": best_params.get("pretrain_lr", base_config.get("pretrain_lr", 0.001)),
        "beta": best_params.get("beta", base_config.get("beta", 0.001)),
        "pretrain_epochs": base_config.get("pretrain_epochs", 50),
        "pretrain_batch_size": base_config.get("pretrain_batch_size", 64),
        "pretrain_patience": base_config.get("pretrain_patience", 10),
        "pretrain_min_delta": base_config.get("pretrain_min_delta", 0.0001),
        "kl_warmup_epochs": base_config.get("kl_warmup_epochs", 10),
    }

    return final_params
