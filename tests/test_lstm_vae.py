#!/usr/bin/env python
"""Tests for the LSTM VAE feature generator."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indicators.lstm_features import (  # noqa: E402
    LSTMFeatureGenerator,
    LSTMVAE,
    _seed_lstm_trial,
    calculate_lstm_features,
    tune_lstm_hyperparameters,
)


def _make_feature_frame(rows: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 14:30:00", periods=rows, freq="5min", tz="UTC")
    close = np.linspace(100.0, 103.0, rows) + np.sin(np.linspace(0, 4, rows))
    return pd.DataFrame(
        {
            "close_norm": np.linspace(0.1, 0.9, rows),
            "feat_a": close,
            "feat_b": close * 0.05,
            "feat_c": np.linspace(-1.0, 1.0, rows),
        },
        index=idx,
    )


class TestLSTMVAE(unittest.TestCase):
    def test_forward_shapes(self):
        model = LSTMVAE(input_size=5, hidden_size=16, num_layers=1, latent_size=4)
        x = torch.randn(3, 10, 5)
        reconstructed, mu, logvar = model(x)

        self.assertEqual(reconstructed.shape, x.shape)
        self.assertEqual(mu.shape, (3, 4))
        self.assertEqual(logvar.shape, (3, 4))

    def test_encode_returns_mean_deterministically(self):
        model = LSTMVAE(input_size=5, hidden_size=16, num_layers=1, latent_size=4)
        x = torch.randn(2, 10, 5)
        mu, _ = model.encode_stats(x)
        encoded = model.encode(x)
        self.assertTrue(torch.allclose(mu, encoded))


class TestLSTMTrialSeeding(unittest.TestCase):
    def test_seed_lstm_trial_uses_deterministic_helper_when_enabled(self):
        with mock.patch("indicators.lstm_features.config", {"reproducibility": {"deterministic_mode": True}}):
            with mock.patch("utils.seeding.enable_full_determinism") as mock_enable:
                with mock.patch("utils.seeding.set_global_seed") as mock_set:
                    _seed_lstm_trial(123)
        mock_enable.assert_called_once_with(123)
        mock_set.assert_not_called()

    def test_seed_lstm_trial_uses_global_seed_when_determinism_disabled(self):
        with mock.patch("indicators.lstm_features.config", {"reproducibility": {"deterministic_mode": False}}):
            with mock.patch("utils.seeding.enable_full_determinism") as mock_enable:
                with mock.patch("utils.seeding.set_global_seed") as mock_set:
                    _seed_lstm_trial(321)
        mock_set.assert_called_once_with(321)
        mock_enable.assert_not_called()


class TestLSTMFeatureGeneratorVAE(unittest.TestCase):
    def test_prepare_input_handles_zero_and_nonfinite_bars(self):
        df = _make_feature_frame()
        df.loc[df.index[5], "feat_a"] = 0.0
        df.loc[df.index[6], "feat_b"] = np.inf
        df.loc[df.index[7], "feat_c"] = np.nan

        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            feature_columns=["close_norm", "feat_a", "feat_b", "feat_c"],
            device="cpu",
            pretrain_verbose=False,
        )
        features = generator._prepare_input(df)

        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.shape[1], 4)

    def test_prepare_input_uses_explicit_feature_columns(self):
        df = _make_feature_frame()
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            feature_columns=["feat_b", "feat_c"],
            device="cpu",
            pretrain_verbose=False,
        )

        features = generator._prepare_input(df)

        self.assertEqual(features.shape, (len(df), 2))

    def test_transform_is_deterministic(self):
        df = _make_feature_frame()
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            feature_columns=["close_norm", "feat_a", "feat_b", "feat_c"],
            device="cpu",
            pretrain_epochs=2,
            pretrain_batch_size=8,
            pretrain_patience=2,
            beta=0.001,
            kl_warmup_epochs=1,
            pretrain_verbose=False,
        )
        generator.fit(df.iloc[:32], df.iloc[32:])

        transformed_a = generator.transform(df)
        transformed_b = generator.transform(df)
        cols = [f"LATENT_F{i}" for i in range(generator.output_size)]
        pd.testing.assert_frame_equal(transformed_a[cols], transformed_b[cols])

    def test_save_load_round_trip(self):
        df = _make_feature_frame()
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            feature_columns=["close_norm", "feat_a", "feat_b", "feat_c"],
            device="cpu",
            pretrain_epochs=2,
            pretrain_batch_size=8,
            pretrain_patience=2,
            beta=0.002,
            kl_warmup_epochs=2,
            pretrain_verbose=False,
        )
        generator.fit(df.iloc[:32], df.iloc[32:])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lstm_generator.pkl")
            generator.save(path)

            loaded = LSTMFeatureGenerator()
            loaded.load(path)

            self.assertEqual(loaded.beta, generator.beta)
            self.assertEqual(loaded.kl_warmup_epochs, generator.kl_warmup_epochs)
            self.assertEqual(loaded.output_size, generator.output_size)
            self.assertEqual(loaded.feature_columns, generator.feature_columns)

            cols = [f"LATENT_F{i}" for i in range(generator.output_size)]
            pd.testing.assert_frame_equal(
                generator.transform(df)[cols],
                loaded.transform(df)[cols],
                atol=1e-5,
                rtol=1e-5,
            )

    def test_old_bundle_format_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old_bundle.pkl")
            import pickle
            with open(path, "wb") as f:
                pickle.dump({"autoencoder_state": {}, "lookback": 10}, f)

            generator = LSTMFeatureGenerator()
            with self.assertRaisesRegex(ValueError, "Regenerate LSTM artifacts"):
                generator.load(path)

    def test_validation_loss_is_batched(self):
        df = _make_feature_frame(rows=96)
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            feature_columns=["close_norm", "feat_a", "feat_b", "feat_c"],
            device="cpu",
            pretrain_epochs=1,
            pretrain_batch_size=7,
            pretrain_patience=1,
            pretrain_verbose=False,
        )
        train_sequences, validation_sequences = generator._prepare_sequence_splits(df.iloc[:64], df.iloc[64:])

        self.assertIsNotNone(validation_sequences)
        self.assertGreater(len(validation_sequences), generator._validation_batch_size())

        best_loss = generator._pretrain_autoencoder(
            train_sequences=train_sequences,
            validation_sequences=validation_sequences,
            checkpoint_path=None,
        )

        self.assertTrue(np.isfinite(best_loss))

    def test_transform_encodes_in_batches(self):
        df = _make_feature_frame(rows=96)
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            feature_columns=["close_norm", "feat_a", "feat_b", "feat_c"],
            device="cpu",
            pretrain_epochs=2,
            pretrain_batch_size=6,
            pretrain_patience=2,
            pretrain_verbose=False,
        )
        generator.fit(df.iloc[:64], df.iloc[64:])
        transformed = generator.transform(df)

        for i in range(generator.output_size):
            col = f"LATENT_F{i}"
            self.assertIn(col, transformed.columns)
            self.assertTrue(np.isfinite(transformed[col].to_numpy()).all())

    def test_calculate_lstm_features_fits_then_reuses_generator(self):
        df = _make_feature_frame(rows=64)
        feature_columns = ["close_norm", "feat_a", "feat_b", "feat_c"]

        with mock.patch("utils.device.get_device", return_value="cpu"):
            transformed_train, generator = calculate_lstm_features(
                df.iloc[:40],
                lookback=8,
                hidden_size=8,
                num_layers=1,
                output_size=3,
                feature_columns=feature_columns,
            )
            transformed_val, reused_generator = calculate_lstm_features(
                df.iloc[40:],
                generator=generator,
            )

        self.assertIs(reused_generator, generator)
        self.assertTrue(generator.is_pretrained)
        self.assertEqual(generator.feature_columns, feature_columns)
        self.assertIn("LATENT_F0", transformed_train.columns)
        self.assertIn("LATENT_F2", transformed_val.columns)
        self.assertTrue(np.isfinite(transformed_val["LATENT_F1"].to_numpy()).all())

    def test_tune_lstm_hyperparameters_merges_defaults_and_saves_results(self):
        df = _make_feature_frame(rows=48)
        feature_columns = ["close_norm", "feat_a", "feat_b", "feat_c"]

        class _FakeTrial:
            number = 7

            def suggest_categorical(self, name, choices):
                return choices[-1]

            def suggest_int(self, name, low, high):
                return high

            def suggest_float(self, name, low, high, log=False):
                return high

            def report(self, value, step):
                self.reported = (value, step)

            def should_prune(self):
                return False

        class _FakeStudy:
            def __init__(self):
                self.best_params = {
                    "hidden_size": 16,
                    "num_layers": 2,
                    "output_size": 6,
                    "lookback": 12,
                    "pretrain_lr": 0.0008,
                    "beta": 0.002,
                }
                self.best_value = 0.123

            def optimize(self, objective, n_trials, show_progress_bar=True):
                objective(_FakeTrial())

        tuning_config = {
            "n_trials": 1,
            "parameters": {
                "hidden_size": {"choices": [8, 16]},
                "num_layers": {"min": 1, "max": 2},
                "output_size": {"choices": [4, 6]},
                "lookback": {"choices": [8, 12]},
                "pretrain_lr": {"min": 1e-4, "max": 8e-4, "log": True},
                "beta": {"min": 1e-4, "max": 0.002, "log": True},
            },
        }
        base_config = {
            "lookback": 8,
            "hidden_size": 8,
            "num_layers": 1,
            "output_size": 4,
            "pretrain_epochs": 6,
            "pretrain_lr": 0.001,
            "pretrain_batch_size": 8,
            "pretrain_patience": 3,
            "pretrain_min_delta": 0.0001,
            "beta": 0.001,
            "kl_warmup_epochs": 2,
        }

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("optuna.create_study", return_value=_FakeStudy()), \
             mock.patch("optuna.samplers.TPESampler"), \
             mock.patch("optuna.pruners.MedianPruner"), \
             mock.patch("optuna.logging.get_verbosity", return_value=10), \
             mock.patch("optuna.logging.set_verbosity") as set_verbosity_mock, \
             mock.patch("utils.device.get_device", return_value="cpu"), \
             mock.patch("indicators.lstm_features.LSTMFeatureGenerator._prepare_sequence_splits", return_value=(
                 np.zeros((12, 8, len(feature_columns)), dtype=np.float32),
                 np.zeros((4, 8, len(feature_columns)), dtype=np.float32),
             )), \
             mock.patch("indicators.lstm_features.LSTMFeatureGenerator._pretrain_autoencoder", return_value=0.123):
            params = tune_lstm_hyperparameters(
                train_data=df.iloc[:32],
                validation_data=df.iloc[32:],
                tuning_config=tuning_config,
                base_config=base_config,
                feature_columns=feature_columns,
                window_folder=tmpdir,
            )

            with open(os.path.join(tmpdir, "lstm_tuning_results.json"), "r") as fh:
                saved = json.load(fh)

        self.assertEqual(params["hidden_size"], 16)
        self.assertEqual(params["num_layers"], 2)
        self.assertEqual(params["output_size"], 6)
        self.assertEqual(params["lookback"], 12)
        self.assertEqual(params["pretrain_batch_size"], 8)
        self.assertEqual(params["kl_warmup_epochs"], 2)
        self.assertEqual(saved["best_value"], 0.123)
        self.assertEqual(saved["best_params"]["lookback"], 12)
        set_verbosity_mock.assert_any_call(mock.ANY)
        set_verbosity_mock.assert_called_with(10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
