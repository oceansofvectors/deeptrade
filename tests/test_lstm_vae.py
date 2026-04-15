#!/usr/bin/env python
"""Tests for the LSTM VAE feature generator."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indicators.lstm_features import LSTMFeatureGenerator, LSTMVAE  # noqa: E402


def _make_ohlcv_frame(rows: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 14:30:00", periods=rows, freq="5min", tz="UTC")
    close = np.linspace(100.0, 103.0, rows) + np.sin(np.linspace(0, 4, rows))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.25,
            "low": close - 0.3,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, rows),
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


class TestLSTMFeatureGeneratorVAE(unittest.TestCase):
    def test_prepare_input_handles_zero_and_nonfinite_bars(self):
        df = _make_ohlcv_frame()
        df.loc[df.index[5], "close"] = 0.0
        df.loc[df.index[6], "volume"] = 0.0
        df.loc[df.index[7], "open"] = np.nan

        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
            pretrain_verbose=False,
        )
        features = generator._prepare_input(df)

        self.assertTrue(np.isfinite(features).all())

    def test_transform_is_deterministic(self):
        df = _make_ohlcv_frame()
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
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
        cols = [f"LSTM_F{i}" for i in range(generator.output_size)]
        pd.testing.assert_frame_equal(transformed_a[cols], transformed_b[cols])

    def test_save_load_round_trip(self):
        df = _make_ohlcv_frame()
        generator = LSTMFeatureGenerator(
            lookback=10,
            hidden_size=8,
            num_layers=1,
            output_size=4,
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

            cols = [f"LSTM_F{i}" for i in range(generator.output_size)]
            pd.testing.assert_frame_equal(
                generator.transform(df)[cols],
                loaded.transform(df)[cols],
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
