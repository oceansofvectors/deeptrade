"""
AlgoTrader3: World Models for NQ Futures Trading

Based on Ha & Schmidhuber's World Models paper (2018).
https://arxiv.org/abs/1803.10122

Components:
- VAE (V): Compresses observations into latent space
- MDN-RNN (M): Predicts next latent state distribution
- Controller (C): Simple linear policy trained with CMA-ES
"""

__version__ = "0.1.0"
