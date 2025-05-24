import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_nq_data(period="60d", interval="5m"):
    """Download NQ futures data from Yahoo Finance."""
    print(f"Downloading NQ futures data for period {period} with interval {interval}")
    ticker = "NQ=F"
    data = yf.download(ticker, period=period, interval=interval)
    
    if data.empty:
        raise ValueError("Failed to download NQ futures data")
        
    # Convert to numpy array with [open, high, low, close, volume]
    np_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    # Save both formats
    np.save('nq_5min.npy', np_data)
    data.to_csv('nq_5min.csv')
    
    print(f"Downloaded {len(data)} bars of NQ futures data")
    return np_data

# ----------------------------------------
# Dataset for NQ 5-minute candles
# ----------------------------------------
class CandleDataset(Dataset):
    def __init__(self, data, seq_len=100):
        # data: numpy array of shape (N, features)
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        return torch.tensor(seq, dtype=torch.float)

# ----------------------------------------
# GRU-LSTM Encoder / Decoder / Generator / Supervisor / Discriminator
# ----------------------------------------
class SequenceModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.merge = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out_gru, _ = self.gru(x)
        out_lstm, _ = self.lstm(x)
        # Concatenate last states
        h_gru = out_gru[:, -1, :]
        h_lstm = out_lstm[:, -1, :]
        h = torch.cat([h_gru, h_lstm], dim=1)
        return self.merge(h)

class Encoder(nn.Module):
    def __init__(self, feat_dim, latent_dim):
        super().__init__()
        self.seq_module = SequenceModule(feat_dim, latent_dim)
    def forward(self, x):
        return self.seq_module(x)  # (batch, latent_dim)

class Decoder(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super().__init__()
        self.latent_to_seq = nn.Linear(latent_dim, seq_len * feat_dim)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
    def forward(self, h):
        # h: (batch, latent_dim)
        out = self.latent_to_seq(h)
        out = out.view(-1, self.seq_len, self.feat_dim)
        return out  # reconstructed sequence

class Generator(nn.Module):
    def __init__(self, noise_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, z):
        return self.net(z)  # generate latent sequence

class Supervisor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # maps h_{t-2} to h_t
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, h_lagged):
        return self.net(h_lagged)

class Discriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim, layers=2):
        super().__init__()
        self.seq_module = SequenceModule(feat_dim, hidden_dim, layers)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x):
        h = self.seq_module(x)
        return self.fc(h)  # probability

# ----------------------------------------
# ChronoGAN Model with Losses
# ----------------------------------------
class ChronoGAN(nn.Module):
    def __init__(self, feat_dim, noise_dim, latent_dim, seq_len):
        super().__init__()
        self.encoder = Encoder(feat_dim, latent_dim)
        self.decoder = Decoder(latent_dim, feat_dim, seq_len)
        self.generator = Generator(noise_dim, latent_dim)
        self.supervisor = Supervisor(latent_dim)
        self.discriminator = Discriminator(feat_dim, latent_dim)
        self.seq_len = seq_len
        self.latent_dim = latent_dim

    def forward_autoencoder(self, x):
        h = self.encoder(x)
        x_rec = self.decoder(h)
        return x_rec, h

    def forward_generator(self, batch_size):
        z = torch.randn(batch_size, noise_dim, device=next(self.parameters()).device)
        h_gen = self.generator(z)
        # supervise two-step: apply supervisor twice
        h_sup = self.supervisor(h_gen)
        x_fake = self.decoder(h_sup)
        return x_fake, h_gen, h_sup

    # Loss functions follow the paper's equations:
    def reconstruction_loss(self, x, x_rec):
        return nn.MSELoss()(x_rec, x)

    def adversarial_loss(self, y_pred, y_true):
        return nn.BCELoss()(y_pred, y_true)

    def moment_loss(self, real, fake):
        # mean and variance
        mean_real = real.mean(dim=(0,1))
        mean_fake = fake.mean(dim=(0,1))
        var_real = real.var(dim=(0,1))
        var_fake = fake.var(dim=(0,1))
        return torch.mean(torch.abs(mean_real - mean_fake)) + torch.mean(torch.abs(var_real - var_fake))

    def stepwise_loss(self, h_gen):
        # Ensure h_gen has the right shape for the supervisor
        # h_gen shape: (batch_size, latent_dim)
        if len(h_gen.shape) == 3:
            h_gen = h_gen.view(-1, self.latent_dim)
        return torch.mean((h_gen - self.supervisor(h_gen))**2)

    def time_series_loss(self, real, fake):
        # compute slope, skewness, weighted avg, median for each batch
        def slope(x):
            # x shape: (batch_size, seq_len, features)
            t = torch.arange(x.size(1), device=x.device).float()
            # Sum across features dimension
            x_sum = x.sum(dim=-1)  # (batch_size, seq_len)
            # Compute slope for each sequence in batch
            Xt = (t * x_sum).sum(dim=1)  # (batch_size,)
            X = x_sum.sum(dim=1)  # (batch_size,)
            t_sum = t.sum()
            t_sq_sum = (t**2).sum()
            # Compute slope using least squares formula
            num = x.size(1)*Xt - t_sum*X
            den = x.size(1)*t_sq_sum - (t_sum**2)
            return num/den

        S_r = slope(real)
        S_f = slope(fake)
        loss_slope = nn.MSELoss()(S_r, S_f)

        # skewness
        def skew(x):
            # x shape: (batch_size, seq_len, features)
            x_sum = x.sum(dim=-1)  # (batch_size, seq_len)
            mu = x_sum.mean(dim=1, keepdim=True)  # (batch_size, 1)
            sigma = x_sum.std(dim=1, keepdim=True)  # (batch_size, 1)
            return torch.mean(((x_sum - mu)/sigma)**3, dim=1)  # (batch_size,)

        Sk_r = skew(real)
        Sk_f = skew(fake)
        loss_skew = nn.MSELoss()(Sk_r, Sk_f)

        # weighted avg (weights = time index)
        t = torch.arange(real.size(1), device=real.device).float()
        x_sum_r = real.sum(dim=-1)  # (batch_size, seq_len)
        x_sum_f = fake.sum(dim=-1)  # (batch_size, seq_len)
        wavg_r = (t * x_sum_r).sum(dim=1)/t.sum()  # (batch_size,)
        wavg_f = (t * x_sum_f).sum(dim=1)/t.sum()  # (batch_size,)
        loss_wavg = nn.MSELoss()(wavg_r, wavg_f)

        # median
        med_r = real.sum(dim=-1).median(dim=1).values  # (batch_size,)
        med_f = fake.sum(dim=-1).median(dim=1).values  # (batch_size,)
        loss_med = nn.MSELoss()(med_r, med_f)

        return loss_slope + loss_skew + loss_wavg + loss_med

# ----------------------------------------
# Training Loop Skeleton
# ----------------------------------------
def train_chrono(
    model, dataloader, epochs=1000,
    lr=1e-3, device='cuda'
):
    optim_AE = optim.Adam(
        list(model.encoder.parameters())+
        list(model.decoder.parameters()), lr=lr)
    optim_G = optim.Adam(
        list(model.generator.parameters())+
        list(model.supervisor.parameters()), lr=lr)
    optim_D = optim.Adam(model.discriminator.parameters(), lr=lr)

    bce = nn.BCELoss()
    print("Starting...")
    for epoch in range(epochs):
        for x in dataloader:
            x = x.to(device)
            batch_size = x.size(0)
            # ------------------ Autoencoder training ------------------
            x_rec, h = model.forward_autoencoder(x)
            # reconstruction + adversarial AE
            y_real = torch.ones(batch_size,1, device=device)
            y_fake_ae = model.discriminator(x_rec)
            lae = model.reconstruction_loss(x, x_rec) + bce(y_fake_ae, y_real)
            optim_AE.zero_grad(); lae.backward(); optim_AE.step()

            # ------------------ Generator + Supervisor training ------------------
            # generate fake
            x_fake, h_gen, h_sup = model.forward_generator(batch_size)
            # adversarial loss on fake
            y_fake = torch.ones(batch_size,1, device=device)
            y_fake_d = model.discriminator(x_fake)
            lu = bce(y_fake_d, y_fake)
            lm = model.moment_loss(x, x_fake)
            ls = model.stepwise_loss(h_gen)
            lts = model.time_series_loss(x, x_fake)
            lg = lu + lm + ls + lts
            optim_G.zero_grad(); lg.backward(); optim_G.step()

            # ------------------ Discriminator training ------------------
            y_real_d = model.discriminator(x)
            y_fake_d = model.discriminator(x_fake.detach())
            ld = bce(y_real_d, torch.ones_like(y_real_d)) + bce(y_fake_d, torch.zeros_like(y_fake_d))
            optim_D.zero_grad(); ld.backward(); optim_D.step()

        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/{epochs}] AE:{lae.item():.4f} G:{lg.item():.4f} D:{ld.item():.4f}")

# ----------------------------------------
# Usage Example
# ----------------------------------------
if __name__ == "__main__":
    # Download fresh data from Yahoo Finance
    data = download_nq_data(period="60d", interval="5m")
    seq_len = 50
    dataset = CandleDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = 'mps'
    feat_dim = data.shape[1]
    noise_dim = 32
    latent_dim = 64

    model = ChronoGAN(feat_dim, noise_dim, latent_dim, seq_len).to(device)
    train_chrono(model, loader, epochs=2000, lr=1e-4, device=device)

    # Generate synthetic
    x_syn, _, _ = model.forward_generator(10)
    print("Generated synthetic candles:", x_syn.shape)
