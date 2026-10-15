"""LSTM autoencoder of Davari et al. (SAC 2024): encoder LSTMs 256-200-140-100-60, bottleneck 20,
dropout 0.2, batch 50, 100 epochs, on sequences of cycle feature vectors (44 per cycle).
The paper does not give the sequence length or decoder layout; the decoder mirrors the encoder
(repeat-vector LSTM-AE). Score = RMSE of the last cycle's reconstruction (the decision cycle)."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .base import Detector


class LSTMAENet(nn.Module):
    def __init__(self, n_features, hidden, bottleneck, dropout):
        super().__init__()
        enc = [n_features] + list(hidden) + [bottleneck]
        dec = [bottleneck] + list(hidden)[::-1]
        self.enc = nn.ModuleList(nn.LSTM(a, b, batch_first=True) for a, b in zip(enc[:-1], enc[1:]))
        self.dec = nn.ModuleList(nn.LSTM(a, b, batch_first=True) for a, b in zip(dec[:-1], dec[1:]))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dec[-1], n_features)

    def forward(self, x):
        h = x
        for i, lstm in enumerate(self.enc):
            h, _ = lstm(h)
            if i < len(self.enc) - 1:
                h = self.drop(h)
        z = h[:, -1:, :].repeat(1, x.size(1), 1)        # bottleneck state repeated over time
        h = z
        for lstm in self.dec:
            h, _ = lstm(h)
            h = self.drop(h)
        return self.out(h)


class LSTMAEDetector(Detector):
    name = "lstm_ae"
    representation = "cycles"
    scaler = "zscore"

    def __init__(self, cfg: dict, seed: int, device: str = "cpu"):
        self.p, self.seed, self.device = cfg, seed, device
        self.window = cfg["seq_len"]

    def fit(self, x_train, x_val, log=None):
        torch.manual_seed(self.seed)
        self.net = LSTMAENet(x_train.shape[-1], self.p["hidden"], self.p["bottleneck"], self.p["dropout"]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"])
        xt = torch.as_tensor(x_train, dtype=torch.float32)
        xv = torch.as_tensor(x_val, dtype=torch.float32)
        best, best_state, bad, hist = np.inf, None, 0, {"train": [], "val": []}
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.p["epochs"]):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g)
            tot = 0.0
            for i in range(0, len(xt), self.p["batch_size"]):
                xb = xt[perm[i:i + self.p["batch_size"]]].to(self.device)
                loss = torch.mean((self.net(xb) - xb) ** 2)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            val = self._mse(xv)
            hist["train"].append(tot / len(xt)); hist["val"].append(val)
            if val < best - 1e-7:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
            if log and (epoch % 10 == 0 or bad == 0 and epoch == self.p["epochs"] - 1):
                log.info(f"  lstm_ae epoch {epoch + 1}: train {hist['train'][-1]:.4f} val {val:.4f}")
            if bad >= self.p["patience"]:
                break
        self.net.load_state_dict(best_state)
        hist["best_epoch"] = int(np.argmin(hist["val"]) + 1)
        return hist

    @torch.no_grad()
    def _recon(self, x, bs=1024):
        self.net.eval()
        return torch.cat([self.net(x[i:i + bs].to(self.device)).cpu() for i in range(0, len(x), bs)])

    def _mse(self, xv):
        return float(torch.mean((self._recon(xv) - xv) ** 2)) if len(xv) else float("nan")

    def explain_dims(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32)
        return ((self._recon(xt) - xt)[:, -1, :] ** 2).numpy()     # per-feature squared error, last cycle

    def score(self, x):
        return np.sqrt(self.explain_dims(x).mean(axis=1))
