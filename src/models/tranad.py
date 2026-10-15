"""
TranAD (Tuli, Casale & Jennings, VLDB 2022).

The network classes below are copied from the official implementation
(github.com/imperial-qore/TranAD, src/dlutils.py and src/models.py),
Copyright (c) 2022, Shreshth Tuli, BSD 3-Clause License (see third_party/TranAD_LICENSE.txt).
Only change: torch.nn.TransformerEncoder/Decoder wrappers (num_layers=1, no norm) are replaced by a
direct call to the single layer, because recent PyTorch passes `is_causal` to custom layers.

Training and scoring follow the official code, which differs from the paper (documented in DOCUMENTATION.md):
  loss = (1/n)*MSE(phase-1) + (1-1/n)*MSE(phase-2), n = epoch; no MAML step; target = last window element;
  score = phase-2 squared error per feature, averaged over features (as fed to POT in the official code).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from .base import Detector


# ----------------------------------------------------------------------------- official classes (BSD-3)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(nn.Module):
    def __init__(self, feats, n_window=10, dim_feedforward=16, dropout=0.1):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.n_feats = feats
        self.n_window = n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.transformer_encoder = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


# ----------------------------------------------------------------------------- training / scoring
class TranADDetector(Detector):
    name = "tranad"
    representation = "resampled"
    scaler = "minmax"                # paper eq. 1, fitted on training data

    def __init__(self, cfg: dict, seed: int, device: str = "cpu"):
        self.p, self.seed, self.device = cfg, seed, device
        self.window = cfg["window"]

    def _batches(self, x, bs, shuffle=False, gen=None):
        idx = torch.randperm(len(x), generator=gen) if shuffle else torch.arange(len(x))
        for i in range(0, len(x), bs):
            w = x[idx[i:i + bs]].to(self.device).permute(1, 0, 2)    # (window, batch, feats)
            yield w, w[-1:, :, :]

    def fit(self, x_train, x_val, log=None):
        torch.manual_seed(self.seed)
        f = x_train.shape[-1]
        self.net = TranAD(f, self.window, self.p["dim_feedforward"], self.p["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        sched = torch.optim.lr_scheduler.StepLR(opt, self.p["scheduler_step"], self.p["scheduler_gamma"])
        mse = nn.MSELoss(reduction="none")
        xt = torch.as_tensor(x_train, dtype=torch.float32)
        xv = torch.as_tensor(x_val, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, hist = np.inf, None, 0, {"train": [], "val": []}
        for epoch in range(self.p["epochs"]):
            n = epoch + 1
            self.net.train()
            losses = []
            for w, elem in self._batches(xt, self.p["batch_size"], True, g):
                z = self.net(w, elem)
                loss = torch.mean((1 / n) * mse(z[0], elem) + (1 - 1 / n) * mse(z[1], elem))
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            sched.step()
            val = float(np.mean(self._dims(xv))) if len(xv) else float("nan")
            hist["train"].append(float(np.mean(losses))); hist["val"].append(val)
            if val < best - 1e-9:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
            if log and epoch % 5 == 0:
                log.info(f"  tranad epoch {n}: train {hist['train'][-1]:.5f} val {val:.5f}")
            if bad >= self.p["patience"]:
                break
        self.net.load_state_dict(best_state)
        hist["best_epoch"] = int(np.argmin(hist["val"]) + 1)
        return hist

    @torch.no_grad()
    def _dims(self, x, bs=2048):
        self.net.eval()
        x = torch.as_tensor(x, dtype=torch.float32)
        out = []
        for w, elem in self._batches(x, bs):
            _, x2 = self.net(w, elem)
            out.append(((x2 - elem) ** 2)[0].cpu())
        return torch.cat(out).numpy()

    def explain_dims(self, x):
        return self._dims(x)                         # per-feature score (TranAD's diagnosis signal)

    def score(self, x):
        return self._dims(x).mean(axis=1)
