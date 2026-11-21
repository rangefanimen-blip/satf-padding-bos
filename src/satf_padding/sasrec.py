"""Compact PyTorch SASRec baseline: causal attention and sampled BCE.

Architecture follows Kang & McAuley (ICDM 2018); this is our own implementation,
not an assertion of exact reproduction of the original paper's scores.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 1, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.Dropout(dropout))
    def forward(self, x, causal, nonzero):
        q = self.ln1(x)
        a, _ = self.attn(q, x, x, attn_mask=causal, need_weights=False)
        x = (q + a) * nonzero
        x = (x + self.ff(self.ln2(x))) * nonzero
        return x

class Network(nn.Module):
    def __init__(self, n_items, k, dim, blocks, dropout):
        super().__init__()
        self.items = nn.Embedding(n_items+1, dim, padding_idx=0)
        self.positions = nn.Embedding(k, dim)
        self.blocks = nn.ModuleList([Block(dim, dropout) for _ in range(blocks)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.dim, self.k = dim, k
    def forward(self, ids):
        valid = (ids != 0).unsqueeze(-1)
        x = self.items(ids) * self.dim ** .5
        x = self.dropout(x + self.positions(torch.arange(self.k, device=ids.device))) * valid
        causal = torch.ones((self.k, self.k), device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, causal, valid)
        return self.norm(x) * valid

class SASRec:
    def __init__(self, n_items, k=50, dim=32, blocks=1, dropout=.2, lr=.001, epochs=5, batch_size=128, seed=0, device="cpu"):
        self.n_items, self.k, self.epochs, self.batch_size = n_items, k, epochs, batch_size
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.rng = np.random.default_rng(seed)
        self.net = Network(n_items, k, dim, blocks, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(.9, .98))
        self.losses = []

    def fit(self, histories):
        examples = [np.asarray(h, int) for h in histories if 2 <= len(h) < self.n_items]
        if not examples:
            raise ValueError("No train sequences with positive and negative items")
        self.net.train()
        for _ in range(self.epochs):
            total, count = 0., 0
            for start in range(0, len(examples), self.batch_size):
                # Permute once per epoch, retaining all examples exactly once.
                if start == 0:
                    order = self.rng.permutation(len(examples))
                selected = [examples[i] for i in order[start:start+self.batch_size]]
                xs = np.zeros((len(selected), self.k), dtype=np.int64)
                ys = np.zeros_like(xs); ns = np.zeros_like(xs)
                for row, h in enumerate(selected):
                    x, y = h[:-1][-self.k:] + 1, h[1:][-self.k:] + 1
                    xs[row, -len(x):], ys[row, -len(y):] = x, y
                    neg = self.rng.integers(0, self.n_items, len(y))
                    bad = np.isin(neg, h)
                    while bad.any():
                        neg[bad] = self.rng.integers(0, self.n_items, bad.sum())
                        bad = np.isin(neg, h)
                    ns[row, -len(y):] = neg + 1
                x, y, n = (torch.as_tensor(a, device=self.device) for a in (xs, ys, ns))
                features = self.net(x)
                positive = (features * self.net.items(y)).sum(-1)
                negative = (features * self.net.items(n)).sum(-1)
                mask = y != 0
                loss = (F.softplus(-positive[mask]) + F.softplus(negative[mask])).mean()
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total += loss.item() * mask.sum().item(); count += mask.sum().item()
            self.losses.append(total / count)
        self.net.eval()
        return self

    def score_batch(self, histories):
        x = np.zeros((len(histories), self.k), dtype=np.int64)
        for row, h in enumerate(histories):
            h = np.asarray(h)[-self.k:] + 1
            x[row, -len(h):] = h
        with torch.no_grad():
            z = self.net(torch.as_tensor(x, device=self.device))[:, -1]
            scores = z @ self.net.items.weight[1:].T
            return scores.cpu().numpy()
