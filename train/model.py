"""One transformer, two ways to train it.

The diffusion model and its GPT twin are the same network: learned token and position
embeddings, pre-LayerNorm blocks (self-attention + a 4x tanh-GELU MLP), a final LayerNorm and
an output layer tied to the token embedding. The only differences are the attention mask
(bidirectional vs causal) and the loss (masked-token ELBO vs next-character cross-entropy).

Both use the same 66-token vocabulary: the 65 characters of tiny-shakespeare plus [MASK].
The GPT never sees [MASK]; its row is kept so both models have exactly the same parameters.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 66          # 65 characters + [MASK]
    ctx: int = 256
    n_layer: int = 4
    n_head: int = 4
    d: int = 128
    dropout: float = 0.1
    causal: bool = False


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc = nn.Linear(c.d, 4 * c.d, bias=False)
        self.fc2 = nn.Linear(4 * c.d, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, D = x.shape
        H = self.c.n_head
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.c.causal,
            dropout_p=self.c.dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))
        return x


class Model(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style scaled init on residual projections
            for w in (b.proj.weight, b.fc2.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.tok.weight.T  # tied output layer

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
