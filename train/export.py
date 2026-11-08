"""Quantise both checkpoints to int8 and write what the page and the tests need.

    python train/export.py

For each model (weights/<kind>.pt):
- weights/<kind>.q8   int8 matrices, one scale per row (max |w| / 127), in the order src/model.js reads
- weights/<kind>.f32  float32 per-row scales and LayerNorm parameters, same order
- tests/probes-<kind>.bin + .json   20 fixed inputs and PyTorch's logits for them, computed with the
  dequantised int8 weights (the same numbers the page uses), for tests/probes.mjs
- weights/stats.json  parameter counts, fp32 and int8 validation loss, bytes
"""
import json, pathlib, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import Config, Model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
text = (ROOT / "data" / "input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
val = data[int(0.9 * len(data)):]
MASK, CTX = 65, 256


def order(sd):
    """(name, tensor, kind) in src/model.js order; kind 'm' = int8 matrix, 'v' = float vector."""
    yield "tok", sd["tok.weight"], "m"
    yield "pos", sd["pos.weight"], "m"
    for l in range(4):
        p = f"blocks.{l}."
        yield p + "ln1w", sd[p + "ln1.weight"], "v"
        yield p + "ln1b", sd[p + "ln1.bias"], "v"
        yield p + "qkv", sd[p + "qkv.weight"], "m"
        yield p + "proj", sd[p + "proj.weight"], "m"
        yield p + "ln2w", sd[p + "ln2.weight"], "v"
        yield p + "ln2b", sd[p + "ln2.bias"], "v"
        yield p + "fc", sd[p + "fc.weight"], "m"
        yield p + "fc2", sd[p + "fc2.weight"], "m"
    yield "lnfw", sd["ln_f.weight"], "v"
    yield "lnfb", sd["ln_f.bias"], "v"


def quantise(sd):
    q8, f32, deq = [], [], {}
    for name, t, kind in order(sd):
        t = t.float().cpu()
        if kind == "m":
            s = t.abs().amax(1).clamp_min(1e-12) / 127.0
            q = torch.round(t / s[:, None]).clamp(-127, 127).to(torch.int8)
            q8.append(q.numpy().ravel())
            # store the scale as float32 exactly as the page reads it
            s32 = s.numpy().astype(np.float32)
            f32.append(s32)
            deq[name] = torch.from_numpy(q.numpy().astype(np.float32) * s32[:, None])
        else:
            f32.append(t.numpy().astype(np.float32))
            deq[name] = t
    return np.concatenate(q8), np.concatenate(f32), deq


def dequantised_state(sd, deq):
    out = dict(sd)
    out["tok.weight"], out["pos.weight"] = deq["tok"], deq["pos"]
    for l in range(4):
        p = f"blocks.{l}."
        for k in ("qkv", "proj", "fc", "fc2"):
            out[p + k + ".weight"] = deq[p + k]
    return out


@torch.no_grad()
def val_loss(model, kind, n=400, seed=0):
    """GPT: next-char cross-entropy. Diffusion: the ELBO bound (masked CE / t), nats per char."""
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for b in range(n // 50):
        ix = torch.randint(len(val) - CTX - 1, (50,), generator=g)
        x = torch.stack([val[i:i + CTX + 1] for i in ix])
        if kind == "gpt":
            lg = model(x[:, :-1])
            tot += F.cross_entropy(lg.reshape(-1, 66), x[:, 1:].reshape(-1)).item()
        else:
            x = x[:, :-1]
            t = ((torch.rand(1, generator=g) + torch.arange(50) / 50) % 1).clamp_min(1e-3)
            m = torch.rand(50, CTX, generator=g) < t[:, None]
            lg = model(torch.where(m, torch.full_like(x, MASK), x))
            ce = F.cross_entropy(lg.reshape(-1, 66), x.reshape(-1), reduction="none").view(50, CTX)
            tot += ((ce * m).sum(1) / t).mean().item() / CTX
    return tot / (n // 50)


def probes(model, kind):
    """20 fixed inputs: validation windows, each with position 0 set to a newline (as the page
    starts), and for the diffusion model a different share of positions masked (5%..100%)."""
    g = torch.Generator().manual_seed(2026)
    xs = []
    for k in range(20):
        i = torch.randint(len(val) - CTX, (1,), generator=g).item()
        x = val[i:i + CTX].clone()
        x[0] = 0
        if kind == "diffusion":
            frac = 0.05 + 0.95 * k / 19
            m = torch.rand(CTX, generator=g) < frac
            m[0] = False
            x[m] = MASK
        xs.append(x)
    xs = torch.stack(xs)
    with torch.no_grad():
        lg = model(xs)
    return xs.numpy().astype(np.uint8), lg.numpy().astype(np.float32)


stats = {}
for kind in ("diffusion", "gpt"):
    ck = torch.load(ROOT / "weights" / f"{kind}.pt", map_location="cpu")
    assert ck["chars"] == chars
    sd = ck["model"]
    model = Model(Config(causal=kind == "gpt", dropout=0.0))
    model.load_state_dict(sd)
    model.eval()
    fp = val_loss(model, kind)
    q8, f32, deq = quantise(sd)
    model.load_state_dict(dequantised_state(sd, deq))
    q = val_loss(model, kind)
    (ROOT / "weights" / f"{kind}.q8").write_bytes(q8.tobytes())
    (ROOT / "weights" / f"{kind}.f32").write_bytes(f32.tobytes())
    xs, lg = probes(model, kind)
    (ROOT / "tests" / f"probes-{kind}.bin").write_bytes(xs.tobytes() + lg.tobytes())
    stats[kind] = {"params": model.n_params(), "best_step": ck["step"],
                   "val_fp32": round(fp, 4), "val_int8": round(q, 4),
                   "q8_bytes": int(q8.size), "f32_bytes": int(f32.size * 4)}
    print(kind, stats[kind])

(ROOT / "weights" / "stats.json").write_text(json.dumps(stats, indent=1))
