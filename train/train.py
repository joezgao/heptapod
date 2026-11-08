"""Train the diffusion model or its GPT twin on tiny-shakespeare.

    python train/train.py diffusion
    python train/train.py gpt

Same data, split, batch, optimiser and schedule for both. Keeps the checkpoint with the best
validation loss in weights/<kind>.pt and a log in train/logs/<kind>.tsv.

Diffusion loss (absorbing-state masked diffusion, as in MDLM / LLaDA): draw t ~ U(0,1) per
sequence, replace each character with [MASK] with probability t, and score only the masked
characters, weighted by 1/t. Averaged per character this is an upper bound (ELBO) on the
negative log-likelihood in nats per character, so it can be compared with the GPT's loss.
"""
import json, math, pathlib, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import Config, Model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
KIND = sys.argv[1]
assert KIND in ("diffusion", "gpt")

STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
BATCH, CTX = 64, 256
LR, LR_MIN, WARMUP, WD = 2e-3, 1e-4, 300, 0.1
EVAL_EVERY, EVAL_BATCHES = 250, 40
SEED = 1337

text = (ROOT / "data" / "input.txt").read_text()
chars = sorted(set(text))
assert len(chars) == 65
MASK = 65
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(SEED)
model = Model(Config(causal=(KIND == "gpt"))).to(dev)
print(KIND, model.n_params(), "parameters on", dev)

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.99))


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * p))


def batch(split, g=None):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CTX - 1, (BATCH,), generator=g)
    x = torch.stack([d[i:i + CTX + 1] for i in ix]).to(dev)
    return x


def loss_on(x, g=None):
    if KIND == "gpt":
        logits = model(x[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
    x = x[:, :-1]
    B, T = x.shape
    # stratified t over the batch keeps the loss estimate low-variance
    u = torch.rand(1, generator=g).item()
    t = ((u + torch.arange(B) / B) % 1.0).clamp_min(1e-3).to(dev)
    masked = torch.rand(B, T, generator=g).to(dev) < t[:, None]
    xin = torch.where(masked, torch.full_like(x, MASK), x)
    logits = model(xin)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1),
                         reduction="none").view(B, T)
    return ((ce * masked).sum(1) / t).mean() / T


@torch.no_grad()
def evaluate():
    model.eval()
    g = torch.Generator().manual_seed(0)  # same val windows and masks every time
    out = {}
    for split in ("train", "val"):
        out[split] = sum(loss_on(batch(split, g), g).item() for _ in range(EVAL_BATCHES)) / EVAL_BATCHES
    model.train()
    return out


(ROOT / "train" / "logs").mkdir(exist_ok=True)
log = open(ROOT / "train" / "logs" / f"{KIND}.tsv", "w")
log.write("step\ttrain\tval\tlr\tseconds\n")
best, t0 = float("inf"), time.time()
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0 or step == STEPS:
        ev = evaluate()
        el = time.time() - t0
        log.write(f"{step}\t{ev['train']:.4f}\t{ev['val']:.4f}\t{lr_at(step):.2e}\t{el:.0f}\n")
        log.flush()
        tag = ""
        if ev["val"] < best:
            best = ev["val"]
            torch.save({"model": model.state_dict(), "step": step, "val": best,
                        "chars": chars, "kind": KIND}, ROOT / "weights" / f"{KIND}.pt")
            tag = " *"
        print(f"{step:6d} train {ev['train']:.4f} val {ev['val']:.4f} {el:5.0f}s{tag}", flush=True)
    if step == STEPS:
        break
    for gp in opt.param_groups:
        gp["lr"] = lr_at(step)
    loss = loss_on(batch("train"))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

summary = {"kind": KIND, "params": model.n_params(), "steps": STEPS, "best_val": best,
           "seconds": round(time.time() - t0)}
(ROOT / "train" / "logs" / f"{KIND}.json").write_text(json.dumps(summary, indent=1))
print(summary)
