#!/usr/bin/env python3
"""
Benchmark: the from scratch NumPy transformer vs an equivalent
PyTorch transformer, same corpus, same configuration, same batches.

Fairness protocol, stated so the numbers can be judged:
- Identical model configuration (d_model, layers, heads, block size)
  and identical parameter count to within embedding tying choices
  (the PyTorch model does not tie embeddings; counts are printed).
- Identical training batches, pre generated from one seed and fed to
  both models in the same order.
- Same optimizer family (Adam) and learning rate, same step count.
- Default threading for both libraries, on the same machine, in the
  same session. Thread counts are recorded in the output, not tuned.
- What is compared: milliseconds per training step and eval loss
  before and after the shared steps. Initial losses differ because
  the random initializations differ; the step time and the loss
  after the same number of steps on the same data are the honest
  comparison.

Requires PyTorch (not a dependency of the lab itself):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

Usage:
    python scripts/benchmark_pytorch.py [--steps 30]
                                        [--out results/benchmark_pytorch.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn

from neural_net_lab.transformer import CharTransformerLM
from neural_net_lab.losses import CrossEntropyLoss
from neural_net_lab.optimizers import Adam


class TorchCharLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.ln_f(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "results" / "benchmark_pytorch.json"))
    args = ap.parse_args()

    text = (REPO / "data" / "shakespeare_sonnets.txt").read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    train_data = data[: int(0.9 * len(data))]
    V = len(chars)

    # Pre generate the shared batches
    rng = np.random.default_rng(2024)
    batches = []
    for _ in range(args.warmup + args.steps):
        ix = rng.integers(0, len(train_data) - args.block_size - 1,
                          size=args.batch_size)
        x = np.stack([train_data[i:i + args.block_size] for i in ix])
        y = np.stack([train_data[i + 1:i + args.block_size + 1] for i in ix])
        batches.append((x, y))
    eval_rng = np.random.default_rng(1234)
    eval_batches = []
    for _ in range(10):
        ix = eval_rng.integers(0, len(train_data) - args.block_size - 1,
                               size=args.batch_size)
        x = np.stack([train_data[i:i + args.block_size] for i in ix])
        y = np.stack([train_data[i + 1:i + args.block_size + 1] for i in ix])
        eval_batches.append((x, y))

    # ── NumPy model ──────────────────────────────────────────────────
    np_model = CharTransformerLM(vocab_size=V, d_model=args.d_model,
                                 num_heads=args.n_head,
                                 num_layers=args.n_layer,
                                 max_seq_len=args.block_size, seed=args.seed)
    np_opt = Adam(np_model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()

    def np_eval():
        losses = [loss_fn.forward(np_model.forward(x).reshape(-1, V), y.reshape(-1))
                  for x, y in eval_batches]
        return float(np.mean(losses))

    np_before = np_eval()
    np_times = []
    for i, (x, y) in enumerate(batches):
        t0 = time.perf_counter()
        logits = np_model.forward(x)
        loss_fn.forward(logits.reshape(-1, V), y.reshape(-1))
        np_opt.zero_grad()
        np_model.backward(loss_fn.backward().reshape(args.batch_size,
                                                      args.block_size, V))
        np_opt.step()
        if i >= args.warmup:
            np_times.append(time.perf_counter() - t0)
    np_after = np_eval()

    # ── PyTorch model ────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    t_model = TorchCharLM(V, args.d_model, args.n_head, args.n_layer,
                          args.block_size)
    t_opt = torch.optim.Adam(t_model.parameters(), lr=args.lr)
    t_loss = nn.CrossEntropyLoss()

    def t_eval():
        t_model.eval()
        losses = []
        with torch.no_grad():
            for x, y in eval_batches:
                logits = t_model(torch.tensor(x))
                losses.append(t_loss(logits.reshape(-1, V),
                                     torch.tensor(y).reshape(-1)).item())
        t_model.train()
        return float(np.mean(losses))

    t_before = t_eval()
    t_times = []
    for i, (x, y) in enumerate(batches):
        t0 = time.perf_counter()
        logits = t_model(torch.tensor(x))
        loss = t_loss(logits.reshape(-1, V), torch.tensor(y).reshape(-1))
        t_opt.zero_grad()
        loss.backward()
        t_opt.step()
        if i >= args.warmup:
            t_times.append(time.perf_counter() - t0)
    t_after = t_eval()

    result = {
        "config": {"d_model": args.d_model, "n_layer": args.n_layer,
                   "n_head": args.n_head, "block_size": args.block_size,
                   "batch_size": args.batch_size, "lr": args.lr,
                   "steps_timed": args.steps, "warmup_steps": args.warmup},
        "environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
        },
        "numpy_lab": {
            "parameters": np_model.count_parameters(),
            "mean_step_ms": float(np.mean(np_times) * 1000),
            "eval_loss_before": np_before,
            "eval_loss_after": np_after,
        },
        "pytorch": {
            "parameters": int(sum(p.numel() for p in t_model.parameters())),
            "mean_step_ms": float(np.mean(t_times) * 1000),
            "eval_loss_before": t_before,
            "eval_loss_after": t_after,
        },
    }
    result["speed_ratio_torch_faster_by"] = round(
        result["numpy_lab"]["mean_step_ms"] / result["pytorch"]["mean_step_ms"], 2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
