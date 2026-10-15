#!/usr/bin/env python3
"""
Train the from scratch NumPy transformer as a character level language
model on a public domain corpus. The default corpus is Shakespeare's
sonnets (public domain, Project Gutenberg 1041), which reproduces the
original run. A second corpus, Alice's Adventures in Wonderland by
Lewis Carroll (public domain, Project Gutenberg 11), is the second
benchmark task and is run with the same configuration:

    python scripts/train_char_lm.py --data data/alice_in_wonderland.txt \
        --corpus "Alice's Adventures in Wonderland by Lewis Carroll" \
        --source "Project Gutenberg ebook 11, public domain" \
        --prompt "Alice was beginning" \
        --out results/alice_char_lm_run.json

Every number printed here is also written to the JSON artifact named
by --out, and the README quotes that file, not this terminal.

Usage:
    python scripts/train_char_lm.py [--steps 250] [--seed 42]
                                    [--d-model 96] [--n-layer 3]
                                    [--n-head 4] [--block-size 96]
                                    [--batch-size 16] [--lr 0.003]
                                    [--data data/shakespeare_sonnets.txt]
                                    [--prompt "Shall I compare thee"]
                                    [--out results/char_lm_run.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_net_lab.transformer import CharTransformerLM
from neural_net_lab.losses import CrossEntropyLoss
from neural_net_lab.optimizers import Adam

REPO = Path(__file__).resolve().parents[1]


def get_batch(data, batch_size, block_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--data", default=str(REPO / "data" / "shakespeare_sonnets.txt"))
    ap.add_argument("--corpus", default="Shakespeare's sonnets by William Shakespeare")
    ap.add_argument("--source", default="Project Gutenberg ebook 1041, public domain")
    ap.add_argument("--prompt", default="Shall I compare thee")
    ap.add_argument("--out", default=str(REPO / "results" / "char_lm_run.json"))
    args = ap.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    rng = np.random.default_rng(args.seed)
    model = CharTransformerLM(vocab_size=len(chars), d_model=args.d_model,
                              num_heads=args.n_head, num_layers=args.n_layer,
                              max_seq_len=args.block_size, seed=args.seed)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()
    V = len(chars)

    def eval_loss(split_data, batches=10):
        losses = []
        erng = np.random.default_rng(1234)
        for _ in range(batches):
            x, y = get_batch(split_data, args.batch_size, args.block_size, erng)
            logits = model.forward(x)
            losses.append(loss_fn.forward(logits.reshape(-1, V), y.reshape(-1)))
        return float(np.mean(losses))

    print(f"corpus: {args.corpus}  source: {args.source}")
    print(f"corpus chars: {len(data)}  vocab: {V}  params: {model.count_parameters()}")
    print(f"config: d_model={args.d_model} layers={args.n_layer} heads={args.n_head} "
          f"block={args.block_size} batch={args.batch_size} lr={args.lr}")
    initial_train = eval_loss(train_data)
    initial_val = eval_loss(val_data)
    print(f"step 0  train {initial_train:.4f}  val {initial_val:.4f}")

    history = [{"step": 0, "train_loss": initial_train, "val_loss": initial_val}]
    t0 = time.perf_counter()
    step_times = []
    for step in range(1, args.steps + 1):
        ts = time.perf_counter()
        x, y = get_batch(train_data, args.batch_size, args.block_size, rng)
        logits = model.forward(x)
        loss_fn.forward(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        model.backward(loss_fn.backward().reshape(args.batch_size,
                                                   args.block_size, V))
        opt.step()
        step_times.append(time.perf_counter() - ts)
        if step % args.eval_every == 0 or step == args.steps:
            tl = eval_loss(train_data)
            vl = eval_loss(val_data)
            history.append({"step": step, "train_loss": tl, "val_loss": vl})
            print(f"step {step}  train {tl:.4f}  val {vl:.4f}")
    wall = time.perf_counter() - t0

    # Sample with a fixed seed so the artifact is reproducible
    prompt = args.prompt
    start = np.array([stoi[c] for c in prompt])
    sample_ids = model.generate(start, 300, temperature=0.8,
                                rng=np.random.default_rng(7))
    sample = "".join(itos[int(i)] for i in sample_ids)

    result = {
        "config": {
            "d_model": args.d_model, "n_layer": args.n_layer,
            "n_head": args.n_head, "block_size": args.block_size,
            "batch_size": args.batch_size, "lr": args.lr,
            "steps": args.steps, "seed": args.seed,
        },
        "corpus": args.corpus,
        "source": args.source,
        "data_file": Path(args.data).name,
        "corpus_chars": int(len(data)),
        "vocab_size": V,
        "parameters": model.count_parameters(),
        "initial_train_loss": initial_train,
        "initial_val_loss": initial_val,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "history": history,
        "wall_seconds": wall,
        "mean_step_ms": float(np.mean(step_times) * 1000),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sample_prompt": prompt,
        "sample_text": sample,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwall time {wall:.1f}s, mean step {result['mean_step_ms']:.1f} ms")
    print(f"wrote {out_path}")
    print("\nsample:\n" + sample)


if __name__ == "__main__":
    main()
