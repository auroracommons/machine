#!/usr/bin/env python3
"""
Sampling quality, measured (added October 17, 2026).

Why this exists: the lab's README has always described the generated
sample honestly as poor, but that judgment rested on reading one sample.
This script puts numbers on it. It trains one small model with a fixed
seed, then samples from the same prompt at three temperatures and
measures each sample the same way:

- distinct-1 and distinct-2: the share of unique characters and
  character pairs among all characters and pairs in the sample. Real
  text from the corpus scores high; a model stuck repeating itself
  scores low.
- most common character share: how much of the sample is a single
  character, the simplest repetition signal.

The corpus's own numbers on a same-length window are printed beside the
samples, so the comparison has a floor to stand on. Every number is
written to results/sampling_quality.json, and the README quotes that
file, not this terminal.

Usage:
    python scripts/sample_quality.py [--steps 120] [--seed 42]
                                     [--out results/sampling_quality.json]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_net_lab.transformer import CharTransformerLM
from neural_net_lab.losses import CrossEntropyLoss
from neural_net_lab.optimizers import Adam

REPO = Path(__file__).resolve().parents[1]
PROMPT = "Shall I compare thee"
SAMPLE_CHARS = 300
TEMPERATURES = [0.5, 1.0, 1.5]


def get_batch(data, batch_size, block_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def quality(text):
    chars = list(text)
    pairs = list(zip(chars, chars[1:]))
    top_char, top_count = Counter(chars).most_common(1)[0]
    return {
        "chars": len(chars),
        "distinct_1": round(len(set(chars)) / max(len(chars), 1), 4),
        "distinct_2": round(len(set(pairs)) / max(len(pairs), 1), 4),
        "most_common_char": top_char,
        "most_common_char_share": round(top_count / max(len(chars), 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/sampling_quality.json")
    args = ap.parse_args()

    text = (REPO / "data" / "shakespeare_sonnets.txt").read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = np.array([stoi[c] for c in text], dtype=int)

    config = {"d_model": 64, "num_layers": 2, "num_heads": 4,
              "block_size": 64, "batch_size": 16, "lr": 0.003,
              "steps": args.steps, "seed": args.seed}
    model = CharTransformerLM(len(vocab), d_model=config["d_model"],
                              num_heads=config["num_heads"],
                              num_layers=config["num_layers"],
                              max_seq_len=config["block_size"], seed=args.seed)
    params = model.parameters()
    opt = Adam(params, lr=config["lr"])
    loss_fn = CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    start = time.time()
    first_loss = last_loss = None
    for step in range(args.steps):
        x, y = get_batch(data, config["batch_size"], config["block_size"], rng)
        logits = model.forward(x)
        loss = loss_fn.forward(logits.reshape(-1, len(vocab)), y.reshape(-1))
        if first_loss is None:
            first_loss = float(loss)
        last_loss = float(loss)
        opt.zero_grad()
        model.backward(loss_fn.backward().reshape(config["batch_size"], config["block_size"], len(vocab)))
        opt.step()
    wall = time.time() - start

    prompt_ids = [stoi[c] for c in PROMPT]
    samples = []
    for temp in TEMPERATURES:
        gen = model.generate(prompt_ids, SAMPLE_CHARS, temperature=temp,
                             rng=np.random.default_rng(1000 + int(temp * 10)))
        sample_text = "".join(vocab[i] for i in gen[len(prompt_ids):])
        samples.append({"temperature": temp, **quality(sample_text),
                        "sample": sample_text})

    corpus_window = text[:SAMPLE_CHARS]
    artifact = {
        "model": {"vocab_size": len(vocab), "parameters": int(model.count_parameters()), **config},
        "training": {"first_loss": round(first_loss, 3), "last_loss": round(last_loss, 3),
                     "wall_seconds": round(wall, 1)},
        "prompt": PROMPT,
        "corpus_window_quality": quality(corpus_window),
        "samples": samples,
        "note": "One small model, one seed, one prompt. These numbers describe this run, not the lab's larger runs.",
    }
    out = REPO / args.out
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"train loss {first_loss:.3f} -> {last_loss:.3f} in {wall:.1f}s")
    print(f"corpus window: distinct-1 {artifact['corpus_window_quality']['distinct_1']}, distinct-2 {artifact['corpus_window_quality']['distinct_2']}")
    for s in samples:
        print(f"temp {s['temperature']}: distinct-1 {s['distinct_1']}, distinct-2 {s['distinct_2']}, top char share {s['most_common_char_share']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
