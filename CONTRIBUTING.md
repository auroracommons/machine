# Contributing to Neural Net Lab

Contributions are welcome, and the door is kept visibly open: there
is always at least one good first issue available. The house rule is
the evidence rule: a contribution ships with the run that proves it
(the gradient check output, the benchmark artifact), not just the
code.

## Setup

```bash
pip install numpy matplotlib
python -m unittest discover -s tests   # gradient checks + training smoke
python main.py                         # the original demos
```

## Good first issues (sized in hours)

- **Add a gradient check (1 to 2 hours).** The original layers
  (Dense, BatchNorm, Dropout) predate the test suite. Pick one and
  add a finite difference check in the style of
  `tests/test_transformer_gradients.py`.
- **Grow the eval corpus (about an hour).** Add one public domain
  text to `data/` with its source named, and record the trained
  model's loss on it next to the sonnets run. Comparisons across
  corpora teach more than single numbers.
- **Run the benchmark on your machine (under an hour).** Run
  `scripts/benchmark_pytorch.py` and file the JSON with your CPU and
  thread counts noted. The published table is one machine; more
  machines make it a measurement instead of an anecdote.
- **Write the missing explanation (1 to 2 hours).** The README
  explains that the key bias gradient is zero by symmetry. Pick one
  other gradient in `transformer.py` and write its derivation as a
  comment or a short docs note, checked against the finite
  difference test.

## Ground rules

- No framework code in the core library. PyTorch appears only in
  the benchmark script, as the measured opponent.
- Every new backward pass lands with its gradient check in the same
  change. An unverified backward does not merge.
- Numbers in the README come only from artifacts in `results/`.
  If you change a number, the artifact that produced it changes in
  the same commit.

## Pull requests

Keep them small and single purpose. State the evidence in the PR
description: what you ran and what it showed. Contributors are
named in release notes.
