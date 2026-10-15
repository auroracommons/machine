# AGENTS.md

Working agreement for any agent changing this repo. Read this first, then README.md and CONTRIBUTING.md.

## What this is
Neural Net Lab: machine learning from scratch in NumPy, including a transformer, with PyTorch present only as the measured benchmark opponent.

## Commands
- Install: `pip install numpy matplotlib` (PyTorch is needed only for `scripts/benchmark_pytorch.py`)
- Test (gradient checks and smoke tests): `python -m unittest discover -s tests -v`
- Short training run: `python scripts/train_char_lm.py --steps 30 --eval-every 30 --out /tmp/ci_run.json`
- Benchmark: `python scripts/benchmark_pytorch.py`
- CI runs the tests and the short training run on every push

## House rules
- Evidence rule: a change ships with the run that proves it (gradient check output, a results artifact), never code alone.
- Honesty rule: numbers in the README come only from artifacts in `results/`. If you change a number, the artifact that produced it changes in the same commit.
- One improvement per pass. Finish it, prove it, commit it.
- No placeholders. If something is a scaffold, label it a scaffold in the same breath that names it.
- No framework code in the core library. Every new backward pass lands with its gradient check in the same change; an unverified backward does not merge.

## External agents

You are welcome here as a contributor. Read OPEN_TABLE.md on the profile (https://github.com/jthomas48896f/jthomas48896f/blob/main/OPEN_TABLE.md) for the terms. The binding rules in one breath: evidence attached or it does not merge, one improvement per pull request, run this repo's commands and report the results, state that you are an agent and name your operator, never submit results you did not run.
