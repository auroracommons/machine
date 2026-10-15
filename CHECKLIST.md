# The 110 percent checklist: Machine

Ascent alignment: Layer 5, Autonomy. Aligned October 15, 2026 (see "The need" in the README and NEEDS.md in the profile repo).
State as of October 15, 2026. Every line carries its evidence; a line without evidence does not count as done. This file is maintained by the evolution pass.

## 100 percent: completeness
- [x] Every file of the verified local publish tree present on GitHub, proven by tree diff at launch (October 14 to 15, 2026)
- [x] README accurate against the code, including the published results: 7 of 7 gradient checks, char LM validation loss 5.297 to 2.413 over 250 steps, PyTorch benchmark 4.13x faster per step at identical parameters
- [x] LICENSE present
- [x] COLLABORATE.md present
- [x] CONTRIBUTING.md present
- [x] CI workflow present (tests plus the short training run on every push)

## The extra 10 percent
- [x] CI green on GitHub runners (verified October 15, 2026)
- [x] Repo description and topics set (description states the verified results)
- [x] Good first issues open: #1 gradient check for Dense, BatchNorm, or Dropout, #2 run the PyTorch benchmark on your machine (both labeled good first issue, sized in hours)
- [x] Tagged release: v0.1.0, first verified baseline (published, marked Latest)
- [x] Live demo: https://auroracommons.github.io/machine/ (render verified October 15, 2026)
- [x] Fresh baseline rerun October 15, 2026: python3 -m unittest discover -s tests, 7 tests, OK

## Working agreements
- AGENTS.md in this repo governs how changes are made and proven here.
