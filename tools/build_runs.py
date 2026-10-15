#!/usr/bin/env python3
"""Build the language model runs on the Machine page.

One set of artifacts, two layers, both written by this script:

- the surface layer (between the runs markers): plain words. Each run is
  one full sentence that says what the error number measures, how many
  parts the model had, and how many steps it learned over. The surface
  loss figures use the settled form for each run (three decimal places
  for the sonnets run, four for the Alice run).
- the layer for agents and builders (between the runs-tech markers):
  the exact table as it has always been produced from the results files,
  the gradient check figures, and the PyTorch comparison read from
  results/benchmark_pytorch.json.

Sources: results/char_lm_run.json, results/alice_char_lm_run.json, and
results/benchmark_pytorch.json. The gradient check count and its
tolerance stand in one place, in TECH_NOTE below, and match the test
suite in tests/test_transformer_gradients.py (7 tests, tolerance
1 in 100,000, which is 0.00001 in the test file).

Run: python3 tools/build_runs.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- runs:begin -->"
END = "<!-- runs:end -->"
TECH_BEGIN = "<!-- runs-tech:begin -->"
TECH_END = "<!-- runs-tech:end -->"

# label, results file, text name, decimal places for the surface sentence
RUNS = [
    ("Sonnets", "char_lm_run.json", "Shakespeare sonnets", 3),
    ("Alice", "alice_char_lm_run.json",
     "Alice's Adventures in Wonderland, Project Gutenberg ebook 11", 4),
]

TECH_NOTE = {
    "gradient_tests": 7,
    "gradient_total": 7,
    "tolerance_words": "1 in 100,000",
}


def load(filename):
    return json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))


def surface_block():
    paras = []
    for label, filename, _text_name, places in RUNS:
        data = load(filename)
        fmt = "%." + str(places) + "f"
        paras.append(
            "  <p>In the %s run, the model had %s parts and learned over "
            "%d steps. Its error on text it had not learned from fell from "
            "%s at the start to %s at the end.</p>"
            % (label.lower() if label != "Alice" else "Alice",
               format(data["parameters"], ","), data["config"]["steps"],
               fmt % data["initial_val_loss"], fmt % data["final_val_loss"]))
    return """%s
  <p>The model learned to write more like the text it was trained on. Its error was measured on text it had not learned from, and a lower error means its guesses at the next letter were closer.</p>
%s
  <p>The machine also checked its own math, and it got every check right, %d of %d.</p>
%s""" % (BEGIN, "\n".join(paras), TECH_NOTE["gradient_tests"],
         TECH_NOTE["gradient_total"], END)


def tech_block():
    rows = []
    for label, filename, text_name, _places in RUNS:
        data = load(filename)
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td>"
            "<td>%.3f</td><td>%.3f</td></tr>"
            % (label, text_name, format(data["parameters"], ","),
               data["config"]["steps"], data["corpus_chars"],
               data["initial_val_loss"], data["final_val_loss"]))
    bench = load("benchmark_pytorch.json")
    files = " and ".join("results/" + filename for _l, filename, _t, _p in RUNS)
    return """%s
  <h3 id="runs-h">The two language model runs in full</h3>
  <table class="runs-table">
    <thead><tr><th scope="col">Run</th><th scope="col">Text it learned from</th><th scope="col">Parameters</th><th scope="col">Steps</th><th scope="col">Characters of text</th><th scope="col">Validation loss at the start</th><th scope="col">Validation loss at the end</th></tr></thead>
    <tbody>
    %s
    </tbody>
  </table>
  <p>Key: parameters are the model's parts, counted by the code. Validation loss is the model's error on text it did not train on. Lower is better. The run figures stand in %s.</p>
  <p>Gradient checks: %d of %d tests pass, at a tolerance of %s. Tolerance here means how far a checked value may stray from the value worked out by hand before the check fails.</p>
  <p>PyTorch comparison, from results/benchmark_pytorch.json: a matching PyTorch model took %.1f milliseconds per step, against %.1f milliseconds per step for the model built here, so PyTorch was %.2f times faster per step.</p>
%s""" % (TECH_BEGIN, "\n    ".join(rows), files,
         TECH_NOTE["gradient_tests"], TECH_NOTE["gradient_total"],
         TECH_NOTE["tolerance_words"], bench["pytorch"]["mean_step_ms"],
         bench["numpy_lab"]["mean_step_ms"],
         bench["speed_ratio_torch_faster_by"], TECH_END)


def replace_block(page, begin, end, block, label):
    if begin not in page or end not in page:
        print("build_runs: %s markers not found" % label, file=sys.stderr)
        sys.exit(1)
    before, rest = page.split(begin, 1)
    _, after = rest.split(end, 1)
    return before + block + after


def main():
    path = ROOT / "index.html"
    page = path.read_text(encoding="utf-8")
    page = replace_block(page, BEGIN, END, surface_block(), "runs")
    page = replace_block(page, TECH_BEGIN, TECH_END, tech_block(), "runs-tech")
    path.write_text(page, encoding="utf-8")
    print("build_runs: wrote both layers from the results files")


if __name__ == "__main__":
    main()
