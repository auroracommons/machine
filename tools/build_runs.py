#!/usr/bin/env python3
"""Build the language model runs table on the Machine page.

Both runs are read from their committed results files, results/
char_lm_run.json and results/alice_char_lm_run.json, so the table and
the prose beside it share one source. Run this after a new run, and
commit the page it writes.

Run: python3 tools/build_runs.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- runs:begin -->"
END = "<!-- runs:end -->"

RUNS = [
    ("Sonnets", "char_lm_run.json", "Shakespeare sonnets"),
    ("Alice", "alice_char_lm_run.json",
     "Alice's Adventures in Wonderland, Project Gutenberg ebook 11"),
]


def main():
    rows = []
    for label, filename, text_name in RUNS:
        data = json.loads(
            (ROOT / "results" / filename).read_text(encoding="utf-8"))
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td>"
            "<td>%.3f</td><td>%.3f</td></tr>"
            % (label, text_name, format(data["parameters"], ","),
               data["config"]["steps"], data["corpus_chars"],
               data["initial_val_loss"], data["final_val_loss"]))
    table = """%s
<section aria-labelledby="runs-h">
  <h2 id="runs-h">The two language model runs</h2>
  <table class="runs-table">
    <thead><tr><th scope="col">Run</th><th scope="col">Text it learned from</th><th scope="col">Parameters</th><th scope="col">Steps</th><th scope="col">Characters of text</th><th scope="col">Validation loss at the start</th><th scope="col">Validation loss at the end</th></tr></thead>
    <tbody>
    %s
    </tbody>
  </table>
  <p>Key: parameters are the model's parts, counted by the code. Validation loss is the model's error on text it did not train on. Lower is better. Both runs stand in their results files in this repository, named in the build script for this table.</p>
</section>
%s""" % (BEGIN, "\n    ".join(rows), END)

    path = ROOT / "index.html"
    page = path.read_text(encoding="utf-8")
    if BEGIN in page and END in page:
        before, rest = page.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        page = before + table + after
    else:
        anchor = "  <section>\n      <h2>About</h2>"
        if anchor not in page:
            anchor = "<h2>About</h2>"
            if anchor not in page:
                print("build_runs: About section not found", file=sys.stderr)
                sys.exit(1)
            page = page.replace(anchor, table + "\n" + anchor, 1)
        else:
            page = page.replace(anchor, table + "\n" + anchor, 1)
    path.write_text(page, encoding="utf-8")
    print("build_runs: wrote the runs table from the results files")


if __name__ == "__main__":
    main()
