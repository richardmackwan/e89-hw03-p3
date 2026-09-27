"""Assemble e89_Mackwan_Richard_HW03_Prob3.ipynb from the step scripts.

Each script becomes one code cell preceded by a markdown cell naming the
script. Blocks between "# [script-only]" and "# [/script-only]" (cross-script
imports and reloading saved artifacts) are dropped, because in the notebook
the earlier cells already defined those names.
Run from the repository root: python scripts/build_notebook.py
"""
import ast
import re
from pathlib import Path

import nbformat

SCRIPTS_DIR = Path(__file__).resolve().parent
NOTEBOOK = SCRIPTS_DIR.parent / "e89_Mackwan_Richard_HW03_Prob3.ipynb"
SCRIPT_ONLY = re.compile(r"# \[script-only\]\n.*?# \[/script-only\]\n", re.S)


def split_script(path):
    source = path.read_text()
    docstring = ast.get_docstring(ast.parse(source))
    # drop the module docstring (it becomes the markdown cell) and the
    # script-only blocks
    body = re.sub(r'\A""".*?"""\n', "", source, flags=re.S)
    body = SCRIPT_ONLY.sub("", body)
    return " ".join(docstring.split()), body.strip() + "\n"


cells = [nbformat.v4.new_markdown_cell(
    "# Richard Mackwan\n"
    "## Assignment 03, Problem 3\n\n"
    "Building an image classifier for Fashion MNIST with PyTorch, following "
    "the section *Building an Image Classifier with PyTorch* of "
    "`reference/10_neural_nets_with_pytorch.ipynb`.")]
for script in sorted(SCRIPTS_DIR.glob("step*.py")):
    description, code = split_script(script)
    cells.append(nbformat.v4.new_markdown_cell(
        f"**From `scripts/{script.name}`** — {description}"))
    cells.append(nbformat.v4.new_code_cell(code))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
nbformat.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK.name} with {len(cells)} cells")
