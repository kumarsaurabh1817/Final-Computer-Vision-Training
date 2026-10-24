"""
Convert the Week 4 percent-format scripts (``# %%`` cells) into Colab notebooks.

Each day is authored once as a plain ``.py`` file so it can be diffed, linted, and
run with ``python dayN_*.py``. This tool turns it into a Colab-ready ``.ipynb``
with a T4 GPU runtime preselected. Standard library only.

Cell markers:
    # %% [markdown]   -> markdown cell (leading "# " stripped from each line)
    # %%              -> code cell
Anything before the first marker (module docstring) is skipped.

Every day folder is self-contained: its .py holds all the helper code it needs, so the
notebook needs nothing outside its own folder.

Usage:
    python tools/py_to_ipynb.py                      # convert every day folder
    python tools/py_to_ipynb.py day1_ocr_preprocessing/day1_ocr_preprocessing.py
"""

import json
import sys
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parent.parent

# Source script -> notebook name
NOTEBOOKS = {
    "day1_ocr_preprocessing/day1_ocr_preprocessing.py": "Week4_Day1_Colab.ipynb",
    "day2_layout_extraction/day2_layout_extraction.py": "Week4_Day2_Colab.ipynb",
    "day3_managed_services/day3_managed_services.py": "Week4_Day3_Colab.ipynb",
    "day4_kv_tables/day4_kv_tables.py": "Week4_Day4_Colab.ipynb",
    "day5_evaluation/day5_evaluation.py": "Week4_Day5_Colab.ipynb",
}


def _finish(cells: list[dict], kind: str | None, lines: list[str]) -> None:
    """Append a cell if it has non-blank content."""
    if kind is None:
        return
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)


def convert(py_path: Path, nb_path: Path) -> int:
    cells: list[dict] = []
    kind: str | None = None
    buf: list[str] = []

    for raw in py_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("# %%"):
            _finish(cells, kind, buf)
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        if kind == "markdown":
            if raw.startswith("# "):
                buf.append(raw[2:])
            elif raw.startswith("#"):
                buf.append(raw[1:])
            else:
                buf.append(raw)
        elif kind == "code":
            buf.append(raw)
    _finish(cells, kind, buf)

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Stable cell ids (nbformat 4.5)
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{i:03d}"

    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(cells)


def main(argv: list[str]) -> None:
    targets = argv or list(NOTEBOOKS)
    for rel in targets:
        rel = rel.replace("\\", "/")
        py_path = (WEEK_DIR / rel).resolve()
        nb_name = NOTEBOOKS.get(rel, py_path.stem + ".ipynb")
        nb_path = py_path.parent / nb_name
        n = convert(py_path, nb_path)
        print(f"{rel} -> {nb_path.relative_to(WEEK_DIR)} ({n} cells)")


if __name__ == "__main__":
    main(sys.argv[1:])
