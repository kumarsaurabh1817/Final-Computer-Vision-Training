"""
Copy the shared helper modules into each day's script, so every Colab notebook stays self-contained.

A day script marks where a shared module goes:

    # ==== shared: corpus ====
    ...                        <- replaced by tools/shared/corpus.py on every run
    # ==== end shared: corpus ====

A day script can also embed modules as one source string (used to write a service file at run time):

    # ==== shared-source: RAG_CORE_SRC = serving_header,llm,index ====
    ...                        <- replaced by RAG_CORE_SRC = r'''<the modules concatenated>'''
    # ==== end shared-source: RAG_CORE_SRC ====

Edit the module in tools/shared/ (never the copy inside a day script), then run:

    python tools/inline_shared.py && python tools/py_to_ipynb.py

Modules are looked up in this week's tools/shared/ first, then in the other weeks' tools/shared/ folders.
Standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parent.parent
SEARCH = [WEEK_DIR / "tools" / "shared"] + sorted(p / "tools" / "shared" for p in WEEK_DIR.parent.glob("week*") if p != WEEK_DIR)
BLOCK = re.compile(r"(# ==== shared: (\w+) ====\n)(.*?)(# ==== end shared: \2 ====)", re.S)
SOURCE_BLOCK = re.compile(r"(# ==== shared-source: (\w+) = ([\w,]+) ====\n)(.*?)(# ==== end shared-source: \2 ====)", re.S)
TRIPLE = "'" * 3


def module_source(name: str) -> str:
    for folder in SEARCH:
        path = folder / f"{name}.py"
        if path.exists():
            return path.read_text(encoding="utf-8").rstrip() + "\n"
    raise FileNotFoundError(f"shared module {name!r} not found in {[str(s) for s in SEARCH]}")


def source_string(name: str, modules: str) -> str:
    code = "\n\n".join(module_source(m) for m in modules.split(","))
    if TRIPLE in code:
        raise ValueError(f"{name}: a module contains triple single quotes and cannot be embedded as a raw string")
    return f"{name} = r{TRIPLE}{code}{TRIPLE}\n"


def inline(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    new, n = BLOCK.subn(lambda m: m.group(1) + module_source(m.group(2)) + m.group(4), text)
    new, k = SOURCE_BLOCK.subn(lambda m: m.group(1) + source_string(m.group(2), m.group(3)) + m.group(5), new)
    if new != text:
        path.write_text(new, encoding="utf-8")
    return n + k


def main(argv: list[str]) -> None:
    targets = [Path(a).resolve() for a in argv] or sorted(WEEK_DIR.glob("day*/day*.py"))
    for path in targets:
        print(f"{path.relative_to(WEEK_DIR)}: {inline(path)} shared block(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
