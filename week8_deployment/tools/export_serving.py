"""
Write the service files defined in day3_serving/day3_serving.py (SERVING_FILES) into week8_deployment/serving/,
so they can be built with Docker without running the notebook. Reads the script with ast; executes nothing.

Usage (after tools/inline_shared.py, so rag_core.py is up to date):
    python tools/export_serving.py
"""

from __future__ import annotations

import ast
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parent.parent
SCRIPT = WEEK_DIR / "day3_serving" / "day3_serving.py"
OUT = WEEK_DIR / "serving"


def main() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    strings: dict[str, str] = {}
    files = None
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            strings[name] = node.value.value
        elif name == "SERVING_FILES" and isinstance(node.value, ast.Dict):
            files = {}
            for k, v in zip(node.value.keys, node.value.values):
                files[k.value] = v.value if isinstance(v, ast.Constant) else strings[v.id]
    if files is None:
        raise SystemExit("SERVING_FILES not found")
    for rel, content in files.items():
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote serving/{rel} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
