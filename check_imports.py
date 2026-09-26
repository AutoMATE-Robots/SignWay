#!/usr/bin/env python3
"""Report every module eval_openloop's import chain needs but can't find.

Walks the imports statically instead of discovering them one crash at a time.
"""
import ast
import importlib.util
import os
import sys

ROOTS = [
    os.path.expanduser("~/SignWay/signway_dataset/eval_openloop.py"),
    os.path.join(os.environ.get("SCRATCH", ""),
                 "openvla-oft/experiments/robot/openvla_utils.py"),
    os.path.join(os.environ.get("SCRATCH", ""),
                 "openvla-oft/experiments/robot/robot_utils.py"),
]
STDLIB = set(sys.stdlib_module_names)
LOCAL = {"experiments", "prismatic", "bag_to_episode", "eval_openloop",
         "tfds_builder", "inspect_dataset", "screen_bags"}


def top_level_imports(path):
    try:
        tree = ast.parse(open(path).read())
    except Exception as e:                                   # noqa: BLE001
        print(f"  [skip] {path}: {e}")
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return mods


need = set()
for r in ROOTS:
    if os.path.exists(r):
        need |= top_level_imports(r)
    else:
        print(f"  [missing file] {r}")

missing = []
for m in sorted(need - STDLIB - LOCAL):
    if importlib.util.find_spec(m) is None:
        missing.append(m)

print(f"\nchecked {len(need)} top-level imports")
if missing:
    print("MISSING:", " ".join(missing))
    print("\n  pip install " + " ".join(missing))
else:
    print("all third-party imports resolve in this env")
