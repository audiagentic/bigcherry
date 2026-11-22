#!/usr/bin/env python3
"""Write (or check) every native lab arm YAML listed in native-arms.json, using arm_yaml.write_arm."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[2]


def _arm_yaml():
    spec = importlib.util.spec_from_file_location("arm_yaml", LAB / "arm_yaml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipes() -> dict[str, dict]:
    return json.loads((LAB / "native-arms.json").read_text(encoding="utf-8"))["arms"]


def render(recipe: dict, out: Path) -> None:
    _arm_yaml().write_arm(REPO / "engines/radiance/configs" / f"{recipe['base']}.yaml", out, list(recipe["set"]))


def main(argv: list[str]) -> int:
    check = argv[:1] == ["--check"]
    stale = []
    for name, recipe in recipes().items():
        target = LAB / "native-arms" / f"{name}.yaml"
        if check:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "arm.yaml"
                render(recipe, out)
                if not target.is_file() or target.read_text(encoding="utf-8") != out.read_text(encoding="utf-8"):
                    stale.append(name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            render(recipe, target)
    if stale:
        print("stale native arm files: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
