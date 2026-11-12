"""Prepare an explicit experimental Radiance runtime overlay.

The original engine build and its plugins remain untouched. The overlay
uses a rebuilt bc-serve from this worktree to embed experimental layouts
but preserves the existing engine binary. This is not a release bundle.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil


def stage(*, engine_build: Path, serve_bin: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("destination already exists; refuse to overwrite")
    if destination.is_relative_to(engine_build) or engine_build.is_relative_to(destination):
        raise ValueError("staging directory must be separate from original engine build")
    engine = engine_build / "bin/radiance"
    home = engine_build / "radiance_home"
    metadata = engine_build / "bigcherry-build-metadata-radiance.json"
    if not engine.is_file() or not serve_bin.is_file() or not metadata.is_file():
        raise ValueError("missing source engine, planner or build metadata")
    for required in ("kernels/libr3.so", "kernels/libr4d.so", "architectures/qwen35_fp8.so"):
        if not (home / required).is_file():
            raise ValueError(f"engine source missing {required}")
    build_info = json.loads(metadata.read_text(encoding="utf-8"))
    if not build_info.get("build_id"):
        raise ValueError("source engine build has no build identity")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "bin").mkdir()
    (destination / "radiance_home").mkdir()
    (destination / "radiance_home/bin").mkdir()
    (destination / "bin/radiance").symlink_to(engine.resolve())
    for item in home.iterdir():
        if item.name == "bin":
            continue
        (destination / "radiance_home" / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    (destination / "radiance_home/bin/bc-serve").symlink_to(serve_bin.resolve())
    shutil.copyfile(metadata, destination / metadata.name)
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "type": "EXPERIMENTAL runtime overlay; NOT a verified release",
        "engine_binary": str(engine.resolve()),
        "source_build_id": build_info["build_id"],
        "serve_binary": str(serve_bin.resolve()),
        "warning": "Bundle hash covers original runtime, not this overlay; engine and planner are independently compiled.",
    }
    (destination / "OVERLAY-PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-build", type=Path, required=True)
    parser.add_argument("--serve-bin", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = stage(**vars(args))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
