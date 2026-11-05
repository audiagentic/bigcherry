#!/usr/bin/env python3
"""BU01: export and safely import CI ccache on compiler-compatible lab hosts.

The CI artifact is produced only by a successful main build. Cache entries are
untrusted accelerators, never release binaries: verify digest and exact
toolchain fingerprint before copying them into BC_WORK_ROOT/.ccache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = "radiance-ccache-main"
ARCHIVE = "ci-ccache.tar.gz"
SCHEMA = "bigcherry-ci-ccache-v1"
MAX_UNPACK_BYTES = 16 * 1024**3


class CacheError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise CacheError(f"required compiler/metadata file missing: {path}")
    return path


def compiler(name: str) -> Path:
    path = shutil.which(name)
    if not path:
        raise CacheError(f"required compiler missing: {name}")
    return require_file(Path(path).resolve())


def fingerprint() -> dict:
    sdk_path = os.environ.get("ROCM_PATH")
    if not sdk_path:
        raise CacheError("ROCM_PATH must be configured before importing a CI cache")
    sdk = Path(sdk_path).resolve()
    version = require_file(sdk / ".info/version").read_text().strip()
    cc = os.environ.get("RADIANCE_HOST_CC", "gcc-14")
    cxx = os.environ.get("RADIANCE_HOST_CXX", "g++-14")
    ccbase = os.environ.get("CCACHE_BASEDIR", "")
    if Path(ccbase).resolve() != ROOT.resolve():
        raise CacheError("CCACHE_BASEDIR must equal this checkout root; set it before importing")
    if os.environ.get("CCACHE_NOHASHDIR", "").lower() not in {"1", "true", "yes"}:
        raise CacheError("CCACHE_NOHASHDIR=true required to share entries across build directories")
    if os.environ.get("CCACHE_COMPILERCHECK", "") != "content":
        raise CacheError("CCACHE_COMPILERCHECK=content required for portable compiler identity")
    # ccache versions can share format-compatible entries; only the actual
    # host/HIP compilers must be byte-identical. Require ccache to exist.
    compiler("ccache")
    compilers = {
        "cc": compiler(cc),
        "cxx": compiler(cxx),
        "clang++": require_file((sdk / "lib/llvm/bin/clang++").resolve()),
        "hipcc": require_file((sdk / "bin/hipcc").resolve()),
    }
    return {
        "schema": SCHEMA,
        "rocm_version": version,
        "compiler_sha256": {name: sha256(path) for name, path in compilers.items()},
        "recipes_sha256": {
            path: sha256(require_file(ROOT / path))
            for path in ("config/recipes.toml", "engines/radiance/recipes.toml")
        },
        "ccache_bases_relative": True,
        "compiler_check": "content",
    }


def write_metadata(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(fingerprint(), indent=2, sort_keys=True) + "\n")
    print(f"wrote CI ccache toolchain fingerprint: {dest}")


def pack(cache: Path, out_dir: Path) -> None:
    if not cache.is_dir():
        raise CacheError(f"ccache directory does not exist: {cache}")
    if not (out_dir / "compiler.json").is_file():
        raise CacheError("compiler.json must be generated inside the CI build image first")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / ARCHIVE
    with tarfile.open(archive, "w:gz", compresslevel=1, format=tarfile.PAX_FORMAT) as tar:
        for source in sorted(cache.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            name = PurePosixPath(".ccache") / PurePosixPath(source.relative_to(cache).as_posix())
            tar.add(source, arcname=str(name), recursive=False)
    digest = sha256(archive)
    (out_dir / "SHA256SUMS").write_text(f"{digest}  {ARCHIVE}\n")
    print(f"packaged {archive} sha256={digest}")


def verify(source: Path) -> Path:
    try:
        expected = json.loads(require_file(source / "compiler.json").read_text())
    except (ValueError, OSError) as exc:
        raise CacheError(f"invalid compiler.json: {exc}") from exc
    actual = fingerprint()
    if expected != actual:
        diffs = [key for key in actual if expected.get(key) != actual[key]]
        raise CacheError(f"CI cache compiler/recipe mismatch in {', '.join(diffs)}; refusing import. "
                         "Use the matching ROCm build image, host compilers and recipes.")
    archive = require_file(source / ARCHIVE)
    checksum = require_file(source / "SHA256SUMS").read_text().strip()
    if checksum != f"{sha256(archive)}  {ARCHIVE}":
        raise CacheError("CI cache archive checksum mismatch; refusing import")
    return archive


def import_cache(source: Path, work: Path) -> None:
    archive = verify(source)
    work.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="bu01-ccache-", dir=work))
    try:
        size = 0
        with tarfile.open(archive, "r:gz") as tar:
            for item in tar:
                rel = PurePosixPath(item.name)
                if (rel.is_absolute() or ".." in rel.parts or not rel.parts
                    or rel.parts[0] != ".ccache" or item.issym() or item.islnk()):
                    raise CacheError(f"unsafe CI cache archive member: {item.name}")
                if item.isdir():
                    continue
                if not item.isfile():
                    raise CacheError(f"non-regular CI cache archive member: {item.name}")
                size += item.size
                if size > MAX_UNPACK_BYTES:
                    raise CacheError("CI cache exceeds import size limit")
                target = temp.joinpath(*rel.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(item) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
        target_root = work / ".ccache"
        for source_file in temp.rglob("*"):
            if source_file.is_file():
                dest = target_root / source_file.relative_to(temp)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest)
        print(f"imported {size} bytes of verified CI ccache to {target_root}")
    finally:
        shutil.rmtree(temp)


def gh(args: list[str], *, cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(["gh", *args], cwd=cwd, check=True, text=True,
                                capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CacheError(f"GitHub CLI failed: {exc}") from exc
    return result.stdout


def fetch(work: Path, run_id: str | None) -> None:
    if not run_id:
        runs = json.loads(gh(["run", "list", "-w", "radiance-build.yml", "-b", "main",
                              "-s", "success", "-L", "1", "--json", "databaseId"]))
        if not runs:
            raise CacheError("no successful main radiance-build workflow to seed from")
        run_id = str(runs[0]["databaseId"])
    info = json.loads(gh(["run", "view", run_id, "--json",
                          "headBranch,conclusion,event,workflowName"]))
    if (info.get("headBranch") != "main" or info.get("conclusion") != "success"
        or info.get("workflowName") != "Radiance build"
        or info.get("event") not in {"push", "workflow_dispatch"}):
        raise CacheError("CI cache must come from a successful main Radiance build")
    work.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="bu01-ccache-download-", dir=work))
    try:
        gh(["run", "download", run_id, "-n", ARTIFACT, "-D", str(temp)])
        import_cache(temp, work)
    finally:
        shutil.rmtree(temp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    meta = sub.add_parser("fingerprint")
    meta.add_argument("--output", type=Path, required=True)
    pkg = sub.add_parser("pack")
    pkg.add_argument("--cache", type=Path, required=True)
    pkg.add_argument("--out", type=Path, required=True)
    inp = sub.add_parser("import")
    inp.add_argument("--source", type=Path, required=True)
    get = sub.add_parser("fetch")
    get.add_argument("--run-id")
    for action in (inp, get):
        action.add_argument("--work-root", type=Path,
                            default=Path(os.environ.get("BC_WORK_ROOT", ROOT / "work")))
    args = parser.parse_args()
    try:
        if args.command == "fingerprint":
            write_metadata(args.output)
        elif args.command == "pack":
            pack(args.cache, args.out)
        elif args.command == "import":
            import_cache(args.source, args.work_root)
        else:
            fetch(args.work_root, args.run_id)
    except CacheError as exc:
        print(f"BU01 ccache: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
