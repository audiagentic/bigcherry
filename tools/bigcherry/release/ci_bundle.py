"""CI side of the Radiance runtime release bundle (BU01): pack the lane build a CI run just made, check the archive by restoring
it, and find an earlier release whose bundle already is this runtime.

A runtime is identified by the git trees of ``engines/radiance`` and of ``config``, the trees of the build tooling that decides
how the lane is compiled and configured (``BUILD_INPUTS``), and the ROCm version the build image carries (``runtime_key``).
Most radiance releases change only documentation or unrelated tooling, so a release whose key equals an earlier release's key
reuses that bundle instead of building again. A change to the build tooling changes the key, so a tooling change that alters
compile flags can never reuse a stale bundle (it may rebuild when it did not need to; that is the safe side). The receipt published beside the archive is the same
``runtime_assets`` entry a lab ``pin-release`` records in git, plus ``release_tag`` and ``runtime_key``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import runtime_assets as ra

RECEIPT_SUFFIX = ".receipt.json"
TAG = re.compile(r"bc-radiance-[0-9]+\.[0-9]+\.[0-9]+")
DOCKERFILE = Path(".github/docker/radiance-build/Dockerfile")
#: Paths whose content decides how the lane is built: the lane planner and compiler launchers, the campaign build code, the
#: engine adapter and source materialiser, the patch applier, and the build image. Release notes, lab scripts and the rest of
#: the CLI are not here, so editing them does not force a rebuild.
BUILD_INPUTS = (
    "tools/bigcherry/build",
    "tools/bigcherry/campaign",
    "tools/bigcherry/core",
    "tools/bigcherry/engine_adapters",
    "tools/bigcherry/source",
    "tools/bigcherry/patch",
    ".github/docker/radiance-build",
)


class CiBundleError(RuntimeError):
    pass


def image_rocm_version(root: Path) -> str:
    """The ROCm version the CI build image is built with (its ARG ROCM_VERSION), for deciding reuse before any build runs."""
    text = (root / DOCKERFILE).read_text(encoding="utf-8")
    found = re.search(r"(?m)^ARG ROCM_VERSION=([0-9][0-9.]*)\s*$", text)
    if not found:
        raise CiBundleError(f"{DOCKERFILE} has no ARG ROCM_VERSION=")
    return found.group(1)


def build_inputs_tree(root: Path) -> str:
    """One digest over the git tree ids of BUILD_INPUTS at HEAD (a missing path is an error: the list must follow the code)."""
    lines = []
    for rel in BUILD_INPUTS:
        lines.append(f"{rel} {ra.git_tree(root, rel)}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def runtime_key(root: Path, rocm_version: str) -> dict[str, str]:
    return {**ra.stock_identity(root), "build_inputs": build_inputs_tree(root), "rocm_version": rocm_version}


def lane_build(work_root: Path, plan_id: str | None = None) -> tuple[Path, str, str]:
    """The one lane build under ``work_root/builds``: (directory, plan id, sha256 of bin/radiance)."""
    pattern = f"*/{plan_id}" if plan_id else "*/*"
    found = [p for p in (work_root / "builds").glob(pattern) if (p / "bin/radiance").is_file()]
    if len(found) != 1:
        raise CiBundleError(f"expected one lane build in {work_root / 'builds'}, found {len(found)}")
    build = found[0]
    if not re.fullmatch(r"[0-9a-f]{32}", build.name):
        raise CiBundleError(f"unexpected lane build directory name: {build.name}")
    return build, build.name, ra.digest(build / "bin/radiance")


def pack_and_verify(tag: str, *, root: Path, work_root: Path, sdk: Path, plan_id: str | None = None) -> dict:
    """Pack the lane build into the release archive and restore it into a scratch work root; only a restorable archive is kept."""
    if not TAG.fullmatch(tag):
        raise CiBundleError(f"not a bc-radiance release tag: {tag}")
    build, plan, binary_sha = lane_build(work_root, plan_id)
    if not (build / ra.BUILD_METADATA).is_file():
        raise CiBundleError(f"the lane build has no {ra.BUILD_METADATA}: an installed bundle would not launch")
    record = {"notes": f"release asset lane build: radiance:stock:linux-radiance build_plan_id={plan} binary_sha256={binary_sha}"}
    asset = ra.pack(record, root, work_root, sdk, tag)
    out = work_root / "release-assets" / tag
    archive, checksum = out / asset["name"], out / asset["checksum_name"]
    scratch = Path(tempfile.mkdtemp(prefix="ci-bundle-verify-", dir=out))
    try:
        ra.restore(archive, checksum, asset, root, scratch, sdk)
    finally:
        shutil.rmtree(scratch)
    receipt = {**asset, "release_tag": tag, "runtime_key": runtime_key(root, asset["rocm_version"])}
    (out / (asset["name"][: -len(".tar.gz")] + RECEIPT_SUFFIX)).write_text(json.dumps(receipt, sort_keys=True, indent=1) + "\n")
    return receipt


def _gh(args: list[str], root: Path) -> str:
    result = subprocess.run(["gh", *args], cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise CiBundleError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()[-300:]}")
    return result.stdout


def find_reusable(tag: str, *, root: Path, rocm_version: str, limit: int = 40) -> tuple[str, dict] | None:
    """The newest earlier bc-radiance release whose published receipt has this runtime key, with that receipt."""
    want = runtime_key(root, rocm_version)
    listing = json.loads(_gh(["release", "list", "--limit", str(limit), "--json", "tagName"], root))
    for entry in listing:
        other = entry["tagName"]
        if other == tag or not TAG.fullmatch(other):
            continue
        info = json.loads(_gh(["release", "view", other, "--json", "assets"], root))
        names = [a["name"] for a in info.get("assets", []) if a["name"].endswith(RECEIPT_SUFFIX)]
        if not names:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            _gh(["release", "download", other, "-p", names[0], "-D", tmp], root)
            receipt = json.loads((Path(tmp) / names[0]).read_text())
        if receipt.get("runtime_key") == want and receipt.get("schema") == ra.SCHEMA:
            return other, receipt
    return None


def cmd_release_bundle(args) -> int:
    root = Path(args.root).resolve()
    try:
        if args.find_reusable:
            rocm = args.rocm or image_rocm_version(root)
            found = find_reusable(args.tag, root=root, rocm_version=rocm)
            print(json.dumps({"reuse": found[0], "receipt": found[1]} if found else {"reuse": None}, sort_keys=True))
            return 0
        if not args.sdk:
            raise CiBundleError("--sdk is required to pack a bundle")
        sdk = Path(args.sdk)
        receipt = pack_and_verify(args.tag, root=root, work_root=Path(args.work), sdk=sdk, plan_id=args.plan_id)
    except (CiBundleError, ra.BundleError, OSError, ValueError, KeyError) as exc:
        print(f"release-bundle: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("release-bundle",
                       help="CI: pack and verify the runtime bundle of a lane build, or find an earlier release with the same runtime")
    p.add_argument("tag", help="bc-radiance-* release tag the bundle is for")
    p.add_argument("--root", default=".", help="repository checkout at the tag")
    p.add_argument("--work", default="work", help="work root holding builds/ (default work)")
    p.add_argument("--sdk", help="ROCm SDK root the build used (required to pack; the build names its SDK, never the host default)")
    p.add_argument("--plan-id", help="lane build plan id (default: the one build under the work root)")
    p.add_argument("--find-reusable", action="store_true",
                   help="print the earlier release whose bundle has this runtime key (no build needed), then exit")
    p.add_argument("--rocm", help="with --find-reusable: ROCm version of the build image (default: the Dockerfile's)")
    p.set_defaults(func=cmd_release_bundle)
