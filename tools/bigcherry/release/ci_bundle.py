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

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from . import runtime_assets as ra

TAG = re.compile(r"bc-radiance-[0-9]+\.[0-9]+\.[0-9]+")
DOCKERFILE = Path(".github/docker/radiance-build/Dockerfile")
#: Compile-affecting source only: build executor/compiler, Radiance adapter,
#: source materialiser and patch applier. The engine's recipes, patches and
#: components are already covered by stock_identity(engine_tree). Whole
#: campaign/core/patch trees would also pull in jobs, release, slice and
#: worktree ceremony, which must not invalidate an engine binary's reuse key.
# Reviewed source of truth for build-key scope. The intentionally separate
# literal test oracle catches policy drift; static import discovery tests catch
# newly referenced modules without silently changing the runtime key.
BUILD_INPUTS = (
    ".github/docker/radiance-build",
    ".github/workflows/radiance-build.yml",
    "tools/bigcherry/build",
    "tools/bigcherry/campaign/build.py",
    "tools/bigcherry/campaign/source.py",
    "tools/bigcherry/core/config.py",
    "tools/bigcherry/core/engines.py",
    "tools/bigcherry/core/canonical.py",
    "tools/bigcherry/engine_adapters/base.py",
    "tools/bigcherry/engine_adapters/radiance.py",
    "tools/bigcherry/engine_adapters/radiance_inventory.py",
    "tools/bigcherry/engine_adapters/radiance_resources.py",
    "tools/bigcherry/source/identity.py",
    "tools/bigcherry/source/sources.py",
    "tools/bigcherry/source/upstream.py",
    "tools/bigcherry/source/workspace.py",
    "tools/bigcherry/patch/apply.py",
    "tools/bigcherry/patch/overlay.py",
    "tools/bigcherry/patch/patchset.py",
    "tools/bigcherry/patch/selection.py",
    "tools/bigcherry/patch/registry.py",
    "tools/bigcherry/patch/catalog.py",
    "tools/bigcherry/patch/pinned_source.py",
    "tools/bigcherry/patch/source.py",
    "tools/bigcherry/bin/bc-ccache",
    "tools/bigcherry/campaign/resolution.py",
    "tools/bigcherry/campaign/runtime_bundle.py",
    "tools/bigcherry/campaign/ccache_paths.py",
    "tools/bigcherry/core/artifacts.py",
    "tools/bigcherry/core/component_coupling.py",
    "tools/bigcherry/core/components.py",
    "tools/bigcherry/core/context.py",
    "tools/bigcherry/core/csource.py",
    "tools/bigcherry/core/environment.py",
    "tools/bigcherry/core/paths.py",
    "tools/bigcherry/core/process.py",
    "tools/bigcherry/core/provenance.py",
    "tools/bigcherry/engine_adapters/radiance_bump.py",
    "tools/bigcherry/engine_adapters/radiance_layout.py",
    "tools/bigcherry/patch/docs.py",
    "tools/bigcherry/source/audit.py",
)


class CiBundleError(RuntimeError):
    pass


def discover_compile_dependencies(root: Path) -> tuple[str, ...]:
    base = root / "tools/bigcherry"
    pending = ["campaign/build.py", "engine_adapters/radiance.py", "patch/apply.py"]
    seen: set[str] = set()
    allowed = {"build", "campaign", "patch", "source", "core", "engine_adapters"}
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        module = "bigcherry." + rel[:-3].replace("/", ".")
        package = module.rpartition(".")[0].split(".")
        tree = ast.parse((base / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    head = package[:len(package) - node.level + 1]
                    prefix = ".".join(head + ([node.module] if node.module else []))
                else:
                    prefix = node.module or ""
                names = [prefix] + [prefix + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if not name.startswith("bigcherry."):
                    continue
                path = name[len("bigcherry."):].replace(".", "/") + ".py"
                if (path.split("/")[0] in allowed and (base / path).is_file()
                        and path not in seen):
                    pending.append(path)
    return tuple("tools/bigcherry/" + item for item in sorted(seen))


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


def _payload_key(target: ra.ReleaseTarget, native_profiles: tuple[str, ...]) -> str:
    policy = ra.payload_contract_key(native_profiles=native_profiles)
    return hashlib.sha256(f"{policy} {target.lane} {target.schema_fingerprint} "
                          f"{target.semantic_fingerprint}".encode()).hexdigest()


def runtime_key(root: Path, rocm_version: str, *, native_profiles: tuple[str, ...] = ()) -> dict[str, str]:
    target = ra.release_target(root)
    profiles = target.select_profiles(native_profiles)
    return {**ra.stock_identity(root), "build_inputs": build_inputs_tree(root),
            "payload_contract": _payload_key(target, profiles), "rocm_version": rocm_version}


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


def pack_and_verify(tag: str, *, root: Path, work_root: Path, sdk: Path, plan_id: str | None = None,
                    native_profiles: tuple[str, ...] = ()) -> dict:
    """Pack the lane build into the release archive and restore it into a scratch work root; only a restorable archive is kept."""
    if not TAG.fullmatch(tag):
        raise CiBundleError(f"not a bc-radiance release tag: {tag}")
    _build, plan, binary_sha = lane_build(work_root, plan_id)
    target = ra.release_target(root)
    record = {"notes": f"release asset lane build: {target.lane} build_plan_id={plan} binary_sha256={binary_sha}"}
    asset = ra.pack(record, root, work_root, sdk, tag, native_profiles=native_profiles, target=target)
    out = work_root / "release-assets" / tag
    archive, checksum = out / asset["name"], out / asset["checksum_name"]
    scratch = Path(tempfile.mkdtemp(prefix="ci-bundle-verify-", dir=out))
    try:
        ra.restore(archive, checksum, asset, root, scratch, sdk)
    finally:
        shutil.rmtree(scratch)
    receipt = {**asset, "release_tag": tag,
               "runtime_key": runtime_key(root, asset["rocm_version"], native_profiles=native_profiles)}
    (out / (asset["name"][: -len(".tar.gz")] + ra.CI_RECEIPT_SUFFIX)).write_text(json.dumps(receipt, sort_keys=True, indent=1) + "\n")
    return receipt


def tagged_config(root: Path, tag: str, name: str) -> bytes:
    """Read the immutable config at the exact release tag, never HEAD."""
    if name not in ra.NATIVE_RELEASE_PROFILES or not TAG.fullmatch(tag):
        raise CiBundleError("unapproved tag/config in engine release")
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"refs/tags/{tag}:engines/radiance/configs/{name}"],
        check=False, capture_output=True,
    )
    if result.returncode:
        raise CiBundleError(f"RSC12 config is absent from release tag {tag}: {name}")
    return result.stdout



def tagged_heat(root: Path, tag: str, name: str) -> bytes:
    """Read the approved heat source from the immutable release tag."""
    if name not in ra.SHIPPED_HEAT_SEEDS or not TAG.fullmatch(tag):
        raise CiBundleError("unapproved tag/heat in engine release")
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"refs/tags/{tag}:tools/lab/radiance/heat/{name}"],
        check=False, capture_output=True,
    )
    if result.returncode:
        raise CiBundleError(f"heat seed is absent from release tag {tag}: {name}")
    return result.stdout

def tagged_build_inputs_tree(root: Path, tag: str) -> str:
    """Compile-affecting inputs of the selected immutable tag, not checkout HEAD."""
    if not TAG.fullmatch(tag):
        raise CiBundleError("invalid release tag for build-input identity")
    lines = []
    for rel in BUILD_INPUTS:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"refs/tags/{tag}:{rel}"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode or not re.fullmatch("[a-f0-9]{40}", result.stdout.strip()):
            raise CiBundleError(f"missing tagged build input at {tag}: {rel}")
        lines.append(f"{rel} {result.stdout.strip()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def verify_publish_candidate(tag: str, directory: Path, *, root: Path) -> dict:
    """Final mandatory, SDK-free, Python-free payload gate before GH upload.

    Both freshly built and verbatim reused assets pass the *same* real
    archive extraction and file-map checks. Nothing under 'tools/' enters
    the runtime archive; this Python implementation is platform-side CI
    tooling, not a release-time interpreter dependency.
    """
    if not TAG.fullmatch(tag):
        raise CiBundleError("publishing requires one exact Radiance release tag")
    if not directory.is_dir() or directory.is_symlink():
        raise CiBundleError("bundle staging directory is missing or unsafe")
    entries = list(directory.iterdir())
    if any(not item.is_file() or item.is_symlink() for item in entries):
        raise CiBundleError("bundle staging contains nonregular files")
    archives = [item for item in entries if ra.CI_ARCHIVE.fullmatch(item.name)]
    if len(archives) != 1:
        raise CiBundleError("bundle staging must contain exactly one engine archive")
    archive = archives[0]
    stem = archive.name[:-len(".tar.gz")]
    receipt_path = directory / (stem + ra.CI_RECEIPT_SUFFIX)
    checksum = directory / (archive.name + ".sha256")
    if {item.name for item in entries} != {archive.name, receipt_path.name, checksum.name}:
        raise CiBundleError("bundle staging has missing or unexpected release files")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise CiBundleError("bundle staging release receipt is unreadable") from exc
    ra.require_product("upload receipt", receipt)
    if (receipt.get("schema") != ra.SCHEMA or receipt.get("payload_scope") != ra.RUN_SCOPE
            or receipt.get("release_tag") != tag or receipt.get("name") != archive.name
            or receipt.get("checksum_name") != checksum.name):
        raise CiBundleError("upload receipt has wrong schema, scope, tag or archive name")
    expected = receipt.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch("[a-f0-9]{64}", expected):
        raise CiBundleError("upload receipt has no valid archive digest")
    if checksum.read_text().strip() != f"{expected}  {archive.name}":
        raise CiBundleError("upload checksum differs from tagged release receipt")
    if ra.digest(archive) != expected:
        raise CiBundleError("upload archive does not match tag-bound digest")
    origin = receipt.get("reused_from", tag)
    if not isinstance(origin, str) or not TAG.fullmatch(origin):
        raise CiBundleError("upload archive has invalid original release tag")
    if "reused_from" in receipt and origin == tag:
        raise CiBundleError("redundant reused_from tag")
    identity = ra.tag_identity(root, tag)
    if origin != tag and ra.tag_identity(root, origin) != identity:
        raise CiBundleError("reused archive source trees do not match target release")
    key = receipt.get("runtime_key")
    names = receipt.get("file_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise CiBundleError("upload receipt has no valid runtime file names")
    target = ra.release_target(root, tag)
    native_profiles = ra.selected_native_profiles(set(names))
    if native_profiles != target.select_profiles(native_profiles):
        raise CiBundleError("release archive omits the native release's required profiles")
    if (not isinstance(key, dict)
            or set(key) != {"engine_tree", "config_tree", "build_inputs",
                            "payload_contract", "rocm_version"}
            or any(key.get(k) != v or receipt.get(k) != v for k, v in identity.items())
            or key.get("build_inputs") != tagged_build_inputs_tree(root, tag)
            or key.get("rocm_version") != receipt.get("rocm_version")
            or key.get("payload_contract") != _payload_key(target, native_profiles)):
        raise CiBundleError("release runtime key differs from tagged source/payload policy")
    with tempfile.TemporaryDirectory(prefix="rsp01-upload-verify-") as scratch:
        extracted = Path(scratch) / "unpacked"
        manifest = ra.checked_extract(archive, receipt, extracted)
        ra.require_product("upload manifest", manifest)
        if manifest.get("serving_contract") != target.serving_contract(manifest["files"]):
            raise CiBundleError("archive engine/owner contract differs from the tagged release")
        if manifest.get("release_tag") != origin:
            raise CiBundleError("archive manifest belongs to a different original release")
        for name in native_profiles:
            if (extracted / "configs" / name).read_bytes() != tagged_config(root, tag, name):
                raise CiBundleError(f"released RSC12 config differs from tagged source: {name}")
        for name in ra.SHIPPED_HEAT_SEEDS:
            if (extracted / "heat" / name).read_bytes() != tagged_heat(root, tag, name):
                raise CiBundleError(f"released heat seed differs from tagged source: {name}")
    return receipt


def _gh(args: list[str], root: Path) -> str:
    result = subprocess.run(["gh", *args], cwd=root, check=False, capture_output=True, text=True)
    if result.returncode:
        raise CiBundleError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()[-300:]}")
    return result.stdout


def verify_reusable_archive(tag: str, root: Path, receipt: dict) -> None:
    """Verify the actual published archive, not merely its asserted scope."""
    name = receipt.get("name")
    if not isinstance(name, str) or not ra.CI_ARCHIVE.fullmatch(name):
        raise CiBundleError("unsafe reusable archive name")
    checksum = receipt.get("checksum_name")
    if checksum != name + ".sha256":
        raise CiBundleError("unsafe reusable checksum name")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _gh(["release", "download", tag, "-p", name, "-p", checksum,
             "-D", tmp], root)
        expected = f"{receipt['sha256']}  {name}"
        if (directory / checksum).read_text().strip() != expected:
            raise CiBundleError("reusable archive checksum differs from receipt")
        ra.require_product("reusable receipt", receipt)
        extracted = directory / "unpacked"
        manifest = ra.checked_extract(directory / name, receipt, extracted)
        ra.require_product("reusable bundle", manifest)
        if (receipt.get("release_tag") != tag or
                manifest.get("release_tag") != receipt.get("reused_from", tag)):
            raise CiBundleError("reusable archive source tag mismatch")
        target = ra.release_target(root, tag)
        native_profiles = ra.selected_native_profiles(set(manifest["files"]))
        if native_profiles != target.select_profiles(native_profiles):
            raise CiBundleError("reused archive omits the native release's required profiles")
        if manifest.get("serving_contract") != target.serving_contract(manifest["files"]):
            raise CiBundleError("reused engine/owner contract differs from the tagged release")
        for name in native_profiles:
            if (extracted / "configs" / name).read_bytes() != tagged_config(root, tag, name):
                raise CiBundleError(f"released native profile differs from tagged source: {name}")


def find_reusable(tag: str, *, root: Path, rocm_version: str, limit: int = 40,
                  native_profiles: tuple[str, ...] = ()) -> tuple[str, dict] | None:
    """The newest earlier bc-radiance release whose published receipt has this runtime key, with that receipt."""
    native_profiles = ra.release_target(root).select_profiles(native_profiles)
    want = runtime_key(root, rocm_version, native_profiles=native_profiles)
    listing = json.loads(_gh(["release", "list", "--limit", str(limit), "--json", "tagName"], root))
    for entry in listing:
        other = entry["tagName"]
        if other == tag or not TAG.fullmatch(other):
            continue
        info = json.loads(_gh(["release", "view", other, "--json", "assets"], root))
        names = [a["name"] for a in info.get("assets", []) if a["name"].endswith(ra.CI_RECEIPT_SUFFIX)]
        if not names:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            _gh(["release", "download", other, "-p", names[0], "-D", tmp], root)
            receipt = json.loads((Path(tmp) / names[0]).read_text())
        if (receipt.get("runtime_key") != want or receipt.get("schema") != ra.SCHEMA
                or receipt.get("payload_scope") != ra.RUN_SCOPE):
            continue
        names = receipt.get("file_names")
        if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
                or names != sorted(set(names))):
            continue
        try:
            ra.check_run_manifest(set(names))
            if set(ra.selected_native_profiles(set(names))) != set(native_profiles):
                continue
            verify_reusable_archive(other, root, receipt)
        except (ra.BundleError, CiBundleError, OSError, ValueError, KeyError, tarfile.TarError):
            continue
        return other, receipt
    return None


KEEP_MARK = "keep-bundle"


def prune_plan(*, root: Path, keep: int = 10, limit: int = 200) -> dict:
    """Which releases' bundle assets are older than the newest `keep` distinct runtimes (owner decision: keep the last 10).

    Releases are walked newest first. A release is kept when its runtime key is among the first `keep` distinct keys seen
    (releases that reuse a bundle share the key, so they count once), when it is the newest radiance release (the current pin's),
    or when its notes carry a `keep-bundle` line (owner-marked). Everything else with a bundle is listed for pruning.
    """
    listing = json.loads(_gh(["release", "list", "--limit", str(limit), "--json", "tagName,createdAt"], root))
    tags = sorted((e["tagName"] for e in listing if TAG.fullmatch(e["tagName"])),
                  key=lambda t: next(e["createdAt"] for e in listing if e["tagName"] == t), reverse=True)
    seen: list[str] = []
    keep_tags, prune = [], []
    for position, tag in enumerate(tags):
        info = json.loads(_gh(["release", "view", tag, "--json", "assets,body"], root))
        receipts = [a["name"] for a in info.get("assets", []) if a["name"].endswith(ra.CI_RECEIPT_SUFFIX)]
        if not receipts:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            _gh(["release", "download", tag, "-p", receipts[0], "-D", tmp], root)
            key = json.dumps(json.loads((Path(tmp) / receipts[0]).read_text()).get("runtime_key"), sort_keys=True)
        marked = any(line.strip() == KEEP_MARK for line in (info.get("body") or "").splitlines())
        if key not in seen:
            seen.append(key)
        if position == 0 or marked or seen.index(key) < keep:
            keep_tags.append(tag)
        else:
            prune.append({"tag": tag, "assets": [a["name"] for a in info["assets"]
                                                 if a["name"].startswith(ra.PRODUCT + "-")]})
    return {"keep": keep, "kept": keep_tags, "prune": prune}


def apply_prune(plan: dict, *, root: Path) -> None:
    for entry in plan["prune"]:
        for name in entry["assets"]:
            _gh(["release", "delete-asset", entry["tag"], name, "-y"], root)


def cmd_release_bundle(args) -> int:
    root = Path(args.root).resolve()
    try:
        native_profiles = tuple(args.native_profile)
        if native_profiles and (args.verify_publish_dir is not None or args.prune or args.print_lane):
            raise CiBundleError("--native-profile selects members for packing or reuse lookup")
        if args.print_lane:
            if not TAG.fullmatch(args.tag):
                raise CiBundleError("lane selection requires one exact Radiance release tag")
            print(ra.release_target(root, args.tag).lane)
            return 0
        if args.verify_publish_dir is not None:
            result = verify_publish_candidate(args.tag, Path(args.verify_publish_dir), root=root)
            print(json.dumps({"verified": True, "tag": args.tag, "archive": result["name"]}, sort_keys=True))
            return 0
        if args.prune:
            plan = prune_plan(root=root, keep=args.keep)
            print(json.dumps(plan, sort_keys=True))
            if args.apply:
                apply_prune(plan, root=root)
            return 0
        if args.find_reusable:
            rocm = args.rocm or image_rocm_version(root)
            found = find_reusable(args.tag, root=root, rocm_version=rocm, native_profiles=native_profiles)
            print(json.dumps({"reuse": found[0], "receipt": found[1]} if found else {"reuse": None}, sort_keys=True))
            return 0
        if not args.sdk:
            raise CiBundleError("--sdk is required to pack a bundle")
        sdk = Path(args.sdk)
        receipt = pack_and_verify(args.tag, root=root, work_root=Path(args.work), sdk=sdk,
                                  plan_id=args.plan_id, native_profiles=native_profiles)
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
    p.add_argument("--native-profile", action="append", choices=ra.OPTIONAL_NATIVE_PROFILES, default=[],
                   help="include an explicitly approved committed YAML profile in the runtime archive")
    p.add_argument("--verify-publish-dir",
                   help="final pre-upload gate: verify exactly one Python-free tagged engine archive/receipt/checksum (no SDK)")
    p.add_argument("--print-lane", action="store_true", help="print the build lane selected by the exact release tag")
    p.add_argument("--find-reusable", action="store_true",
                   help="print the earlier release whose bundle has this runtime key (no build needed), then exit")
    p.add_argument("--prune", action="store_true",
                   help="list (and with --apply delete) the bundle assets of releases older than the newest --keep distinct runtimes")
    p.add_argument("--keep", type=int, default=10, help="with --prune: distinct runtimes to keep (default 10)")
    p.add_argument("--apply", action="store_true", help="with --prune: delete; without it only the plan is printed")
    p.add_argument("--rocm", help="with --find-reusable: ROCm version of the build image (default: the Dockerfile's)")
    p.set_defaults(func=cmd_release_bundle)
