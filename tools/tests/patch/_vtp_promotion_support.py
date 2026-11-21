"""Shared assertions and pinned-source setup for promoted Radiance VTP patches.

The selected source composition, not the textual recipe order or numeric patch
IDs, determines which edits precede a focal patch.
"""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from bigcherry.patch import patchset, registry

REPO = Path(__file__).resolve().parents[3]
PATCHES = REPO / "engines/radiance/patches"



def recipe():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as handle:
        return tomllib.load(handle)


CANDIDATE_COMMIT = recipe()["pinned"]


def serving_chain_to(target: str) -> list[str]:
    """Actual build order of the production source through a serving-core patch."""
    data = recipe()
    core = data["patch-set"]["serving-core"]["patches"]
    assert core.count(target) == 1, f"{target} must occur exactly once in serving-core"
    source_sets = data["source"]["radiance"]["patch-sets"]
    assert source_sets.count("serving-core") == 1
    assert "serving-core" not in data["source"]["radiance-native"]["patch-sets"]
    selected = tuple(dict.fromkeys(
        patch for name in source_sets for patch in data["patch-set"][name]["patches"]
    ))
    # Resolve the build order for the focal mechanics test. The source policy
    # suite checks release eligibility, and assert_promoted_recipe checks the
    # focal packages' states.
    resolved = patchset.resolve_exact("radiance", selected, directory=PATCHES)
    ordered = [module.patch_id for module in resolved.modules]
    assert ordered.count(target) == 1
    return ordered[:ordered.index(target) + 1]


def assert_promoted_recipe(testcase, *targets: str) -> None:
    """Check state, production selection, and transitive REQUIRES ordering."""
    data = recipe()
    core = data["patch-set"]["serving-core"]["patches"]
    source_sets = data["source"]["radiance"]["patch-sets"]
    native_sets = data["source"]["radiance-native"]["patch-sets"]
    testcase.assertEqual(source_sets.count("serving-core"), 1)
    testcase.assertNotIn("serving-core", native_sets)
    testcase.assertFalse(
        any(target in data["patch-set"][name]["patches"] for name in native_sets for target in targets)
    )
    ordered = serving_chain_to(targets[-1])
    for target in targets:
        testcase.assertEqual(core.count(target), 1, target)
        with (PATCHES / target / "patch.toml").open("rb") as handle:
            meta = tomllib.load(handle)
        testcase.assertEqual(meta["state"], "validated", target)
        closure = patchset.expand_composition("radiance", (target,), directory=PATCHES).expanded
        for required in closure:
            testcase.assertIn(required, ordered, (target, required))
            if required != target:
                testcase.assertLess(ordered.index(required), ordered.index(target), (target, required))


def load_patch_files(name: str):
    """Use the production loader (PATCH/PATCHES and synthesized env docs)."""
    reg = registry.load_registry("radiance", PATCHES)
    return registry.load_implementation("radiance", reg.get(name), root=PATCHES)


def seed_pinned_paths(checkout: Path, root: Path, files, commit: str) -> None:
    """Stage pinned upstream paths; only an earlier create may lack a git blob."""
    created = {item.path for item in files if item.create}
    for path in sorted({item.path for item in files if not item.create}):
        dest = root / path
        if dest.exists():
            continue
        shown = subprocess.run(
            ["git", "-C", str(checkout), "show", f"{commit}:{path}"],
            capture_output=True,
        )
        if shown.returncode != 0:
            if path not in created:
                raise AssertionError(f"pinned source missing {path}: {shown.stderr.decode(errors='replace')}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(shown.stdout)


from contextlib import contextmanager
import os
import tempfile

from bigcherry.patcher import apply_all


@contextmanager
def patched_pinned_source(checkout: Path, target: str, *, full_checkout: bool = False):
    """Apply the migrated production prefix to the exact reviewed ABI43 source."""
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "--verify", f"{CANDIDATE_COMMIT}^{{commit}}"], text=True,
    ).strip()
    if commit != CANDIDATE_COMMIT:
        raise AssertionError(f"candidate source resolved to {commit}, expected {CANDIDATE_COMMIT}")
    names = serving_chain_to(target)
    groups = [(name, load_patch_files(name)) for name in names]
    files = [item for _, group in groups for item in group]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
        root = Path(temp) / "radiance"
        root.mkdir()
        if full_checkout:
            import io
            import tarfile
            archive = subprocess.run(
                ["git", "-C", str(checkout), "archive", "--format=tar", commit],
                check=True, capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(root, filter="data")
        seed_pinned_paths(checkout, root, files, commit)
        for name, group in groups:
            first = apply_all(group, root)
            assert first and all(item.ok for item in first), (name, first)
            if name == target:
                assert all(
                    edit.status == "applied" for item in first for edit in item.results
                ), (name, first)
        before = {item.path: (root / item.path).read_bytes() for item in files}
        for name, group in groups:
            again = apply_all(group, root)
            assert again and all(item.ok for item in again), (name, again)
            assert all(
                edit.status in {"already-applied", "not-applicable"}
                for item in again for edit in item.results
            ), (name, again)
        assert before == {item.path: (root / item.path).read_bytes() for item in files}, "replay changed bytes"
        yield root


def assert_focal_guards_unique(testcase, target: str) -> None:
    """No target guard may collide with itself or the full preceding build chain."""
    mine = [edit.guard for item in load_patch_files(target) for edit in item.edits]
    testcase.assertEqual(len(mine), len(set(mine)), target)
    others = [
        edit.guard
        for name in serving_chain_to(target)[:-1]
        for item in load_patch_files(name)
        for edit in item.edits
    ]
    for a in mine:
        for b in others:
            testcase.assertFalse(a == b or a.startswith(b) or b.startswith(a), (target, a, b))
