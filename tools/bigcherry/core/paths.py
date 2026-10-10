"""Repository layout.

Everything else in the package resolves paths through here so the tree can be
rearranged without a grep-and-replace.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from . import engines

# tools/bigcherry/paths.py -> tools/bigcherry -> tools -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]

# The llama.cpp pillar's layout (engines/llamacpp/engine.toml). SRC_OVERLAY and PATCHES are that engine's
# locations in this checkout; code working on another project root asks LLAMACPP for them.
LLAMACPP = engines.load(REPO_ROOT, engines.LLAMACPP)
SRC_OVERLAY = LLAMACPP.overlay_root(REPO_ROOT)
PATCHES = LLAMACPP.patches_root(REPO_ROOT)
PATCH_CATALOG = PATCHES / "catalog.toml"
# VA02: a reviewed, one-time structural-grandfather baseline for the
# RD-patch validation-package standard (docs/reference/testing/
# PATCH_VALIDATION.md) -- never auto-regenerated, see
# tools/bigcherry/patch/validation_policy.py.
VALIDATION_PACKAGE_GRANDFATHER = PATCHES / "_validation" / "validation-package-grandfather.json"
SQL = REPO_ROOT / "sql"
DOCS = REPO_ROOT / "docs"
ARTIFACTS = REPO_ROOT / "artifacts"
# NOT under ARTIFACTS: `artifacts/` is entirely gitignored, and a
# disposition (HI152) is a real, reviewed decision meant to be shared and
# persist across bumps/sessions/machines -- not a disposable working
# artifact. Found live: dispositions recorded under artifacts/pin-bump/
# were silently never committed, defeating the whole point.
DISPOSITIONS = REPO_ROOT / "dispositions"
CONFIG = REPO_ROOT / "config"
RECIPES = CONFIG / "recipes.toml"
EXTERNAL_SOURCES = CONFIG / "external-sources.toml"
EXPERIMENT_CONTRACTS = CONFIG / "experiment-contracts.toml"
MODELS = CONFIG / "models.toml"

_ENV_PRIMARY_ROOT = "BC_PRIMARY_ROOT"


class PrimaryRootError(RuntimeError):
    """The shared primary checkout cannot be resolved safely."""


def _git_common_dir(root: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise PrimaryRootError(
            f"cannot resolve git common directory from {root}: {detail.strip()}"
        ) from exc
    raw = result.stdout.strip()
    if not raw:
        raise PrimaryRootError(f"git returned an empty common directory for {root}")
    common = Path(raw)
    if not common.is_absolute():
        common = root / common
    return common.resolve()


def primary_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the primary checkout shared by all linked worktrees.

    BC_PRIMARY_ROOT is the sole environment override. Otherwise Git common-dir
    is authoritative. Resolution is fail-closed; a non-repository or an
    override naming a linked worktree is rejected.
    """
    configured = override if override is not None else os.environ.get(_ENV_PRIMARY_ROOT)
    if configured:
        root = Path(configured).expanduser().resolve()
        common = _git_common_dir(root)
        if common.parent != root:
            raise PrimaryRootError(
                f"BC_PRIMARY_ROOT must name the primary checkout, not a linked worktree: {root}"
            )
        return root

    common = _git_common_dir(REPO_ROOT)
    root = common.parent
    if not (root / ".git").exists():
        raise PrimaryRootError(
            f"git common directory {common} does not identify a primary checkout"
        )
    return root.resolve()


def llama_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the llama.cpp checkout patched and built by BigCherry.

    A command-scoped explicit argument wins. Otherwise the vendor checkout
    always belongs to the primary checkout so a slice worktree never creates
    or uses a second vendor clone.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    return LLAMACPP.vendor_root(primary_root())


def cuda_dir(root: Path) -> Path:
    return root / "ggml" / "src" / "ggml-cuda"


def vulkan_dir(root: Path) -> Path:
    return root / "ggml" / "src" / "ggml-vulkan"


def vulkan_shaders_dir(root: Path) -> Path:
    return vulkan_dir(root) / "vulkan-shaders"


def template_instances_dir(root: Path) -> Path:
    return cuda_dir(root) / "template-instances"


def generate_cu_files_py(root: Path) -> Path:
    return template_instances_dir(root) / "generate_cu_files.py"


def artifact_dir(revision: str) -> Path:
    """Per-revision artifact directory, e.g. ``artifacts/22dc605/``."""
    return ARTIFACTS / revision[:12]


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def evidence_dir(run_id: str, *, create: bool = True) -> Path:
    """Raw/large evidence for one run, at ``artifacts/<run_id>/``.

    ``run_id`` must be the plan-item ID that owns the run (``HI65``, ``PA04``),
    optionally with a short free-text suffix (``HI65-pass2``,
    ``2026-08-21-HI35-HI36-27b-r9700``) — never a free-text name with no
    traceable plan-item link. See docs/reference/tooling/TOOLING.md's
    "Evidence and acceptance boundaries" section: this is the raw/machine-local
    counterpart to ``docs/evidence/<run_id>/``'s compact, git-tracked record,
    and is what ``bigcherry check``'s ``TR14.ARTIFACT_UNTRACEABLE_RUN`` finding
    validates against. Do not invent a new top-level ``artifacts/`` naming
    scheme — call this helper instead so the convention is enforced in one
    place.
    """
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"evidence_dir: run_id {run_id!r} must start with a letter/digit and "
            "contain only letters, digits, '.', '_', '-' (no path separators)"
        )
    path = ARTIFACTS / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
