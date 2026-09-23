"""Repository layout.

Everything else in the package resolves paths through here so the tree can be
rearranged without a grep-and-replace.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# tools/bigcherry/paths.py -> tools/bigcherry -> tools -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]

SRC_OVERLAY = REPO_ROOT / "src"
PATCHES = REPO_ROOT / "patches"
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

_ENV_LLAMA_ROOT = "BIGCHERRY_LLAMA_ROOT"


def llama_root(override: str | os.PathLike[str] | None = None) -> Path:
    """The llama.cpp checkout bigcherry patches and builds.

    Resolution order: explicit argument, ``BIGCHERRY_LLAMA_ROOT``, then the
    vendored default. The checkout is a real working tree — builds run from it
    in place — so it is deliberately not a temp directory.
    """
    if override is not None:
        return Path(override).resolve()
    from_env = os.environ.get(_ENV_LLAMA_ROOT)
    if from_env:
        return Path(from_env).resolve()
    return REPO_ROOT / "vendor" / "llama.cpp"


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
