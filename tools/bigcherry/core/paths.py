"""Repository layout.

Everything else in the package resolves paths through here so the tree can be
rearranged without a grep-and-replace.
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/bigcherry/paths.py -> tools/bigcherry -> tools -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]

SRC_OVERLAY = REPO_ROOT / "src"
PATCHES = REPO_ROOT / "patches"
PATCH_CATALOG = PATCHES / "catalog.toml"
SQL = REPO_ROOT / "sql"
DOCS = REPO_ROOT / "docs"
ARTIFACTS = REPO_ROOT / "artifacts"
CONFIG = REPO_ROOT / "config"
RECIPES = CONFIG / "recipes.toml"
EXTERNAL_SOURCES = CONFIG / "external-sources.toml"
EXPERIMENT_CONTRACTS = CONFIG / "experiment-contracts.toml"

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
