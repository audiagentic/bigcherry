"""RE14: build a parity.CampaignArm from real legacy or new-path output.

Two loaders, not one, because the two paths publish their artifacts
differently: the legacy path writes plain files at conventional paths
(``artifact_dir(revision)/hip-autotune-manifest.json``, a ``_build_dir``
binary); the new path publishes everything through :class:`ArtifactStore`
under content-addressed relative paths. Both loaders converge on the same
:class:`~bigcherry.parity.CampaignArm` shape so :func:`~bigcherry.parity.check_parity`
never needs to know which path produced its inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import ArtifactStore
from .builds import binary_hash
from .parity import CampaignArm


def load_legacy_arm(
    name: str, *, manifest_path: Path, descriptor_path: Path, binary_path: Path,
) -> CampaignArm:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    candidate_names = frozenset(c["stable_name"] for c in manifest["candidates"])
    return CampaignArm(
        name=name, manifest=manifest, descriptor=descriptor,
        candidate_names=candidate_names, binary_hash=binary_hash(binary_path),
    )


def load_new_arm(
    name: str, *, store: ArtifactStore, manifest_relative: str | Path,
    binary_relative: str | Path,
) -> CampaignArm:
    """Reads through ``ArtifactStore.resolve``, which raises if the artifact
    is missing -- the loader cannot silently succeed against a partially
    published new-path arm.
    """
    manifest_path = store.resolve(manifest_relative)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["build_descriptor"]
    candidate_names = frozenset(c["stable_name"] for c in manifest["candidates"])
    binary_path = store.resolve(binary_relative)
    return CampaignArm(
        name=name, manifest=manifest, descriptor=descriptor,
        candidate_names=candidate_names, binary_hash=binary_hash(binary_path),
    )
