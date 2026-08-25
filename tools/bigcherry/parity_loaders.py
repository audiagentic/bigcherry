"""RE14: build a parity.CampaignArm from real campaign-published output.

Reads through :class:`ArtifactStore`'s content-addressed relative paths and
converges on :class:`~bigcherry.parity.CampaignArm`, so
:func:`~bigcherry.parity.check_parity` never needs to know anything about
how its inputs were produced.

RE23 note: this module used to have a sibling ``load_legacy_arm`` reading
plain files at conventional legacy-checkout paths, for the legacy-vs-new
parity comparison that ran during the RE14 cutover (see git history for
``re14_parity_run.py``, retired in RE23 once that comparison's job -- proving
the two paths agreed -- was done and the legacy path itself was deleted).
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import ArtifactError, ArtifactStore
from .build.builds import binary_hash
from .parity import CampaignArm


def load_new_arm(
    name: str, *, store: ArtifactStore, manifest_relative: str | Path,
    binary_relative: str | Path, manifest_content_hash: str | None = None,
    toolchain: dict[str, str] | None = None,
) -> CampaignArm:
    """Reads through ``ArtifactStore.resolve``, which raises if the artifact
    is missing -- the loader cannot silently succeed against a partially
    published new-path arm.

    ``manifest_content_hash``, when given, must be the manifest's own
    ArtifactRef.content_hash (as returned by the generate stage). Without
    it, this loader has no way to notice a manifest.json that was modified
    on disk after publication -- ``store.resolve`` only checks the file
    exists, not that its bytes still match what was published. A parity
    gate that silently trusted un-reverified bytes from the "new" side
    could be fooled into reporting false agreement by exactly the kind of
    post-publish corruption ArtifactStore exists to catch. ``binary_hash``
    needs no equivalent parameter: it is always recomputed fresh from the
    binary's actual bytes on disk, never trusted from stored metadata.
    """
    if manifest_content_hash is not None and not store.verify(manifest_relative, manifest_content_hash):
        raise ArtifactError(
            f"{name}: manifest at {manifest_relative!r} does not match its "
            f"published content_hash {manifest_content_hash!r} -- refusing "
            f"to compare parity against unverified content"
        )
    manifest_path = store.resolve(manifest_relative)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["build_descriptor"]
    candidate_names = frozenset(c["stable_name"] for c in manifest["candidates"])
    binary_path = store.resolve(binary_relative)
    return CampaignArm(
        name=name, manifest=manifest, descriptor=descriptor,
        candidate_names=candidate_names, binary_hash=binary_hash(binary_path),
        toolchain=toolchain if toolchain is not None else {},
    )
