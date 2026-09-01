"""Explicit patch-selection identity for the CLI.

``--source NAME`` resolves an EXACT, curated patch-id list via
``campaign/resolution.py``'s v2 machinery -- the sole selection mechanism.
The legacy ``--recipe``/``--groups``/``--states`` predicate-filter path
(and the ``[compat.recipe.*]`` config bridge it depended on) has been
removed entirely; there is no group/state filtering axis to override.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

    from .patchset import PatchModule


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class CliPatchSelection:
    label: str
    patch_ids: tuple[str, ...] = ()
    patch_set_id: str | None = None
    source_name: str | None = None
    source_ref: str | None = None
    #: Populated once by resolve_cli_selection(), never re-read from config
    #: by a caller (a caller that needs fresh overlay bytes calls
    #: tree_state_key() again, which is cheap; it does not re-derive this
    #: field).
    overlay: bool | None = None
    overlay_digest: str | None = None
    #: True only for the no-``--source`` browse-everything selection
    #: ``patches`` uses to list the whole catalog. Never set for a
    #: selection that will be used to mutate a checkout (``apply``
    #: rejects it explicitly).
    select_all: bool = False

    def matches(self, module: "PatchModule") -> bool:
        """Observational membership test (``patches``' own filtering)."""
        if self.select_all:
            return True
        return module.patch_id in self.patch_ids

    def tree_state_key(self, ref: str) -> str:
        """Fingerprint of the mutable checkout state this selection needs.

        ``ref`` must be a real, immutable commit SHA, never a movable
        symbolic ref.
        """
        if self.overlay is None:
            raise SelectionError(
                "selection's overlay flag was never resolved -- refusing "
                "to compute a tree-state key"
            )
        payload = {
            "schema": 1,
            "upstream_revision": ref,
            "patch_set_id": self.patch_set_id,
            "overlay": self.overlay,
            "overlay_digest": self.overlay_digest if self.overlay else None,
        }
        material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode()).hexdigest()[:16]


def resolve_cli_selection(args: "Namespace") -> CliPatchSelection:
    """The single selection resolver shared by ``patches`` and ``apply``.

    No ``--source`` resolves to the browse-everything selection -- valid
    for ``patches``' read-only listing, rejected explicitly by ``apply``.
    """
    source_name = getattr(args, "source", None)
    if not source_name:
        return CliPatchSelection(label="all patches (no --source)", select_all=True)
    return _resolve_exact_selection(source_name)


def _resolve_exact_selection(source_name: str) -> CliPatchSelection:
    from ..campaign import resolution as campaign_resolution  # noqa: PLC0415
    from ..core import config as campaign_config  # noqa: PLC0415
    from ..core import paths as core_paths  # noqa: PLC0415
    from . import patchset  # noqa: PLC0415

    try:
        cfg = campaign_config.load(core_paths.RECIPES)
        selection = campaign_resolution.resolve_canonical_selection(
            source_name, cfg, patchset.catalog(),
        )
    except (campaign_resolution.ResolutionError, campaign_config.ConfigError) as exc:
        raise SelectionError(str(exc)) from exc

    source = cfg.sources[source_name]
    overlay_digest_value = None
    if source.overlay:
        from .rebase import overlay_digest as _overlay_digest  # noqa: PLC0415

        overlay_digest_value = _overlay_digest()

    return CliPatchSelection(
        label=(
            f"source={source_name} ref={selection.source_ref} "
            f"patch_set_id={selection.patch_set_id}"
        ),
        patch_ids=selection.patch_ids,
        patch_set_id=selection.patch_set_id,
        source_name=source_name,
        source_ref=selection.source_ref,
        overlay=source.overlay,
        overlay_digest=overlay_digest_value,
    )
