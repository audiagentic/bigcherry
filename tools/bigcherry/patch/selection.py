"""Explicit patch-selection identity for the CLI (compat.recipe removal).

Two selection modes coexist during the legacy migration (gpt-dev-agent
reviewed design, dev-gpt-agent gateway session ses_5307d9c58ec645cb):

- ``"predicate"``: ``--recipe`` seeds a groups/states filter, and
  ``--groups``/``--states`` override either axis independently. Filters the
  WHOLE catalog by group/state. This is the legacy mechanism (unchanged
  behavior, unchanged identity), kept exactly as-is until
  ``recipes.py``'s ``[compat.recipe.*]`` bridge is deleted.
- ``"exact"``: ``--source`` resolves an EXACT, curated patch-id list via
  ``campaign/resolution.py``'s v2 machinery. No group/state filtering
  axis exists for this mode -- combining it with ``--groups``/``--states``
  is rejected, not silently reinterpreted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from . import patchset

if TYPE_CHECKING:
    from argparse import Namespace

    from .patchset import PatchModule


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class CliPatchSelection:
    mode: Literal["predicate", "exact"]
    label: str
    groups: frozenset[str] | None = None
    states: frozenset[str] | None = None
    patch_ids: tuple[str, ...] = ()
    patch_set_id: str | None = None
    source_name: str | None = None
    source_ref: str | None = None
    #: Only meaningful for mode="exact" -- populated once by
    #: resolve_cli_selection(), never re-read from config by a caller (a
    #: caller that needs fresh overlay bytes calls tree_state_key() again,
    #: which is cheap; it does not re-derive this field).
    overlay: bool | None = None
    overlay_digest: str | None = None

    def matches(self, module: "PatchModule") -> bool:
        """Observational membership test (``patches``' own filtering)."""
        if self.mode == "exact":
            return module.patch_id in self.patch_ids
        return (
            (self.groups is None or module.group in self.groups)
            and (self.states is None or module.state in self.states)
        )

    def tree_state_key(self, ref: str) -> str:
        """Fingerprint of the mutable checkout state this selection needs.

        Deliberately excludes ``source_name`` in exact mode -- two source
        aliases that resolve to byte-identical composition/overlay state
        should share one checkout-state key, matching the legacy
        ``recipes.tree_state_key()``'s own "effective checkout state, not
        logical recipe identity" contract. ``ref`` must be a real,
        immutable commit SHA, never a movable symbolic ref.
        """
        if self.mode == "predicate":
            from .. import recipes as recipes_module  # noqa: PLC0415

            return recipes_module.tree_state_key(ref, self.groups, self.states)
        if self.overlay is None:
            raise SelectionError(
                "exact selection's overlay flag was never resolved -- "
                "refusing to compute a tree-state key"
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

    Exactly one of ``args.source`` / ``args.recipe`` (with optional
    ``args.groups``/``args.states`` overrides) determines the mode.
    """
    source_name = getattr(args, "source", None)
    recipe_name = getattr(args, "recipe", None)
    override_groups = patchset.parse_filter(getattr(args, "groups", None))
    override_states = patchset.parse_filter(getattr(args, "states", None))

    if source_name:
        if override_groups is not None or override_states is not None:
            raise SelectionError(
                "--source cannot be combined with --groups/--states -- v2 "
                "patch-sets are exact, curated lists with no group/state "
                "filtering axis to override"
            )
        return _resolve_exact_selection(source_name)

    from .. import recipes as recipes_module  # noqa: PLC0415

    groups = states = None
    label_parts: list[str] = []
    if recipe_name:
        try:
            recipe = recipes_module.get(recipe_name)
        except recipes_module.RecipeError as exc:
            raise SelectionError(str(exc)) from exc
        groups, states = recipe.groups, recipe.states
        label_parts.append(f"recipe={recipe.name} ref={recipe.ref}")

    if override_groups is not None:
        groups = override_groups
        label_parts.append("groups overridden")
    if override_states is not None:
        states = override_states
        label_parts.append("states overridden")

    def show(value: frozenset[str] | None) -> str:
        if value is None:
            return "all"
        return ",".join(sorted(value)) or "none"

    label_parts.append(f"groups={show(groups)} states={show(states)}")
    return CliPatchSelection(
        mode="predicate", label="  ".join(label_parts),
        groups=groups, states=states,
    )


def _resolve_exact_selection(source_name: str) -> CliPatchSelection:
    from ..campaign import resolution as campaign_resolution  # noqa: PLC0415
    from ..core import config as campaign_config  # noqa: PLC0415
    from ..core import paths as core_paths  # noqa: PLC0415

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
        mode="exact",
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
