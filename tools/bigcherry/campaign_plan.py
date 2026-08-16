"""RE14: real CampaignGraph construction for one campaign lane.

Two graphs, not one, per gpt-auto-agent's review of this design (verified
against the actual primitives before accepting): ``workspace.materialize()``
is what produces ``source_slice_id``, and ``StageNode`` is frozen with
``source_slice_id``/``build_plan_id`` as required identity fields that
``CampaignRun`` hashes before any stage executes. A single graph spanning
materialize -> generate -> build -> smoke would need those IDs before they
exist -- putting ``None`` or a guessed value on the downstream nodes would
weaken exactly the fail-closed identity guarantee RE14 exists to preserve.

So materialisation is its own one-node graph, run to completion first; its
real, verified ``source_slice_id`` (plus a ``BuildPlan`` built from it) then
seeds the second graph's real identity fields. This is the identity
resolution boundary, not a workaround.
"""

from __future__ import annotations

import hashlib

from .campaign_graph import CampaignGraph, ResourceClaim, StageNode


def _resource_key(prefix: str, value: str) -> str:
    """A resource_id derived from an arbitrary string (a filesystem path,
    here), digested rather than used raw. ResourceLock itself sanitises
    resource_id into a lock directory name by replacing every character
    outside [A-Za-z0-9_.-] with '_' -- two different raw paths could
    legitimately sanitise to the same lock name (e.g. paths differing only
    in punctuation the sanitiser strips). Digesting first removes that
    collision risk instead of relying on ResourceLock's sanitiser to also
    happen to keep unrelated paths apart.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}-{digest}"


def materialize_stage_graph(
    *, source_name: str, build_name: str, upstream_repo_path: str
) -> CampaignGraph:
    """Phase A: the one-node graph that resolves real source identity.

    Locked by upstream repository, not by destination path:
    ``UpstreamRepository.add_detached_worktree`` mutates the *shared*
    upstream clone's git worktree administration (``git worktree add``)
    even though the resulting checkout itself is isolated per source plan.
    Two concurrent materialisations against the same upstream clone are
    the actual race to prevent here.
    """
    stage_id = f"{source_name}:{build_name}:materialize"
    resource = ResourceClaim(
        kind="upstream-worktree",
        resource_id=_resource_key("upstream", upstream_repo_path),
    )
    return CampaignGraph(nodes=(
        StageNode(
            stage_id=stage_id, kind="materialize",
            source_slice_id=None, build_plan_id=None, workload_id=None,
            dependencies=(), resources=(resource,),
        ),
    ))


def build_stage_graph(
    *,
    source_name: str,
    build_name: str,
    source_slice_id: str,
    build_plan_id: str,
    workload_id: str | None,
    include_generate: bool = True,
    include_runtime_smoke: bool = True,
    gpu_resource_ids: tuple[str, ...] = (),
) -> CampaignGraph:
    """Phase B: [generate ->] build [-> runtime-smoke], with real identity.

    RE17: one flexible builder, not four named per-build-kind graph
    builders -- config.Build has no "kind" enum; stock/control/record/tune/
    replay are just names, not architecture. The real controls are
    ``include_generate`` (== ``build_cfg.variant_set is not None``) and
    ``include_runtime_smoke`` (== ``spec.validation is not None``), decided
    by the caller from real config, not branched on here by build_name.

    ``build_plan_id`` is on the generate node as well as the build node
    (when present), not just source_slice_id: generation for a
    workload-scoped variant set (e.g. ``workload-max``) depends on the
    inventory and variant set that BuildPlan itself carries, not only on
    which source it runs against. Putting build_plan_id on generate means
    its spec_hash (and therefore CampaignRun's reuse decision) changes if
    those inputs change, not only if the source does.

    The generate and build stages share one exclusive ``build-plan``
    resource claim (serialising them against any other stage building the
    same identity), while runtime-smoke claims the GPU resource(s) it
    actually needs to run on -- a real hardware claim, not a build-tree
    claim.
    """
    if not source_slice_id:
        raise ValueError("source_slice_id must be resolved before building this graph")
    if not build_plan_id:
        raise ValueError("build_plan_id must be resolved before building this graph")

    prefix = f"{source_name}:{build_name}"
    generate_id = f"{prefix}:generate"
    build_id = f"{prefix}:build"
    smoke_id = f"{prefix}:runtime-smoke"

    build_lock = ResourceClaim(kind="build-plan", resource_id=f"build-{build_plan_id}")
    gpu_claims = tuple(
        ResourceClaim(kind="gpu", resource_id=resource_id)
        for resource_id in sorted(gpu_resource_ids)
    )

    nodes: list[StageNode] = []
    if include_generate:
        nodes.append(StageNode(
            stage_id=generate_id, kind="generate",
            source_slice_id=source_slice_id, build_plan_id=build_plan_id,
            workload_id=workload_id, dependencies=(), resources=(build_lock,),
        ))
    nodes.append(StageNode(
        stage_id=build_id, kind="build",
        source_slice_id=source_slice_id, build_plan_id=build_plan_id,
        workload_id=workload_id,
        dependencies=(generate_id,) if include_generate else (),
        resources=(build_lock,),
    ))
    if include_runtime_smoke:
        nodes.append(StageNode(
            stage_id=smoke_id, kind="runtime-smoke",
            source_slice_id=source_slice_id, build_plan_id=build_plan_id,
            workload_id=workload_id, dependencies=(build_id,),
            resources=gpu_claims,
        ))
    return CampaignGraph(nodes=tuple(nodes))
