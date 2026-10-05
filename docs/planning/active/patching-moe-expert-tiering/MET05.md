---
id: MET05
order: 5
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:45:07.116948+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# 1328 aux_rocm_expert_backend: 6900 XT (ROCm3) as auxiliary expert device outside Meta/RCCL

## Description

Let ROCm3 hold and compute selected cold routed-expert layers without joining the tensor-split Meta buffer type or RCCL/cpu-root AllReduce. The implementation is env-gated (`BIGCHERRY_EXPERT_AUX_DEVICE=ROCm3`), uses ordinary ROCm3 weight/compute buffers, and stages Meta<->ROCm3 activations through pinned host memory.

## Steps

1. Register a plain auxiliary backend in the target scheduler after the Meta backend is created, without adding the device to `model.devices` or the Meta communicator.
2. Force Meta/aux scheduler crossings through pinned host staging and reject non-MIRRORED Meta tensors.
3. Preserve exact Qwen4Exp routed/shared MoE semantics: the full routed output from ROCm3 is mirrored to Meta ranks; only the tensor-partial shared-expert branch is AllReduced before the final add.
4. Fail closed on mixed routed-expert placement, shared/router tensors on aux, an aux device already in Meta, unsupported architectures/split modes, or unsupported split states.
5. Sweep 0/2/4/6 auxiliary layers at ub 512/1024/2048 and compare correctness, throughput, copy cost, and ROCm3 memory with the CPU tier.

## Detailed Solution & Technical Design

Patch `1328_aux_rocm_expert_backend` adds an ordinary target-context backend selected by `BIGCHERRY_EXPERT_AUX_DEVICE`. Registration is limited to Qwen4Exp `LLAMA_SPLIT_MODE_TENSOR`, resolves the exact backend device name, verifies it is a GPU and not a constituent of the target Meta device, then initializes it as a standalone backend. Because it is not inserted into `model.devices`, Meta constructs its communicator only from ROCm0/1/2; ROCm3 cannot participate in RCCL or patch 1291's Meta AllReduce.

The scheduler continues to place weighted operations by the buffer that owns their weight, so routed `MUL_MAT_ID` operations whose expert tensors are ordinary ROCm3 buffers execute on ROCm3. Cross-backend inputs between Meta and the named aux backend bypass the normal backend-to-backend async copy path and use one scheduler-owned pinned host bounce buffer. The Meta side must be `MIRRORED`; any partial/sliced crossing aborts with a specific error. This keeps patch 1326's host-input optimization independent: 1326 handles graph inputs before the inter-split path used by 1328.

Qwen4Exp validates each selected routed-expert layer before graph reserve. Every present routed expert weight/scale tensor must be on the aux device, while router and shared-expert tensors must remain on Meta. The routed MoE result is therefore a complete value, not a tensor-parallel partial. The routed/shared final `ADD` receives a marker consumed by the Meta split-state logic. That marker is valid only for exactly one `MIRRORED` source plus one `PARTIAL` source; Meta creates a reduction boundary for the shared partial branch, then performs the marked add as a mirrored result. The full aux result is never AllReduced.

The target aux backend keeps its own HIP graph cache, so decode-shaped aux subgraphs remain graph-capturable. ROCm3 is also used by the draft context, so target aux weights, target aux compute buffers/graph executables, and draft allocations naturally compete for its 16 GiB; registration logs current free/total memory and allocation/reserve failures remain hard failures. The staging buffer is pinned host memory, not ROCm3 VRAM.

## Code Samples & Guidance

Production-shaped invocation:

```text
BIGCHERRY_EXPERT_AUX_DEVICE=ROCm3 \
  llama-server -dev ROCm0,ROCm1,ROCm2 -sm tensor -devd ROCm3 \
  -ot 'blk\.(20|44)\.ffn_.*_exps\.weight=ROCm3' ...
```

Do not add ROCm3 to `-dev`/tensor split. The aux device is intentionally outside that list.

## Files

- `patches/1328_aux_rocm_expert_backend/{patch.toml,patch.py,SUMMARY.md}`
- `tools/tests/patch/test_1328_aux_rocm_expert_backend.py`
- `config/recipes.toml` experiment `deploy-v5-plus-1327-aux6900`
- `tools/lab/flash-next/queue-expert-aux6900.sh`

## Validation

Offline mechanics: apply, idempotence, missing-anchor fail-closed, and composition after every `deploy-v5-plus-1327` patch that touches the same source files.

Hardware queue: layer sets `none`, `20|44`, `8|20|32|44`, `4|12|20|28|36|44` at ub 512/1024/2048 with `DEST=ROCm3`. Required runtime evidence before promotion: trace that ROCm3 is not in Meta/AllReduce membership, greedy/KLD parity, no graph-capture regression, per-tier ROCm3 VRAM headroom, and throughput/copy-latency comparison against the same CPU-offloaded layers.

## Effort & Risk

High correctness risk is isolated to the routed/shared reduction boundary. The implementation deliberately supports only the observed Qwen4Exp shared-expert form and whole routed-expert layer placement; it rejects broader mixed cases rather than inferring semantics.

## Standards

Patch-package conventions from `PATCH_SYSTEM.md` / `PATCH_AUTHORING.md`: anchored edits, explicit guards and match counts, off by default, deterministic idempotence, no line-number patches, and untested state until hardware evidence exists.

## Acceptance Criteria

- Target graph reserve accepts selected `-ot ...=ROCm3` routed-expert layers with ROCm3 absent from `-dev`.
- ROCm3 is an ordinary scheduler backend and is never a Meta communicator rank.
- Meta<->ROCm3 activation transfers use pinned host staging, not P2P.
- Full aux routed output is mirrored and not AllReduced; shared-expert partial output is reduced exactly once before merge.
- Unsupported/mixed placements fail with a clear error.
- Decode graph capture remains enabled on target Meta and aux ROCm3 subgraphs.
- Offline mechanics tests and `patch-lint` pass before hardware execution.

## Notes

The 6900's value is absorbing many individually cold experts so CPU route mass stays <= 0.25-0.5%.

Implemented in the new `1328_aux_rocm_expert_backend` package with the exact v5+1327+1328 recipe, focused offline composition tests, and the ROCm3 12-cell layer/ub sweep queue. Hardware qualification remains pending; no performance/correctness promotion claim is made by this commit.

2026-10-05, folded in from BCOP18 (audit backfill) - topology-aware 6900 XT qualification: separate resident-expert activation traffic from dynamic expert-weight transfer; the PCH-attached 6900 XT (8 GT/s x4) may work when weights stay resident even if it is poor as a miss-driven cache tier. (1) benchmark 4/10/32/128 KiB host-staged and any direct HIP peer transfers, independent of RCCL capability; (2) persistent pinned buffers, event-driven staging, ping/pong buffering, no per-token allocation or global sync; (3) score placement as route probability x (transfer in/out + expert compute + queue delay); (4) ROCm3 stays outside the primary Meta/RCCL collective group; (5) whole-layer auxiliary execution (the 1328 sweep, not yet run) before any expert-granular placement. Placement is decided on measured service cost, not nominal PCIe bandwidth.

## Change Log

- 2026-10-02T04:45:07.116948+00:00 (created-by): Created by agent
- 2026-10-04: Implemented patch 1328 design/package, recipe, mechanics tests, and ROCm3 sweep queue; retained pending state until hardware validation.
- 2026-10-05: Corrected the plan heading to the implemented patch id 1328; hardware qualification remains pending.
- 2026-10-05T04:54:53.020468+00:00 (updated-by): Updated: section:notes
