---
id: PGC16
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Prefill AllReduce count reduction: partial-value algebra and join fusion before collectives

## Description

BigCherry has spent substantial effort making each AllReduce faster, but prefill can also improve by executing fewer collectives. This item owns graph-level elimination/folding of redundant SUM reductions when multiple tensor-partial branches can be combined locally before one equivalent AllReduce.

Core identity for equal rank groups and equal elementwise shapes:

```text
AR(a) + AR(b) == AR(a + b)
```

More generally, linear operations that preserve tensor-partial semantics may move before a SUM AllReduce. Nonlinear operations, mismatched rank groups, mirrored values, different split axes and shape/layout changes block the rewrite unless separately proven.

This is not a generic compiler algebra project. Start from the real Flash-Next prefill AR census and implement only high-frequency/high-byte join topologies with an Amdahl bound >=3%. QFP17 owns model-level prefill acceptance; PGC16 owns partial-value semantics and collective-count reduction. PKC02 may later own reusable recipe/replay metadata once the first rewrite is proven.

## Steps

1. Extend AR telemetry to emit producer/consumer topology around every prefill collective: `{node, layer, dtype/shape, split state, rank set, producer op, sibling producer(s), immediate consumer op, next collective distance, bytes}`. Rank by total prefill collective wall and count per layer.
2. Implement a read-only analyzer that annotates graph values with `MIRRORED`, `PARTIAL(group, axis, reduction=SUM)`, or `UNKNOWN`. Propagate only through explicitly safe linear ops; unknown is fail closed.
3. Find canonical `ADD(AR(a), AR(b))` and equivalent branch-join topologies. Require identical logical shape/dtype/rank membership and that neither reduced value has another live consumer requiring the individually reduced form.
4. Rewrite the first proven topology to `AR(ADD(a,b))`. Reuse the existing Meta AllReduce node/provider; do not add a fused collective implementation yet. Confirm AR count decreases exactly as predicted.
5. Add guarded linear epilogue movement only where useful: replicated bias or elementwise scale may remain after AR; local sums of multiple PARTIAL values may move before AR. Do not move activation/norm/softmax/GLU/nonlinear gates across AR.
6. Audit Qwen4Exp routed/shared-expert joins carefully. A routed partial branch plus a shared partial branch with the same rank group may be locally added before one AR. An auxiliary/mirrored routed branch from MET05 is not equivalent and must remain outside the partial reduction. Encode this as a semantic rule, not a model-name special case.
7. Search attention/output-projection and hyper-connection joins for the same pattern. Only implement additional rewrites if the telemetry proves separate ARs currently exist; do not infer from source graph intent.
8. Measure numerical impact from changed local summation order. Exact algebra does not imply bit identity in f32/bf16 floating point. Use the existing KLD/top-token gates and preserve a disable flag until qualified.
9. If the first two rewrites are profitable, factor the analyzer/matcher into one Meta graph optimization pass with stable diagnostics. Otherwise keep the implementation local and park the genericization.
10. Compose with PGC15 after independent validation: fewer ARs reduce service demand; tiled overlap hides remaining ARs. Their correctness mechanisms must remain separable.

## Detailed Solution & Technical Design

### Partial-state lattice

Do not infer reducibility only from tensor names. The Meta backend already tracks split state. Build a narrow semantic descriptor from that authoritative state:

```cpp
enum bc_value_kind { BC_UNKNOWN, BC_MIRRORED, BC_PARTIAL };
struct bc_partial_value {
    bc_value_kind kind;
    uint64_t rank_mask;
    int split_axis;
    enum ggml_type type;
    int64_t ne[GGML_MAX_DIMS];
};
```

For the first version, only these propagation rules are permitted:

```text
ADD(PARTIAL G, PARTIAL G) -> PARTIAL G
ADD(PARTIAL G, MIRRORED)  -> blocked
MUL(PARTIAL G, replicated scalar) -> PARTIAL G, only if exact graph representation proves scalar replication
VIEW/RESHAPE(PARTIAL G) -> PARTIAL G only when no element reordering and every rank has identical view mapping
AR(PARTIAL G) -> MIRRORED
```

`G` includes rank membership, split semantics and reduction kind. `PARTIAL G1 + PARTIAL G2` is blocked unless `G1 == G2`. Do not treat `MIRRORED` as a partial contribution: adding it on every rank before AR multiplies it by rank count.

### Rewrite safety

Canonical candidate:

```text
p0 ---- AR ---- r0 --+
                      ADD -> out
p1 ---- AR ---- r1 --+
```

becomes:

```text
p0 --+
     ADD_LOCAL -> p01 -> AR -> out
p1 --+
```

Required checks:
- both AR nodes are SUM with the same communicator/provider/rank set;
- `p0/p1` have the same logical shape/dtype and compatible strides;
- `r0/r1` have no other consumers that require separately reduced values;
- local ADD is valid on every rank, including ranks where the Meta scheduler skipped a producer: skipped partials must be explicit zero contributions exactly as current AR semantics require;
- graph ordering does not introduce a dependency cycle or extend a large tensor lifetime enough to erase the win.

Do not combine two ARs merely because they have the same byte count.

### Graph matcher

Prefer a pre-execution graph rewrite in the Meta/backend optimization layer where split-state metadata is available. If the upstream graph optimizer cannot represent Meta split semantics safely, first implement a narrow pattern inside `ggml-backend-meta.cpp` when building/executing subgraphs: recognize the specific pair of consecutive reduction outputs and substitute one local pre-reduce join. Keep the original graph path available for all unmatched cases.

Pseudo-matcher:

```cpp
if (is_add(join) &&
    is_meta_allreduce(join->src[0]) &&
    is_meta_allreduce(join->src[1]) &&
    same_partial_contract(pre_ar0, pre_ar1) &&
    single_consumer(ar0) && single_consumer(ar1)) {
    // build or select local ADD of pre-AR values, then one AR
}
```

Exact graph representation at pin 0504396 must be inspected before implementation; AllReduce may be represented by Meta split boundaries rather than a first-class GGML op. In that case the matcher operates on subgraph boundary metadata rather than literal nodes.

### Floating-point behavior

Original rank/order arithmetic may be `(a0+a1+a2) + (b0+b1+b2)`. Fused form may reduce `(a0+b0) + (a1+b1) + (a2+b2)`. They are mathematically equal but not bit-identical. For f32 expect small roundoff differences; for bf16 wire the provider already introduces quantization. Therefore require KLD/top-token/greedy-reference evidence and record that this is a summation-order change. Do not label divergence automatically as a bug unless it exceeds the established contract.

## Code Samples & Guidance

Primary code ownership:

- `ggml/src/ggml-backend-meta.cpp`: split-state and collective boundary information; likely first matcher/diagnostic site.
- Qwen4Exp graph (`src/models/qwen4exp.cpp`) only if one model-local topology must be made explicit; do not encode generic partial algebra there.
- existing 0830/1277/1242 tracing patches and `tools/lab/flash-next/ar-segment.py`: extend for topology/count evidence rather than creating another trace format.
- PKC02 only after the matcher has a stable semantic contract suitable for a reusable graph recipe.

Suggested diagnostic:

```text
BIGCHERRY_AR_FOLD candidate layer=37 join=ADD ar0=6.4MiB ar1=6.4MiB group=0x7 safe=1 reason=single_consumer
BIGCHERRY_AR_FOLD applied layer=37 before=2 after=1 bytes_before=12.8MiB bytes_after=6.4MiB
```

Add a dry-run mode first that reports candidates and an optimistic wall-time bound from measured AR timing without changing the graph. Require the bound before coding a topology.

## Files

`ggml/src/ggml-backend-meta.cpp`; graph optimizer source if split semantics can be carried there safely; existing AR telemetry/tooling; optional Qwen4Exp graph annotation only when necessary; new patch package after one concrete topology is proven.

## Validation

Offline graph fixtures should cover:
- two compatible PARTIAL branches -> one folded AR;
- different rank masks -> no fold;
- PARTIAL + MIRRORED -> no fold;
- extra consumer of one AR -> no fold;
- nonlinear consumer between partial and AR -> no fold;
- skipped/zero rank contribution -> same logical result.

Hardware: count AR calls and bytes before/after, pp1024/2048/4096, Flash-Next 10K/80K/200K. Compare critical-rank collective wall, prefill t/s, peak memory and output contract. Decode control must remain unchanged unless an explicitly separate decode topology is qualified.

## Effort & Risk

L. Semantic risk is higher than a kernel micro-optimization because an incorrect partial/mirrored classification can silently scale a branch by rank count. The first implementation must therefore be extremely narrow with explicit assertions and dry-run evidence. Secondary risk is changed floating-point summation order and longer partial-tensor lifetime.

## Standards

Split state is authoritative; one semantic analyzer; fail closed on UNKNOWN; no model-name heuristics for generic algebra; dry-run Amdahl evidence before rewrite; output contract acknowledges changed summation order; no duplicate collective provider.

## Acceptance Criteria

- Dry-run identifies at least one real prefill topology whose measured optimistic bound is >=3% E2E or >=5% of prefill wall.
- First rewrite reduces the expected collective count/bytes with exact graph instrumentation and no extra collective elsewhere.
- No PARTIAL+MIRRORED or mismatched-rank fold is possible through the matcher; unit fixtures cover these failures.
- Representative prefill improves >=3% or collective critical-path wall falls >=10%, with accepted KLD/top-token/greedy-reference results and decode regression <=1%.
- If no real topology meets the bound, close/park PGC16 with the census rather than inventing a synthetic fusion target.

## Notes

This item is deliberately separate from 1314 small-AR produce/consume launch fusion, which was neutral because it did not remove the synchronization service. PGC16 removes whole collective boundaries when algebra permits it. It also differs from QFP13 launch fusion: the target is prefill collective count, not local elementwise launch count.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent after prefill plan scan; no existing item owned graph-level reduction elision.
