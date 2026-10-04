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

Prefill may improve more by executing fewer reductions than by only accelerating each reduction. PGC16 owns elimination/folding of redundant SUM reductions where tensor-partial branches can be combined locally before one equivalent AllReduce.

The algebraic identity is:

```text
AR(a) + AR(b) == AR(a + b)
```

only when `a` and `b` are partial values over the same rank group and logical elements. A MIRRORED value cannot be moved before AR because it would be added once per rank. Nonlinear ops, different split groups/axes and live consumers of the individual reduced values block the rewrite.

At llama.cpp `0504396`, Meta split semantics are already represented by `ggml_backend_meta_split_state { axis, nr[16], n_segments, ... }`; use that authoritative state rather than tensor-name heuristics.

## Steps

1. Extend existing AR tracing with producer/consumer topology and split-state summaries around every large prefill reduction.
2. Add a read-only partial-state analyzer with `UNKNOWN`, `MIRRORED`, `PARTIAL_SUM` and strict propagation rules.
3. Dry-run canonical candidates and compute an optimistic bound from measured AR wall before changing a graph.
4. Implement exactly one proven topology first: two compatible reduced branches immediately joined by ADD and no other consumer -> local ADD before one AR.
5. Preserve the existing Meta provider; the optimization changes graph/subgraph structure, not collective transport.
6. Do not move norm, activation, softmax, GLU/gating, routing or other nonlinear operations across a reduction.
7. Treat routed/shared-expert joins carefully: only two PARTIAL branches with identical rank membership can be folded. MET05 auxiliary/mirrored branches are explicitly blocked.
8. Validate changed floating-point summation order with KLD/top-token gates.
9. Generalize into one graph pass only after two profitable real topologies exist.
10. Compose independently with PGC15.

## Detailed Solution & Technical Design

### Partial contract

Source-shaped descriptor:

```cpp
enum bc_partial_kind : uint8_t {
    BC_PARTIAL_UNKNOWN = 0,
    BC_PARTIAL_MIRRORED,
    BC_PARTIAL_SUM,
};

struct bc_partial_contract {
    bc_partial_kind kind = BC_PARTIAL_UNKNOWN;
    ggml_backend_meta_split_axis axis = GGML_BACKEND_SPLIT_AXIS_UNKNOWN;
    uint32_t nr[16] = {};
    uint32_t n_segments = 0;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
};
```

Build it from the real split state:

```cpp
static bc_partial_contract bc_contract(const ggml_tensor * t) {
    bc_partial_contract c;
    if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_meta(t->buffer)) {
        return c;
    }

    const ggml_backend_meta_split_state ss =
        ggml_backend_meta_get_split_state(t, /* assume_sync = */ false);

    c.axis       = ss.axis;
    c.n_segments = ss.n_segments;
    memcpy(c.nr, ss.nr, sizeof(c.nr));
    c.type = t->type;
    memcpy(c.ne, t->ne, sizeof(c.ne));

    if (ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        c.kind = BC_PARTIAL_MIRRORED;
    } else if (ss.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        c.kind = BC_PARTIAL_SUM;
    }
    return c;
}
```

`ggml_backend_meta_get_split_state()` is file-local in `ggml-backend-meta.cpp` at this pin. Keep the first matcher there or expose a private helper in `ggml-backend-impl.h`; do not add a public API for an experiment.

### Contract equality

Do not compare only axis/shape:

```cpp
static bool bc_same_partial_sum(
        const bc_partial_contract & a,
        const bc_partial_contract & b) {
    if (a.kind != BC_PARTIAL_SUM || b.kind != BC_PARTIAL_SUM ||
        a.axis != b.axis || a.n_segments != b.n_segments || a.type != b.type) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (a.ne[d] != b.ne[d]) {
            return false;
        }
    }
    for (size_t i = 0; i < 16; ++i) {
        if (a.nr[i] != b.nr[i]) {
            return false;
        }
    }
    return true;
}
```

If source inspection shows `nr[]` is insufficient to identify rank ownership, extend the descriptor with the actual Meta segment/rank map from the simple tensor container. Fail closed until that map is available.

### Explicit propagation only

First-version propagation code should be intentionally restrictive:

```cpp
static bc_partial_contract bc_propagate_binary(
        const ggml_tensor * op,
        const bc_partial_contract & a,
        const bc_partial_contract & b) {
    if (op->op != GGML_OP_ADD || !bc_same_partial_sum(a, b)) {
        return {};
    }
    bc_partial_contract out = a;
    out.type = op->type;
    memcpy(out.ne, op->ne, sizeof(out.ne));
    return out;
}
```

Do not initially propagate PARTIAL through MUL except a separately proven replicated-scalar case.

VIEW/RESHAPE can propagate only for no-reorder views:

```cpp
static bool bc_view_preserves_elements(const ggml_tensor * src, const ggml_tensor * dst) {
    return ggml_nelements(src) == ggml_nelements(dst) &&
           ggml_nbytes(src)    == ggml_nbytes(dst) &&
           ggml_is_contiguous(src) && ggml_is_contiguous(dst) &&
           src->data == dst->data;
}
```

Anything else => UNKNOWN.

### Candidate record: dry-run first

AllReduce is not a first-class GGML op in this Meta path; it is inserted at subgraph boundaries. Record each boundary and its pre-reduction tensor:

```cpp
struct bc_ar_boundary {
    size_t subgraph;
    ggml_tensor * partial[GGML_BACKEND_META_MAX_DEVICES];
    ggml_tensor * logical;
    bc_partial_contract contract;
    size_t bytes;
};
```

At the existing block that builds `nodes` and calls `backend_ctx->comm_allreduce(...)`, emit:

```cpp
if (bc_ar_fold_trace) {
    const auto c = bc_contract(nodes[0]);
    fprintf(stderr,
        "BIGCHERRY_AR_BOUNDARY sg=%zu name=%s axis=%s seg=%u bytes=%zu\n",
        i,
        nodes[0]->name,
        ggml_backend_meta_split_axis_name(c.axis),
        c.n_segments,
        ggml_nbytes(nodes[0]));
}
```

Augment this with logical producer/consumer ids from the split builder. The dry-run tool reports only candidates whose two reduced outputs feed one ADD and have no other consumer.

### First rewrite: before Meta split construction

Do **not** fake two boundaries inside `ggml_backend_meta_graph_compute()`. Rewrite the logical graph before scheduler/Meta split construction so one local ADD is naturally present before one PARTIAL boundary.

Minimal matcher shape:

```cpp
static bool bc_can_fold_two_reductions(
        ggml_tensor * join,
        ggml_tensor * a_partial,
        ggml_tensor * b_partial,
        int n_consumers_a,
        int n_consumers_b) {
    if (join == nullptr || join->op != GGML_OP_ADD ||
        n_consumers_a != 1 || n_consumers_b != 1) {
        return false;
    }
    return bc_same_partial_sum(bc_contract(a_partial), bc_contract(b_partial));
}
```

When the exact candidate is identified, change the builder topology from the conceptual:

```cpp
ra  = reduced_partial_a;
rb  = reduced_partial_b;
out = ggml_add(ctx0, ra, rb);
```

to:

```cpp
local = ggml_add(ctx0, a_partial, b_partial); // remains PARTIAL_SUM
out   = local;                                // Meta inserts one reduction after the join
```

There is no public `reduce_partial()` constructor today. The real patch must arrange placement/split state so Meta sees a single PARTIAL boundary after `local`; trace proof determines the exact builder seam before anchors are written.

### Debug assertion

For the first candidate, assert its contract at the earliest point where split state exists:

```cpp
#ifndef NDEBUG
{
    const auto ca = bc_contract(a_partial);
    const auto cb = bc_contract(b_partial);
    GGML_ASSERT(bc_same_partial_sum(ca, cb));
}
#endif
```

If graph construction precedes buffer/split-state assignment, move this assertion into the Meta split-assignment phase and identify nodes by stable graph identity. Do not silently accept a failed contract after selecting the experimental path.

### Floating-point order

Original:

```text
(a0+a1+a2) + (b0+b1+b2)
```

folded:

```text
(a0+b0) + (a1+b1) + (a2+b2)
```

is mathematically equal but not bit-identical. This is an intentional summation-order change; use exact synthetic tests for indexing/algebra and model KLD/top-token gates for production.

## Code Samples & Guidance

Implementation sequence:

```text
A. trace only in ggml-backend-meta.cpp
B. offline topology tool identifies exact candidate + consumer counts
C. add bc_partial_contract helpers + negative fixtures
D. rewrite one graph topology before Meta split construction
E. assert one fewer comm_allreduce call for that topology
F. only after a measured win, factor matcher into a generic optimization
```

Do not start at F.

Diagnostics:

```text
BIGCHERRY_AR_FOLD candidate layer=37 join=ADD a=... b=... bytes=... safe=1
BIGCHERRY_AR_FOLD applied layer=37 ar_before=2 ar_after=1
BIGCHERRY_AR_FOLD reject layer=... reason=mirrored|rank_map|extra_consumer|nonlinear
```

## Files

- `ggml/src/ggml-backend-meta.cpp`: authoritative split-state helper, boundary trace/assertions.
- Exact model/graph builder containing the first proven topology, likely `src/models/qwen4exp.cpp` only after trace proof.
- Existing AR telemetry + `tools/lab/flash-next/ar-segment.py`.
- Generic optimizer only after multiple real candidates prove the contract.

## Validation

Fixtures:

```text
PARTIAL G + PARTIAL G -> eligible
PARTIAL G1 + PARTIAL G2 -> reject
PARTIAL + MIRRORED -> reject
extra consumer -> reject
nonlinear between branch and join -> reject
view that changes element order -> reject
```

Hardware: count reductions/bytes before/after at pp1024/2048/4096 and Flash-Next 10K/80K/200K. Record collective wall, prefill t/s, peak memory and numerical contract. Decode unchanged unless separately qualified.

## Effort & Risk

L/high semantic risk. Incorrect PARTIAL/MIRRORED classification can silently multiply values by rank count. First implementation must be narrow and assertion-heavy.

## Standards

Split state authoritative; UNKNOWN fails closed; no model-name heuristic for generic algebra; dry-run Amdahl proof first; no transport changes here.

## Acceptance Criteria

- Dry-run finds a real candidate with >=3% E2E optimistic bound or >=5% of prefill wall.
- First rewrite removes exactly one expected AllReduce without adding another.
- Negative fixtures make PARTIAL+MIRRORED/mismatched groups impossible.
- Representative prefill improves >=3% or collective critical wall >=10%, accepted numerical contract, decode <=1% regression.
- If no candidate meets the bound, park with census.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent after prefill scan.
- 2026-10-05: Added source-shaped split-state analyzer/matcher guidance grounded in llama.cpp 0504396 Meta semantics.
