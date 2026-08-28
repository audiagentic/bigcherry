# Prework pack review — deltas against HI01–HI16

Review of `llama_hip_autotune_prework` (`LLAMA_HIP_AUTOTUNE_PLAN.md`,
`PATCH_MAP.md`, `BUILD_PROFILES.md`, `SOURCE_AUDIT.md`, `TERMINOLOGY.md`,
`TEST_RESULTS.md`, `INTEGRATION_CHECKLIST.md`, `prototype/`, `schemas/`)
against the HI01–HI16 plans and against what has actually been built.

The HI plans are a faithful decomposition of the pack. What follows is what
they *dropped*, what the pack got wrong, and what building HI01–HI06 against a
real checkout revealed that neither anticipated.

---

## A. Defects found — pack or plans are wrong

### A1. The inventory build cannot record (**fixed**)

`BUILD_PROFILES.md` specifies the inventory build as:

```
-DGGML_HIP_DISPATCH_REPLAY=ON -DGGML_HIP_AUTOTUNE=OFF
```

then runs it with `GGML_HIP_DISPATCH_MODE=record` writing to SQLite.

But `AUTOTUNE=OFF` is also what gates the tuner *and* SQLite. Implemented
literally — as HI02/HI10 describe — the inventory build downgrades record mode
to native, links no SQLite, and **records nothing, silently**. Since the
inventory JSON is what drives workload-max candidate generation, the entire
tuning pipeline would begin from an empty inventory with no error anywhere.

**Resolution:** recording is a third capability, not a subset of tuning.
Added `GGML_HIP_AUTOTUNE_RECORD`; SQLite is gated on
`(AUTOTUNE OR RECORD) AND SQLITE`; the runtime downgrades record and tune
independently, each with its own diagnostic. Production replay-full keeps
`RECORD=OFF` and therefore still links no SQLite (standards 9.1) — misses go to
a bounded log instead.

Affects: HI02, HI10, HI11, HI15.

### A2. MMQ's J range is not `range(8, 129, 8)`

Standards 13.2, `SOURCE_AUDIT.md` ("J values 8–128 in steps of 8 are considered
at runtime") and HI06 all describe the J space as the full sixteen values. That
is true of the *switch*, but not of the *config table*, which is sparse and
unevenly so:

| table | rows | `q8_0` direct J |
| --- | --- | --- |
| `mmq-config-rdna3.cuh` | 260 | 16, 32, 48, 64, 80, 96, 112, 128 |
| `mmq-config-rdna4.cuh` | 260 | 16, 32, 48, 64, 80, 96, 112, 128 |
| `mmq-config-cdna.cuh` | **154** | **16, 32, 48, 64** |
| `mmq-config-rdna2.cuh` | 242 | — |

`ggml_cuda_mmq_get_config` returns `GGML_TYPE_COUNT` for undefined
combinations and the native scan skips them. Enumerating all sixteen would
manufacture candidates that abort inside `launch_mul_mat_q`. On CDNA that is
two thirds of the `q8_0` space.

**Resolution:** the catalog is derived from the `CASE` tables, never from the
range. `artifacts/<rev>/mmq-table-coverage.json` records what each table
defined so the change between releases is reviewable.

Affects: HI03, HI06, and standards §13.2 (which should say the *switch* covers
`range(8, 129, 8)`, and that the dispatchable set is the config table).

### A3. `dispatch_api_sketch.cuh` has no launch context

The sketch types `ggml_hip_launch_fn` as `void (*)(void * opaque)`. An opaque
pointer defers the question of what a candidate actually needs, and every
implementation of it would have to cast blindly. The real requirement is
concrete and small: `ctx`, `src0`, `src1`, `ids`, `dst`, fusion args, stream —
exactly what upstream's family entry points already take.

**Resolution:** `ggml_hip_launch_context` is a real struct, so a
`native_wrapper` candidate is a direct forward with no marshalling and no cast.

### A4. `GGML_HIP_EXPORT_METRICS` already exists upstream

`BUILD_PROFILES.md` presents the resource-metrics build as using a proposed
option. It is already in `ggml/CMakeLists.txt` (line 220) and already sets
`-Rpass-analysis=kernel-resource-usage --save-temps`. No patch needed — HI15
should consume its output rather than add the flag.

---

## B. Pack content the HI plans dropped

### B1. MMVQ `small_k` and fusion are separate candidate dimensions

`prototype/autotune_manifest.py` generates MMVQ candidates across **four**
dimensions plus two extra paths:

- `small_k` — a distinct path where `rows_per_block = nwarps` rather than 1 or
  2. Upstream's `calc_rows_per_block(ncols_dst, table_id, small_k, nwarps)`
  takes `small_k` as a parameter, so this is real, not speculative.
- `has_fusion` — the prototype emits `f0`/`f1` candidates. **I was wrong to
  carry this over; it is not a candidate dimension.** Fusion is selected at
  *runtime* inside `mul_mat_vec_q_switch_fusion`:

  ```cpp
  const bool has_fusion = fusion.gate != nullptr || ...;
  if constexpr (c_ncols_dst == 1) {
      if (has_fusion) { launch mul_mat_vec_q<..., true,  ...>; return; }
  }
                        launch mul_mat_vec_q<..., false, ...>;
  ```

  One compiled instance serves both, so a pair of candidates differing only in
  fusion would name the same code and double the measurement work for nothing.
  Fusion belongs on the *operation* side of the key: standards 11.1 makes a
  fused pattern a different semantic operation tuned within its already-selected
  family, and `ggml_hip_dispatch_signature_v1.fusion` already carries it. The
  compiler settled it — including fusion in the instance name produced duplicate
  symbols, because the generated names collided on identical geometry.

**Resolution (implemented):** `small_k` added to the geometry matrix and the
stable name (`mmvq:q8_0:w1:nw2:rpb2:sk1:v1`); fusion deliberately omitted.
Emitted only for `nwarps > 1`, since at `nwarps == 1` the small-K geometry *is*
the ordinary geometry and would duplicate a candidate.

### B2. Screening retention policy

Plan §11.4 specifies exactly what survives screening:

> Retain: native; top 3 candidates; every candidate within 10% of best median.

HI12 says "warmup/screen/finalist stages" without the retention rule. Without
it, "screening" is undefined — the number of finalists drives total tune time
more than any other parameter.

**Action:** HI12 gains explicit `screen_keep_top=3`, `screen_keep_within_pct=10`,
and native always retained.

### B3. Measurement fields the schema needs but the plans omit

`schemas/dispatch-db.sql` records `gpu_mad_us` (median absolute deviation) and
`host_median_us`. My schema has `stddev_us` and `p95_us` but neither of these.

- **MAD** is the right dispersion measure for GPU timings — they are
  right-skewed and MAD is far less sensitive to a stray outlier than stddev.
- **host time** separates launch-overhead-bound candidates from
  kernel-bound ones. For tiny kernels — precisely where MMVQ geometry matters
  — a candidate can win on GPU time and lose on wall clock.

**Action:** add both to `sql/dispatch-db.sql` and to HI12.

Also missing: `winner.reason` (a text explanation of why the candidate won) and
`build.compiler` / `build.dispatch_abi`.

### B4. Five collection points, not two

Plan §9.1 lists five required sites:

1. dense `ggml_cuda_mul_mat` — **done**
2. `ggml_cuda_mul_mat_id` — **done**
3. fused MMVQ graph paths calling `ggml_cuda_mul_mat_vec_q` directly
4. fused MMVF graph paths calling `ggml_cuda_mul_mat_vec_f` directly
5. public MMQ/MMF family entry points, for lower-level variant capture

HI13 covers 3 and 4. **Nothing covers 5.** Upstream has ~8 call sites of
`ggml_cuda_should_fuse_mul_mat_vec_{f,q}` in the graph optimiser, so paths
3–5 are not a rare corner.

**Action:** HI13 gains site 5 explicitly.

### B5. Bounded-width array cache

Plan §8.6: *"For bounded dynamic width, use an array cache indexed by width
after the first lookup."*

The current resolver is a hash map on the dispatch digest for every width. For
MoE, where width varies per token within one operation, an array indexed by
width (bounded at `MMVQ_MAX_BATCH_SIZE = 8`) turns a hash lookup into an array
index on the hottest path in the system.

**Action:** HI11.

### B6. Resource-based blacklist must precede tuning

Plan §17.2: compile full-max as an audit, *inspect compiler resource output*,
and blacklist high-spill geometries **before** runtime tuning.

The blacklist table exists, but nothing parses
`-Rpass-analysis=kernel-resource-usage`. HI15 lists the blacklist as production
hardening — that is too late. A spilling MMVQ geometry that reaches the tuner
wastes measurement time on a candidate that was never viable.

**Action:** new work item between HI09 and HI12 — parse the resource report,
emit the blacklist, feed it to the catalog. Sequenced right after the full-max
compile audit that produces the report.

### B7. Two standalone tools

`PATCH_MAP.md` Patches 10 and 11 specify:

- `tools/hip-autotune-export.cpp` — SQLite winners → compact binary cache
- `tools/hip-autotune-inspect.cpp` — inspect a binary/registry

Neither appears in HI15/HI16 as a deliverable. Export in particular is not
optional: it is the only bridge from tune builds to production.

**Note:** export can be Python (it reads SQLite, writes a binary file, and runs
offline). Inspect should be C++ so it links the real registry and validates the
cache with the same code production uses — a Python re-implementation could
disagree with the loader, which is the one thing it exists to check.

### B8. Restart-only cache reload

Plan §12.5: *"If a replay cache changes, restart initially. Do not implement
live cache reload until graph and binding invalidation are proven."*

An explicit non-goal worth recording, because live reload is exactly the
feature someone adds later without realising that captured graphs hold the
resolved candidate.

**Action:** state as a non-goal in HI11 and HI15.

### B9. Divisibility class as a refinement

Plan §8.4 lists "selected divisibility classes" among refinements. The
signature has `alignment_class`, `occupancy_bucket`, `offset_modulo` but no
divisibility class. MMQ's `fallback` already depends on `ne01 % 128`, which is
captured as hard identity — but other divisibility relationships (K against
`QK8_1`, N against tile width) are not.

**Action:** HI05 refinement set, low priority — refinements are opt-in and
unpromoted by default.

### B10. Variant-set naming disagrees

`BUILD_PROFILES.md` uses `NATIVE`, `WORKLOAD_MAX`, `FULL_MAX` (uppercase);
HI02/HI03 and the implementation use `inventory`, `workload-max`, `full-max`.

The lowercase names are better — `inventory` says what the build is *for*,
where `NATIVE` collides with native selection, native mode and native
candidate, three things that already mean something specific.

**Action:** keep lowercase; have CMake accept the uppercase spellings as
aliases so the pack's documented commands work verbatim.

---

## C. Findings from building against a real checkout

Neither the pack nor the plans could have these — the pack explicitly notes no
HIP compiler or GPU was available.

### C1. `ggml_cuda_mul_mat_cublas` is `static`

The BLAS candidate cannot call it. Resolved with a non-static forwarder
carrying its own forward declaration, so upstream's linkage is untouched.

### C2. Overlay files compile into *every* HIP build

`ggml-hip/CMakeLists.txt` globs `../ggml-cuda/*.cu`. Guarding on
`GGML_USE_HIP` alone — as standards §12.2 could be read to allow — puts the
dispatch layer into builds carrying none of its dependencies. Both conditions
are required. The compiler caught this.

### C3. Upstream has drifted from the audit target

Audit target was `0ef6e55`; current master is `22dc605`. All 32 audited
invariants still hold, but `ggml_cuda_mul_mat` now has a `GGML_HINT_SRC0_IS_HADAMARD`
early-out ahead of the family ladder, and MMQ has gained `Q1_0`, `Q2_0` and
`NVFP4` (22 types, not the pack's assumed set).

### C4. The audit must survive its own patches

Once HI06 lands, the J switch has moved and a re-run of the audit reports a
false failure. The audit now looks for each construct in its post-patch home
first and reports which state it found. Worth generalising as HI07–HI09 land.

### C5. Transposed-vector MMVF path

`ggml_cuda_mul_mat` has a path that swaps `src0`/`src1` and synthesises a `dst`.
Its launch context is not the one the signature describes. Dispatch currently
declines it, which is safe but means the shape is never tuned. Worth a decision
in HI13 rather than leaving it implicit.

---

## D. Recommended sequence change

The pack's Phase order (plan §14) differs from the HI order, and the pack's is
better in one respect: it puts **manual replay (Phase 3) before the tuner
(Phase 4)**, so the whole resolve/bind/fallback path is proven with
hand-written winners before any measurement code exists. HI11 and HI12 are in
that order already — worth stating *why*, so it is not reordered for
convenience.

Proposed insertion, from B6:

```
HI09  MMVQ explicit geometry variants
HI09b resource-report parsing and candidate blacklist   <-- new
HI10  record mode and SQLite
HI11  replay cache and manual seeding
HI12  tuning engine
```

---

## E. Summary of actions

| Item | Where | Status |
| --- | --- | --- |
| A1 record capability split | HI02, HI10 | **fixed** |
| A2 sparse MMQ config tables | HI03, HI06, standards 13.2 | **fixed** |
| A3 concrete launch context | HI04 | **fixed** |
| A4 EXPORT_METRICS exists upstream | HI15 | note added |
| B1 MMVQ `small_k` + fusion | HI09 | **open — changes stable names** |
| B2 screening retention policy | HI12 | open |
| B3 MAD, host time, winner reason | HI12, sql | open |
| B4 collection point 5 | HI13 | open |
| B5 bounded-width array cache | HI11 | open |
| B6 resource blacklist before tuning | new HI09b | open |
| B7 export and inspect tools | HI15, HI16 | open |
| B8 restart-only reload | HI11, HI15 | open |
| B9 divisibility refinement | HI05 | open, low priority |
| B10 variant-set aliases | HI02 | open |
| C1–C5 real-checkout findings | various | C1, C2 fixed; C3–C5 noted |
