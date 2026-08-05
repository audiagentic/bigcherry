# llama.cpp HIP Autotune — Project Standards

Non-negotiable rules for the HIP measured-dispatch autotuner project.
These supplement (and never contradict) the upstream llama.cpp conventions.

## 1. Terminology — use the canonical terms

The definitions in `TERMINOLOGY.md` are normative and enforced through code reviews.

| Correct term | Not | Why |
| --- | --- | --- |
| Kernel family | kernel type, path | Family = MMQ/MMVF/MMF/MMVQ/BLAS — a major algorithmic path |
| Coverage specialization | quant support, type instantiation | Compile-time requirement to execute a type/bounded shape |
| Performance variant | configuration, option | Semantically equivalent performance choice within a family |
| Candidate | option, kernel, launch | Complete launchable choice = family + coverage spec + perf variant + setup path |
| Native selector / native candidate | default, upstream choice | Upstream policy and its concrete result — always measured, fallback, near-tie preference |
| Signature | key, shape, operation ID | Canonical device-local operation description |
| Hardware key | GPU key, device type | Executing GPU class + features — no runtime ordinal |
| Dispatch key | lookup key | software namespace + hardware key + signature + objective |
| Replay | cached dispatch, lookup | Production resolution of dispatch key to stored winner — no benchmarking |
| Binding | cache hit, resolved | Process-level caching so later executions skip hashing |

## 2. Candidate identity

### 2.1 Stable names are persistent database identities

A stable name (`mmq:q8_0:j64:fb0:v1`) is the durable key across builds, databases and caches. The suffix `v1` encodes the implementation version — it must be incremented when the candidate's compiled behavior changes (e.g., config table update). Runtime numeric IDs are per-build `uint32_t` values only. Never use a runtime ID as a persistent reference.

### 2.2 Stable names encode the full config

For MMQ the stable name contains the complete resolved config (`mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1`), not only `J`. The config table can change independently of `J`.

### 2.3 Source class taxonomy

Every candidate must be classified into exactly one source class:

| Source class | Meaning |
| --- | --- |
| `native_wrapper` | Upstream default path wrapped with stable identity |
| `existing_runtime` | Existing code behind a runtime switch (e.g., MMQ J switch) |
| `existing_alternative` | Existing code not in native path but available (e.g., BLAS when MMQ is native) |
| `new_generated_variant` | Newly generated template instance (e.g., explicit MMVQ geometry) |
| `vendor_auto` | Vendor library auto-select (e.g., hipBLAS default) |
| `vendor_explicit` | Vendor library with explicit algorithm/handle (future hipBLASLt) |

This classification drives build decisions (which need new compilation units vs. which reuse existing code).

### 2.4 Candidate properties: graph_safe and deterministic

Every candidate declares two boolean properties in the manifest:

- `graph_safe` — the candidate can be launched inside a HIP graph capture/update cycle.
- `deterministic` — the candidate produces bitwise-deterministic output for identical inputs.

Both are required fields. If uncertain, mark `graph_safe=false`; the validation phase (HI14) confirms graph safety end-to-end.

### 2.5 One source of truth

The candidate catalog generator produces all downstream artifacts:

- JSON manifest (`hip-autotune-manifest.json`)
- Runtime registry include (`hip-autotune-registry.inc`)
- Generated template instances (`mmvq-autotune-instance-*.cu`)
- Build hash header (`hip-autotune-build-hash.h`)

Never maintain these independently. The catalog is the single authoritative source.

### 2.6 Architecture mask

The manifest lists candidate architectures as an array of strings (`["gfx1100", "gfx1201"]`). At runtime this is compiled into a `uint64_t` architecture mask on the candidate descriptor. A candidate supports a hardware key when `(architecture_mask & (1u << hw.architecture_code))` is nonzero.

This allows a single fat binary to carry candidates for multiple architectures without runtime string comparisons.

## 3. Precision semantics

### 3.1 Never tune across precision boundaries

A candidate that computes in F16 must be measured against an F16 request. A lower-precision candidate (F16 vs F32) is not automatically a performance variant — it is a different operation unless the objective explicitly permits quality alternatives.

### 3.2 MMVF accumulator mode — F16 only

For F16 source type, two accumulator modes exist (F32 and F16) and ARE distinct performance variants within the same family. For other source types (F32, BF16), only F32 accumulator is available.

A candidate with a lower-precision accumulator than the operation request is not automatically a performance variant — it is a different operation unless the objective explicitly permits quality alternatives.

## 4. Selection and dispatch

### 4.1 Separate selection from launch

Native selection returns a descriptor (`ggml_hip_native_selection`), not an immediate launch:

```cpp
auto native = ggml_hip_native_select(ctx, op);
auto bound  = ggml_hip_dispatch_resolve(ctx, signature, native);
ggml_hip_dispatch_launch(bound, launch_context);
```

### 4.2 Native mode parity

In native mode (no tuning/replay), the system must:

- choose the same family as upstream for every signature;
- produce bitwise-identical outputs;
- show no material performance regression.

### 4.3 Force flags are incompatible with dispatch

`GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS` hide legal families from measured dispatch. The build must reject these when dispatch replay or autotuning is enabled.

## 5. Signature construction

### 5.1 Signatures are canonical, not incidental

A signature contains hard identity fields (operation, types, exact device-local M/N/K, ne[]/nb[], layout, broadcast, fusion) and optional refinements (alignment class, offset modulo, occupancy bucket). It must NOT contain diagnostic identity (model name, layer number, pointer values, request ID, device ordinal, clock).

### 5.2 Construct after device slicing

Signatures are built from the actual local tensor slice on the executing GPU. Heterogeneous GPUs and unequal splits receive their own local dimensions — not the global node shape.

### 5.3 Dispatch signature schema version

The dispatch signature struct carries a `schema_version` field (`uint16_t`). This supports ABI evolution: if new fields are added to the signature, the version increments and replay caches from older builds are rejected.

### 5.4 Stable serialization and hashing

Use `blake2b` with 16-byte digest and per-component `person=` prefixes:

- Signature: `person=b"llama-hip-tune"`
- Hardware key: `person=b"llama-hardware"`
- Dispatch key: `person=b"llama-dispatch"`

Canonicalize fields explicitly before hashing — sorted keys in JSON, deterministic separators. Never hash raw padded C++ structs or rely on struct layout for identity.

### 5.5 Refined-to-base fallback

Lookup order: refined key → base key → native fallback. Promote a refinement only when measurements prove it changes the winner materially.

## 6. Build profiles

### 6.1 Never enable both force flags with dispatch

A build with `GGML_HIP_DISPATCH_REPLAY=ON` or `GGML_HIP_AUTOTUNE=ON` must reject `FORCE_MMQ=ON` and `FORCE_CUBLAS=ON`.

### 6.2 Build profile hierarchy

| Profile | Purpose | Candidates |
| --- | --- | --- |
| **Inventory** | Record signatures and native candidates only | Native-only |
| **Workload-max tune** | Tune with all variants for observed types/paths | From inventory JSON |
| **Full-max compile audit** | Compile stress test — discover resource failures | All bounded matrix |
| **Resource metrics** | Inspect finalists and problematic variants | Workload-max + metrics export |
| **Replay-full production** | Production with all candidates, no tuner | Workload-max, replay-only |

Workload-max emits variants only for types and semantic paths observed in inventory — this is the practical "maximal" build. Full-max generates thousands of logical MMVQ candidates and is an audit/stress profile only.

### 6.3 Profiling builds are separate

`GGML_HIP_EXPORT_METRICS=ON` produces diagnostic output not required for normal timing. Keep profiling builds isolated from the primary tuning build.

### 6.4 Runtime environment variables

The dispatch system is controlled at runtime via these environment variables:

| Variable | Values | Purpose |
| --- | --- | --- |
| `GGML_HIP_DISPATCH_MODE` | `record`, `tune`, `replay` | Record: capture signatures. Tune: benchmark candidates. Replay: use stored winners. |
| `GGML_HIP_DISPATCH_DB` | path/to/file.sqlite | SQLite database for record/tune modes. |
| `GGML_HIP_DISPATCH_CACHE` | path/to/replay.bin | Compact binary cache for replay mode. |
| `GGML_HIP_DISPATCH_MISS` | `native`, `native-record` | Miss behavior: fallback to native, optionally recording the miss. |

## 7. Tuning algorithm

### 7.1 Complete-path timing — not terminal-kernel only

Time the complete candidate path including:

- activation conversion/quantization;
- per-call workspace preparation;
- main kernel;
- reductions/copies;
- required output conversion.

MMQ must include activation quantization and related setup. Separate one-time persistent setup.

### 7.2 Native reference for correctness

Always use the native candidate's output as the live correctness reference. Reject candidates that fail:

- launch failure;
- NaN/Inf introduced;
- tolerance failure (NMSE, max absolute/relative error);
- illegal workspace/resource use;
- non-deterministic or unstable behavior beyond policy.

### 7.3 Replacement threshold and winner selection

A non-native candidate replaces native only when:

- improvement ≥ 1% on median GPU time;
- statistical confidence supports a real improvement.

**Near-tie resolution:** collect all candidates within `tie_pct` (default 0.5%) of best median. Sort by: `(p95, workspace_bytes, not_native, name)` — the first entry wins. The native candidate is preferred in ties (`not_native` sorts native before non-native).

**Workspace filter:** candidates exceeding `max_workspace_bytes` are excluded before selection.

**Native must be measured:** the native candidate must appear in valid measurements; if absent, reject the tuning run for that signature.

### 7.4 Hot-first priority

Tune by call frequency and estimated bytes/work first. Do not spend substantial tune time on one-off prompt tails until high-coverage signatures are complete.

## 8. Measurement hygiene

### 8.1 Use live tensors

Rotate across real tensors sharing the signature where possible to avoid unrealistic cache residency in measurements.

### 8.2 Interleaved candidates

Randomize or interleave candidate order in final measurement rounds to eliminate systematic ordering bias.

### 8.3 Adequate sample counts

Screening: 15–30 samples with warmup. Final: 100 samples with warmup. Launches per sample must exceed event noise for tiny kernels.

## 9. Replay and production

### 9.1 Production never benchmarks

Production builds contain no tuning engine, no SQLite, no benchmarking. They load a compact immutable winner cache and dispatch directly.

### 9.2 Miss handling is native-fallback only

A replay miss falls back to the native selector and records the miss. Production never attempts online measurement.

### 9.3 Cache validation at startup

Validate schema/ABI/manifest hash/checksum before accepting a cache. Incompatible or corrupted caches fall back to native mode safely.

## 10. Multi-GPU behavior

### 10.1 Per-device signatures and winners

Each GPU uses its own hardware key and local tensor slice. The same logical model node can have different keys and winners on `gfx1100` and `gfx1201`.

### 10.2 Identical hardware can share winners

When hardware key and local signature match, identical GPUs can share a winner. PCIe bus identity is not part of the local-kernel key.

### 10.3 Topology belongs in validation — not signatures

The full topology (tensor split, RCCL, graph mode) must be validated end-to-end but is not part of the per-kernel signature. Local candidate timing isolates one device; final validation uses actual production topology.

## 11. Fused operations

### 11.1 Fused paths are distinct semantics

A fused graph pattern (e.g., fused MMVQ) is a different semantic operation. Tune lower-level variants within the already-selected fused family. Do not compare fused versus unfused decompositions in the matmul tuner.

### 11.2 Record fused signatures separately

Fused operations get their own semantic signature fields. Graph-pattern tuning is a later subsystem.

## 12. C++ coding conventions for this project

### 12.1 File naming

- HIP autotune types: `ggml/src/ggml-cuda/hip-autotune-types.h`
- Dispatch logic: `ggml/src/ggml-cuda/hip-autotune-dispatch.{cu,cuh}`
- Signature construction: `ggml/src/ggml-cuda/hip-autotune-signature.cpp`
- Database persistence: `ggml/src/ggml-cuda/hip-autotune-db.cpp`
- Record mode: `ggml/src/ggml-cuda/hip-autotune-record.cpp`
- Tuning engine: `ggml/src/ggml-cuda/hip-autotune-tuner.cu`
- Metrics collection: `ggml/src/ggml-cuda/hip-autotune-metrics.cpp`
- Replay cache: `ggml/src/ggml-cuda/hip-autotune-replay.cpp`
- Public header: `ggml/include/ggml-hip-autotune.h`

### 12.2 Compilation guards

All HIP autotune code is gated by `GGML_USE_HIP` and the appropriate feature flag (`GGML_HIP_AUTOTUNE`, `GGML_HIP_DISPATCH_REPLAY`). Code must not compile into CUDA-only or CPU builds.

### 12.3 No upstream behavior change without proof

Any refactor that touches existing dispatch paths must demonstrate:

- same family chosen for every tested signature;
- bitwise-identical output;
- no material performance regression in native mode.

### 12.4 Hard eligibility before launch

Eligibility checks (shared memory, resource limits, semantic compatibility) are performed before attempting a launch — never let invalid candidates fail at launch time.

## 13. Upstream churn controls

### 13.1 Embed source and manifest hashes

Every build embeds the llama.cpp source revision and candidate manifest hash. A changed upstream selector or candidate set creates a new build namespace rather than silently reusing old results.

### 13.2 Source audit invariants

Run `source_audit.py` in strict mode on every new upstream checkout before applying patches. Save the audit JSON with build artifacts. The audit validates these hard invariants:

- MMQ generated types == runtime type cases (set equality)
- MMQ J switch == `range(8, 129, 8)` — exactly 8, 16, 24, ..., 128
- MMVF block size switch == `range(32, 257, 32)` — exactly 32, 64, ..., 256
- MMF generated widths == `range(1, 17)` — exactly 1 through 16
- MMF nwarps switch == `range(1, 9)` — exactly 1 through 8
- MMVQ has `calc_nwarps` and `calc_rows_per_block` constexpr policy
- HIP build globs `ggml-cuda/*.cu`, `mmq*.cu`, `mmf*.cu`, links rocBLAS and hipBLAS

MMQ config files are architecture-specific:

| Config file | Architecture |
| --- | --- |
| `mmq-config-rdna3.cuh` | gfx1100 |
| `mmq-config-rdna4.cuh` | gfx1201 |
| `mmq-config-cdna.cuh` | CDNA targets |
| `mmq-config-rdna2.cuh` | RDNA2 targets |
| `mmq-config-rdna3-5.cuh` | RDNA 3.5 targets |

### 13.3 Source audit before patching

Run `source_audit.py` in strict mode on every new upstream checkout before applying patches. Save the audit JSON with build artifacts.

## 14. MMVQ compile-time bounds

The explicit MMVQ kernel enforces these `static_assert` bounds:

- `width >= 1 && width <= MMVQ_MAX_BATCH_SIZE` (8)
- `nwarps >= 1 && nwarps <= 8`
- `rows_per_block >= 1`

Generated variants must satisfy all three before compilation. Invalid geometries are rejected at compile time, not runtime.

## 15. Observation and record mode semantics

### 15.1 Duplicate signature merging

When a signature is observed a second time:

- increment `calls` counter;
- append new site to `sites_json` array (if not already present);
- merge new diagnostics into `diagnostics_json` (later values override earlier for same key).

Diagnostics fields (model name, layer number) must NOT affect the signature hash — they are observation metadata only.

### 15.2 Replay resolver process cache

The first encounter of a dispatch key performs DB lookup. The result is cached in-process keyed on the dispatch key digest. Later encounters are dictionary hits with zero hashing cost.

## 16. BLAS integration

### 16.1 hipBLAS is initially one opaque candidate

Treat `blas:hipblas-auto:v1` as a single candidate (source class `vendor_auto`). Do not enumerate internal hipBLAS solutions at first.

### 16.2 Vendor solutions require exact library namespace

Later hipBLASLt integration must namespace results by the exact ROCm/hipBLASLt version.
