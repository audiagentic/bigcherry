---
id: VTP14
order: 13
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-11T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Radiance-native three-GPU TP3 integration — build with Radiance, not beside it

## Decision and success condition

**Radiance is the only serving engine.** Develop the asymmetric three-physical-GPU
Qwen3.8-27B path as small, source-pinned, tested **Radiance changes** using its
existing architecture plugins, kernel registry, weight loader, rank threads,
activation graph, scheduler, KV/GDN state, memory planner, host-staged collectives,
tokeniser and sampler. Do **not** grow the VTP20–VTP29 Python/HIP programs into
another layer loop, model loader, scheduler, KV manager, logits path or service.
Those programs become immutable reference oracles and microbenchmarks.

Success is a **real Radiance** process running one physical rank/context per
RX 7900 XTX, RX 7900 XTX and R9700 (gfx1100, gfx1100, gfx1201), with all
three participating concurrently in the same distributed layers, generating
quality-qualified text, and showing a defensible improvement over the best
same-quality dual-XTX TP2 in **4/6/8-stream throughput and measured resident
KV capacity**. If the gains do not materialise, record NO-GO rather than
keep building a separate runner. Full-length context up to the model's
262,144-token limit is an *admission experiment*, not a promised capability.

This is the execution breakdown of VTP13 (experimental worktree, not yet in main), not a second
competing implementation. VTP03 (experimental worktree) virtual TP4 remains an optional
comparison **only** if native TP3 demonstrates a concrete bottleneck that
another rank layout might fix. Current production defaults are untouched.

## Ground truth and integration boundaries (11 November 2026)

- **Done, card-free only:** commit `72f51ea44`, opt-in
  `rad0016_vtp30_native_tp3_ownership` under
  `source.radiance-vtp30-native-tp3`, compiled against pinned Radiance
  v1.17.0 headers, with managed source-patch/idempotence tests. It declares
  Q/output-gate heads **6/6/12**, attention KV **1/1/2**, GDN key **4/4/8**,
  GDN value **12/12/24**, and FFN widths **5120/4864/7424** using
  `rad_weight_shard_span` and `rad_weight_shard_span_parts`.
  This is **not** a runnable mixed-architecture engine.
- **Existing Radiance:** one process with **one rank thread per GPU**;
  each rank already binds its HIP device in `Engine::declare`; a shared
  registry resolves kernel rows, with rank-indexed architecture state.
  Native loader, native graph, state/KV, scheduler and sampler already exist.
- **Known blocker A:** `core/engine_bringup.cpp:usable_devices` keeps
  devices matching the first card's architecture. The engine passes
  `devs[0]` to `Registry::set_code_device`; `core/plugin/loader.cpp`
  removes GPU rows unsupported by that *one* card; selection also uses
  one registry-wide device-fact view. Removing the filter alone would
  select invalid gfx1100/gfx1201 code. Also the reference-kernel
  admission gate presently checks `ranks_[0]->program` only.
- **Existing no-P2P collective, to prove rather than rewrite:** Radiance
  `libr4d/r4d_rows.cpp` generates `_host` twins of the three-rank
  exact device all-reduce rows with `device.p2p == 0`, calling
  `r4d_rad_ar_host_init` and using pinned-host mailboxes
  (`libr4d/r4d_ar_entry.hip`). The host-backend
  `libref/ref_collective.cpp` is **not** proof of GPU-device
  collective correctness. Check the actual chosen `*_3rank_exact_host`
  row on each physical device; reuse the native existing transport.
- **Existing oracles:** VTP13 header/owner audit, VTP19 exact collective,
  VTP20 GDN, VTP23 attention, VTP25 FFN, VTP27 connected GDN→FFN
  provide **bounded** source-backed operator evidence; no 64-layer
  model generation, KV continuation at scale, native mixed-gfx dispatch,
  4+/8+ streams or end-to-end speedup is yet established.

## Ordered implementation slices and executable gates

Each slice stays **inside the Radiance extension/patch workflow**,
with its own isolated source build, source-drift guard, C++/component
tests, CPU smoke where applicable, original TP1/TP2 regression, and
reviewable evidence. The following are dependencies, not parallel
streams inventing independent execution paths.

| Order | Native change / owner | Must be demonstrated to advance |
| --- | --- | --- |
| **N0 — baseline and source provenance** | Reproduce the pinned v1.17.0 production and experimental build pipeline, record source/patch set/plugin binary hashes, native TP2 exact-dispatch graph, full model identity, launch/power/KV settings, current quality and throughput baseline. Owner: BigCherry recipes and existing benchmark/evidence tools; no Radiance code. | The **same** native Radiance model can be built and run in known TP2 configuration; original source vs experimental patches are distinguishable. No unlabelled mismatched harness comparisons. |
| **N1 — heterogeneous per-rank eligibility** | Extend Radiance device enumeration/config to opt in to an **explicit ordered physical device assignment** (stable BDF/UUID, unique GPU per rank). Refactor plugin code-object eligibility and selector device facts to bind to **each rank's device**, without mutating a global registry during concurrent declare. Accept compatible fatbins/plugins per rank, not merely a library's claimed `build_target`. Keep uniform TP1/TP2 unchanged. Owner: `core/engine_bringup.cpp`, `core/plugin/loader.cpp`, `core/plugin/select.cpp`, `core/plugin/registry.h` and any minimal typed build-context extension. | Fake gfx1100+gfx1201 inventory selects correct rows on each; deliberately missing gfx1201 code refuses **before model-weight/KV allocation** with device/rank, op/shape and reason; wrong BDF, duplicate physical GPU, unselected GPU, CPU-only execution with device-only buffers and wrong fatbin all refuse. Native TP2 remains byte-identical at the interface and pass-equivalent in tests. |
| **N2 — all-rank graph and native source-shard proof** | Connect N1 to the existing Qwen BF16 architecture and VTP30 ownership. Use the **normal Radiance model file/checkpoint loader and encoding/layout conversion**. Validate *all* declared source views, per-rank matrix shapes, stacked GDN pieces, shared norms, full attention/gated Q, embeddings, vocab shards, output head, metadata and optional MTP. Enforce identical collective *call order* across unequal ranks. Owner: `arch/qwen35_bf16`, `arch/common/rad_block_{gdn,attn_gated,mlp}.h`, `core/build`/loader **only where an existing native contract is genuinely insufficient**. | A real native `--debug-graph` on **each** rank names device, kernel, domain, operand shapes, weight spans, nonreference status and collective slot. Every original source row is assigned once or explicitly replicated; no silent reshaping/padding, host CPU fallback, out-of-range source span or collective-rendezvous mismatch. |
| **N3 — existing three-rank no-P2P collective and scheduling** | Reuse `libr4d` **exact three-rank pinned-host twins** for the no-P2P world; validate matched selection on gfx1100 and gfx1201, correct peer/device pointers, ascending-rank F32 sum, BF16 residual protocol, stream/event lifetimes, cancellation and graph replay. Repair the native library or core only if measured tests show a defect. No alternative Python relay or hand-coded per-layer runtime. | Under an exclusive GPU lease, one native three-card exact all-reduce/broadcast on M1/M4/M8 and representative prefill sizes matches original-BF16 numerical oracles and does not hang on abort/restart. Every rank uses `*_host` transport when `--p2p off`; record pinned bytes, PCIe bytes, p50/p95 and exposed wall time. |
| **N4 — first complete original-BF16 Radiance inference** | Execute normal Radiance embedding→64 native GDN/attention/FFN layers→norm→lm_head→sampler with physical TP3, initially text-only, speculative decoding OFF and **only after existing native model admission succeeds**. Preserve existing recurrent conv/GDN scan/decode state, paged causal KV and state-update ordering; no synthetic hidden handoff or per-layer disk files. Fill native GPU kernel gaps via `libr4d`/`libr3` rows or opt-in kernel plugins **selected through Radiance**, not a parallel harness. | Native engine produces repeated, nonempty and deterministically reproducible tokens from real prompts. Differential BF16 1-step and multi-step hidden/greedy checks vs trusted TP2/reference, plus GDN state continuation and attention prefix history, pass named tolerances. Every rank's nonreference kernel table is complete or startup refuses. |
| **N5 — measured admission and long-context safety** | Extend **Radiance's current rank memory planner and KV allocator**, not a new memory manager: each physical BDF's actual weights, scale/relayout, embedding/head, GDN recurrent/conv, attention KV, graph scratch, pinned host mailboxes/staging, launch margin and live free memory. Ensure R9700 two-KV-head/24-value-head allocation is real, not guessed from TP2. BigCherry launcher consumes engine measurements through the existing RSC03/RSC10/RSC11 measured-plan/supervisor contracts when activated, rather than a second config implementation. | Feasible prompt/context/4/6/8-stream cases have per-rank allocation receipts and no hidden spill; deliberately infeasible allocations refuse **before starting** with device, required/available bytes and safe adjustment. Verify 8K, then 32K/64K/128K/262144 **only if admissible**, independently for one request and aggregate concurrent KV. |
| **N6 — native quantisation and optimisations** | First compare native original BF16 TP3 and all-three **same-quality W8A8** baseline. Plug XDP10's **selected-only** artifact/variant loader into Radiance's existing weight encoding and per-card operator selection; test XTX INT8 + R9700 FP8 only after a shared activation/residual/collective ABI and independent KLD/PPL/top-1 pass. IQ6R only after VTP06 quality+kernel qualification. Optimise chosen `libr3`/`libr4d` kernels and Radiance call paths; no shadow GEMM runtime. | End-to-end correct decoding and quality first; then actual 4/6/8-stream slowest-rank barriers improve. No per-layer weight variant switching, silent precision fallback, extra VRAM resident variant copies or hidden conversion costs. |
| **N7 — matched end-to-end acceptance and promotion decision** | Run Radiance-native TP2 vs native physical TP3 at same model/quality/KV, fixed prompt versions, power, CPU/host-pool, clock, prefill/decode and sampling; optional virtual TP4 only if N6 identifies a reason. Own all performance and context receipts under BigCherry's existing canonical workloads/ledger. | Balanced fresh-start ABBA repeats (at least 3), 4/6/8 streams at 8K plus pp512/2048/8192, then admitted long contexts. Report actual output tok/s, TTFT, p50/p95 latency, each GPU's critical-path work, exact-collective/PCIe cost and physical resident weights/KV/GDN/scratch. PASS only if quality, cancellation, admission, and throughput **and resident capacity goals** meet the recorded decision threshold; otherwise NO-GO. |

### Hard execution order

**N0 → N1 → N2 → N3 → N4 → N5 → N6 → N7.**
Individual source-audit and CPU tests may proceed without GPUs; no
full-model hardware benchmarking while N1–N4 remains unproven. Keep the
known-good native TP2 executable pinned as the immediate regression
control at every slice. After N4 proves the end-to-end original BF16
path, focus on performance/quality and capacity; **do not resume a
standalone VTP28+ model runtime.** If N4 fails for one missing operator,
add only its Radiance kernel row and corresponding test, then retry N4.

## Shared build/test and ownership contract

**One engine, one build:** maintain a single pinned upstream
Radiance v1.17.0 source baseline with BigCherry managed opt-in
source-patch packages and existing `engines/radiance/recipes.toml`
source/build entries. Materialise a new **isolated Radiance build** for
each slice (content-addressed source+patch set, no edits to live
upstream checkout/production binary). Build Radiance itself, its
normal `qwen35_bf16` architecture, `libr4d`/`libr3` kernel
plugins and component tests from that source; use the same normal
`bc-serve`/engine launch path. A standalone shared-library build
like VTP30 is only a **compile smoke**. Move accepted generic changes
as compact upstream-compatible Radiance patches when permitted; the
pinned-source guard must fail loudly on upstream release drift.
Do not build an internal `radiance2` fork or a permanent VTP-side
serving library.

**Boundaries of code ownership:**

| Radiance is authoritative | BigCherry is authoritative | Prototype-only |
| --- | --- | --- |
| GPU enumeration and per-rank kernel selection; original model graph; source weight spans; GPU operators; all-reduce and event protocol; KV/GDN state; memory planning; scheduler; decode/sampling | Versioned source recipe and patch catalogue; stable ordered GPU/BDF policy via manifest/supervisor; artifact variant selection; build provenance; workload/result ledger, qualification and promotion | VTP13/VTP19/VTP20/VTP23/VTP25/VTP27 original-BF16 numeric and timing oracles; read-only fixture/receipt creation |
| Fix a missing kernel by adding a Radiance plugin row, not by adding a direct HIP call in Python | Do not duplicate a layer runner, native memory allocator, kernel registry or sampler | Freeze receipts; only extend a microbench to isolate a specific native discrepancy |

**Every code slice's local gate:** unique source-pinned anchors and
idempotent atomic apply; static/formatter tests; CMake build of the
actual Radiance target; affected component CTests and CPU-backend
rank/graph tests; negative/error-path tests; native TP1/TP2
regression; per-rank debug-graph/kernel-domain table; optional leased
GPU smoke with matched numerical oracle; independently recorded
build/source/power/lease receipts. A missing test or unsupported
gfx1201 kernel must be reported as a blocker, not silently substituted.

**Critical failure cases to simulate:** heterogeneous GPU missing,
wrong ordered BDF mapping, incompatible fatbin/kernel selector,
device-only activation sent to CPU fallback, uneven stacked source
span, incorrect per-rank KV/GDN cache bytes, missing/no-P2P
`all_reduce` host twin, rank-specific extra collective, one rank
failing during initialization, cancellation/graph replay after a
partial collective, and host-memory or VRAM exhaustion.

**GPU discipline:** all hardware tests acquire and release their own
exclusive `agent-lease gpu`; do not acquire a manual lease over
an idle build; do not interrupt a competing owner. No GPU required
for source changes, CPU fixtures, patch application or builds.

**Integration with the serving programme:** RSC11 v2 supervision,
RSC10 resource refusal/retry and XDP10 variant selection remain
separate workstreams and provide their already-defined boundaries.
A new TP3 policy is never allowed to fork the launcher/manifest
contract or bypass power transactions. If the v2 manifest isn't
activated, run this opt-in Radiance source recipe in a research lane
until the real cutover.

## Concrete next slice: N1 (not N4 or another GPU harness)

1. Create one managed Radiance source patch for **rank-specific
   code eligibility**, with per-rank device facts. First audit
   all calls that consult the singleton registry and any
   `device.p2p`/device-code row filtering; choose a rank-scoped
   *immutable selection context* or per-device selector views.
   Do **not** mutate global `code_device_` while rank threads
   resolve concurrently. Preserve plugin lifetime and collective
   initialization rendezvous.
2. Make the ordered three-device mixed-gfx selection an **explicit,
   default-off** config with input validation. Do not allow a generic
   `--tp 3` to bypass the existing uniform-architecture filter.
   Record rank→BDF/arch, library+row, transport and source patch in
   a startup receipt. Keep the ordinary `--tp 2` path unchanged.
3. Validate all per-rank resolved kernel rows **before loading model weights or allocating model KV**,
   including the reference-kernel gate on every rank
   (not only rank 0). Missing gfx1201 code, an invalid GPU-only host
   fallback or no native no-P2P exact collective refuses with an
   actionable reason.
4. Build the **whole Radiance tree**, run host fakes/negative cases
   and TP2 regressions; finish N1 with a recorded per-rank kernel
   coverage table. Only then start N2 on original BF16 weight
   mapping and native graph execution.

**Current status:** N0 baseline needs refreshing and N1–N7 are open.
The VTP30 source/geometry patch is compile-only, not an N1/N2
integration acceptance. No production activation, no auto-merge,
no push/PR without explicit instruction under VTP13's research
workflow.

## N1 implementation start, 2026-11-11 (source-pinned card-free slice)

In \`feat/vtp14-n1-rank-code-eligibility\`, managed patch
\`rad0016_vtp14_rank_code_selection\` adds an **inactive** registry
primitive: an explicitly ordered list of physical device IDs with
immutable per-plugin kernel code-object eligibility per rank.
The existing \`RadBuildCtx.rank\` passes through the native builder
to the selector without a global mutable current-rank setting.

This is deliberately **not** complete N1: heterogeneous device
enumeration, rank-scoped selector \`device.*\` facts, per-rank
debug-graph trace counts, verified complete-graph coverage and the
startup safety gate remain to be built and tested. Production
\`source.radiance\`, the same-architecture filter and
\`set_code_device\` remain unchanged. This is the first
standalone PR-sized native implementation sub-slice.

Baseline N0: the base of this worktree is origin/main
commit 5dbe356f3, pinned engine v1.17.0. The
reproducible **native TP2 hardware** acceptance is still
pending and must be performed under an exclusive GPU lease
before any TP3 comparative result.

## N1 rework, 2026-11-11: applies to real v1.17.0, wired behind `--rank-devices`

The first rad0016 did not apply to the pinned source. Applied to the real v1.17.0 source, edits to `rad_core.h`,
`loader.cpp` and `select.cpp` failed, because they were anchored on text that v1.17.0 does not contain. Its test
built its fixture from the patch's own anchors, so it could not catch this. The rewrite:

- **Patch.** Every anchor is exact upstream text. The patch test now also applies the patch to the pinned commit
  19d4e8fe when a radiance checkout is reachable.
- **Row mask.** `KernelRow::code_ranks` is a `uint32_t` mask, read from the fat binaries the dlopen registered, for
  each rank's card. Selection refuses a row whose code the asking rank's card cannot run, names the plugin, and
  falls through to the next library. A question asked for no rank requires every rank's bit.
- **Rank list.** `--rank-devices BDF,...` is an explicit, ordered list of PCI addresses. It is the only route to a
  mixed-architecture world, and it is refused unless it names exactly `--tp` unique, discrete, visible cards.
- **Startup checks.** Every rank passes the reference-kernel gate, every rank must issue the same collective
  kernels in the same order, and each rank logs a receipt.
- **Tests without a card.** `tests/bc_rank_devices_test.cpp` covers planning, refusals, per-rank selection, mask
  forwarding through resolve and bands, and the unchanged legacy path.

## N3 finding: the all-reduce rendezvous is per library (confirmed by source reading)

**What blocks it**

- `libr4d/r4d_ar_entry.hip` holds its peer rendezvous (`Rendezvous g_rv`, `rv_barrier(world)`) in static storage
  of the kernel library.
- In the target world, the XTX ranks select the all-reduce from libr3 (the gfx1100 rewrite of libr4d,
  `engines/radiance/components/libr3`) and the R9700 rank selects it from libr4d. Each library therefore has its
  own `g_rv`, and neither one ever counts `world` arrivals.
- The first all-reduce waits until the 60 s collective timeout.
- The N1 agreement check passes this world, because the kernel names and protocol constants are identical in both
  libraries (`R4D_AR_WIDE_MAX_BLOCKS 64`, `NB_DESIGN 16`, `MAX_PEERS 7`, `R4D_AR_MAX_BLOCKS 512`). Only the
  rendezvous storage is split.

**Design (N3 patch)**

- A managed core patch exports a process-wide `rad_bc_shared_slot(key, size, init)`: the engine allocates the
  slot once under a mutex and keeps it for the life of the process.
- A managed libr4d patch looks the slot up with `dlsym(RTLD_DEFAULT, …)` and keys it on the protocol version.
- There is no fallback.
  - The engine decides explicitly whether the world is single-library or mixed, from the per-rank receipt.
  - A single-library world keeps the library's own `g_rv`.
  - A mixed world requires the shared slot. A failed lookup or a protocol-fingerprint mismatch refuses startup with
    the reason. Falling back silently would bring back the 60 s deadlock.
- libr3 is generated from the patched libr4d source, so it inherits the change. Its `rewrites.txt` and `.rw`
  anchors must keep matching; the libr3 build checks this.
- The N1 agreement check is then extended to compare a protocol fingerprint as well as kernel names.

**Acceptance (needs the GPU lease)**

- one exact three-rank all-reduce on M1, M4 and M8 with `--p2p off`, matching the BF16 oracle;
- no hang on abort or restart.

## N1 hardware smoke and N3 reproduced on cards, 2026-11-11

All results below are from build_plan_id 2a68d91b0326be18a461da62fb58d294 (source
radiance-vtp14-rank-code-selection, stock lane, ROCm 7.2.4), taken under the gpu lease.

**Builds and tests**

- **HIP test build** of the same materialised tree:
  - `bc_rank_devices_test` (115 checks) passes.
  - `plugin_test` and `config_test` pass.
- **Pre-existing failures** (images_test, videos_test, glm5next_test and arch_test, the last two failing on
  `bc_moe_remote_*` links) are identical on the production `radiance` source, so they are not caused by rad0016.

**Mixed-card run**

- Command: `--tp 2 --rank-devices 0000:03:00.0,0000:09:00.0 --debug-graph`, with qwen3.8-27b-fp8-df2. That puts
  one XTX (gfx1100) and the R9700 (gfx1201) in one world.
- The ranks are planned and libraries are masked per rank:
  - libr3: 587 device rows, offered only to rank 0 (XTX).
  - libr4d: 587 device rows, offered only to rank 1 (R9700).
  - libbcgfx12 and spechub: offered only to the R9700.
  - libr2 (gfx1030): dropped for both ranks.
- p2p is off (0->2 and 2->0), so the collectives select the `*_host` twins.
- **N3 reproduced:** declare then fails in all-reduce init. Every `ar_oneshot_2rank_exact_host` and
  `ar_rmsnorm_quant_fp8_host` instance in both libraries is refused ("called in the wrong phase"): rank 0 in
  libr3, rank 1 in libr4d. The process hangs and does not exit.
- The per-library rendezvous therefore blocks the **two-rank family as well as the wide (3/4/8-rank) family**.
  The N3 slot must cover every `g_rv` user.
- **Consequence:** the shared-slot rendezvous is a prerequisite for *any* tensor-parallel world that mixes libraries
  (libr3 + libr4d), TP2 included. Do not try a mixed XTX+R9700 TP2 as an interim step before N3 lands: it hangs in
  declare.

**Cosmetic issue**

- Upstream's `usable_devices` note ("using 2 of 4 … ignoring gfx1201") is still printed before `--rank-devices`
  overrides the choice. A small follow-up should suppress that note when the list is given.

## Independent assessment (reviewer-gpt-agent, session ses_ad7d3f16e1f44251)

The full text is in the task scratch folder (`vtp14/gpt-assessment-n2-n3.md`). These points are adopted into
the plan:

**N3**

- Fail closed. Do not fall back to the local `g_rv` in a mixed world.
- Share only `Rendezvous`. `ArInst`, `g_cache` and `g_ar8` stay local to each library.
- The slot key must include the world/session identity and a layout fingerprint (sizeof and offsets).
- Poison the rendezvous and wake every waiter on timeout. Never reuse a slot that is only partly initialised.
- Assert `nm -D` exports `rad_bc_shared_slot`.
- Run `r3_rewrite.py --check`.
- The N1 agreement check must also compare launch geometry: world, transport, dtype, numel, variant and
  **nblocks**. A per-rank tuned nblocks can strand spin blocks.

**N2**

- Use the existing `rad_weight_shard_span(_parts)` ABI. No new loader is needed.
- Embedding and lm_head need explicit vocab spans (248320 % 3 != 0). The ragged vocab is
  [0,82816), [82816,165632), [165632,248320).
- Qwen BF16 calibration assumes uniform global widths. Fix it or refuse it under an asymmetric profile.
- Select the ownership profile through typed config, not an environment variable.
- The experimental VTP30 ownership patch in `wt-xtx27-prefill` (also numbered rad0016) is the starting reference,
  read-only. N2 will take a new patch number.

## Milestone, 2026-11-11: mixed XTX + R9700 tensor parallelism serves natively

- **Build:** build_plan_id b306fbef72b21fd746a02394852250f3, source radiance-vtp14-rank-code-selection (rad0016 + rad0017),
  ROCm 7.2.4. Run under the gpu lease.
- **Command:** `--tp 2 --rank-devices 0000:03:00.0,0000:09:00.0 --max-num-seqs 4 --max-model-len 4096`,
  with qwen3.8-27b-fp8-df2.
- **What the receipts show:**
  - rank 0 is the gfx1100 card (libr3, 4425 device bands);
  - rank 1 is the gfx1201 card (libr4d, 4425 device bands);
  - both ranks issue 901 matching collective bands;
  - the all-reduce rendezvous is "shared by libr3.so, libr4d.so".
- **Output at temperature 0:** both answers are correct ("The capital of France is Paris." and
  "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"), at 43.1 and 45.5 tok/s single stream, with a TTFT of 0.26–0.30 s. The server
  shuts down cleanly on SIGINT.
- These are not benchmark numbers. They are not leased A/B measurements and there is no N0 TP2 baseline beside
  them.

**Two blockers fixed on the way**

1. **N3 (rad0017):** the shared all-reduce rendezvous.
2. **HIP gives no handle for code device 0 cannot run (fixed in rad0016, `core/device/aql.cpp`).**
   - HIP's own `__hipRegisterFatBinary` returns no handle for a fat binary that has no code for device 0, even
     when a later visible card runs it.
   - Measured with an LD_PRELOAD probe while all four cards were visible: 109 of 109 libr4d binaries returned null,
     as did libbcgfx12, spechub and libr2. Every libr3 binary got a handle.
   - The AQL backend loads code objects from the bundle per device, so it now issues its own handle for such a
     binary and never passes that handle to HIP.
   - Production avoided the bug only by restricting `ROCR_VISIBLE_DEVICES` to one architecture.

**Next:** N2 (6/6/12 ownership, required for TP3), then N0 (the TP2 baseline) and the leased TP3 run.

## Milestone, 2026-11-11: native TP3 serves the full 64-layer Qwen3.8-27B (BF16, 6/6/12)

- **Build:** build_plan_id 4d0658f6c48493cb971dbda684b95e84, source radiance-vtp14-rank-code-selection (rad0016, rad0017,
  rad0018 and the libr3 fixes below), ROCm 7.2.4. Run under the gpu lease.
- **Model:** `qwen3.8-27b-bf16.rad`, converted by rad-convert from the original HF checkpoint (51 GiB, 1024 weights,
  2.5 min).
- **Command:** `--tp 3 --rank-devices 0000:03:00.0,0000:06:00.0,0000:09:00.0 --tp-ownership qwen38-27b-6-6-12
  --max-num-seqs 4 --max-model-len 4096`.
  - Ranks 0 and 1 are the two gfx1100 cards (libr3); rank 2 is the gfx1201 card (libr4d).
  - Each rank has 2470 device op bands and 389 matching collective bands.
  - The all-reduce rendezvous is "shared by libr4d.so, libr3.so".
- **Output at temperature 0:** the answers are correct and coherent ("The capital of France is Paris.", "1, ..., 10",
  and a correct two-sentence Rayleigh-scattering explanation).
  - Single stream: 23.2–23.4 tok/s, TTFT 0.10–0.11 s.
  - Four concurrent streams: each correct, 20.0–21.7 tok/s per stream, no errors.
  - The server shuts down cleanly.
- These are smoke numbers. They are not N0/N7 measurements: no matched TP2 baseline, no ABBA, no power control.

**libr3 blockers found and fixed (component, not an upstream patch)**

1. **3/4/8-rank all-reduce never built.**
   - The unit pattern `r4d_[a-z0-9_]+` never matched libr4d's `*_Nrank*` units (capital N). They were skipped without
     a word ("0 left out") and stubbed, so libr3 had no 3/4/8-rank all-reduce.
   - Visible symptom: `r4d_ar_wide_dims` was a stub, `max_peers` stayed 0, and every 3-rank init returned
     RAD_E_SHAPE.
   - libr3 now builds 111 units, up from 107. Configure fails if any libr4d unit line does not parse.
2. **IQR layouts stubbed, and the wrong logits kernel selected.**
   - libr3's hand-kept declaration list missed `r4d_iqr_layout.cpp`, so every IQR layout was a stub.
   - A stub layout returns RAD_E_UNSUPPORTED, which the engine's weight check does not count as a refusal. So
     `logits_gemm_iqr` was selected for a bf16 lm_head on gfx1100 and refused at issue on the first step.
   - The list is now read from libr4d's own R4D_DECLARATION.
   - A row is live only if none of its functions is a stub. libr3 now has no stubs at all.

**Next**

- **N0:** a matched native TP2 baseline (2× XTX) against TP3 at 4/6/8 streams, same model family and settings, under
  the lease.
- **N5:** admission and context, with per-rank KV/GDN state checked per rank.
- **N6:** quantised TP3. qwen35_fp8 and w8a8 need the ownership spans too.

## N0 correction and MXFP4 TP3, 2026-11-11

**The first N0 attempt measured the wrong model.**

- The attempt used qwen3.8-27b-fp8-df2, with speculation off.
- It measured TP2 at 151 ms per decode step and a 10 s TTFT for 271 tokens.
- A same-settings A/B gave exactly the same 6.5 tok/s on the production `radiance` build and on the vtp14 build. So
  that fp8 container is simply slow on gfx1100, which has no fp8 hardware, and the patches are not the cause.
- The XTX `power1_cap = 0` that was suspected first was not the cause either. It predates the power chain, and earlier
  stock runs were unthrottled at that setting.
- The production two-XTX layout (`layouts.toml` `qwen27b-2xtx`) serves the **MXFP4** container, so the two aborted
  N0 runs are discarded.

**Fix: an uneven delta-net split now works on MXFP4 (rad0019).**

- `LinearFP8::span_parts` spanned an MXFP4 linear's codes and scales, but not its `wref` plane. The single-span path
  already did.
- As a result, TP3 refused `blk.0.ssm_inz.ref` (16384 rows over 3 ranks).
- This is a latent upstream bug: nothing used an uneven head split on MXFP4 before.

**Smoke results** (build 93dcced3bff30928b0f7fffc120a00e6, `--kv-cache-dtype fp8 --num-speculative-tokens 0
--max-num-seqs 8`; untimed, with other CPU work running)

| Configuration | Single-stream decode | TTFT | Output |
|---|---|---|---|
| TP2, two XTX (wht6 or exact) | about 50 tok/s | 0.06 s | correct |
| TP3 6/6/12, XTX + XTX + R9700 | 64–65 tok/s | about 0.05 s | correct; rendezvous shared by libr3.so and libr4d.so |

**Next: the timed N0 ABBA on MXFP4** (TP2, TP3, TP3, TP2 at 1/4/6/8 streams, quiet CPU, lease).

- Speculation is off for both, because TP3 refuses a drafter under a profile.
- This makes it the matched baseline, not the production best case. The production TP2 configuration with
  speculation on is recorded separately as a reference.

## N0 result, 2026-11-11 23:54Z: matched ABBA, native TP3 beats native TP2 at 4/6/8 streams

**Setup**

- **Build:** 93dcced3bff30928b0f7fffc120a00e6, used for both arms.
- **Model:** qwen3.8-27b-mxfp4.rad, the container production serves on two XTX.
- **Engine flags:** `--max-model-len 4096 --max-num-seqs 8 --kv-cache-dtype fp8 --num-speculative-tokens 0`, exact
  all-reduce.
- **Arms:**
  - TP2: `--tp 2` on the two XTX.
  - TP3: `--tp 3 --rank-devices 0000:03:00.0,0000:06:00.0,0000:09:00.0 --tp-ownership qwen38-27b-6-6-12`.
- **Benchmark:** `bigcherry stream-bench`, 512-token target prompts (calibrated, 271 words), 128 generated tokens, 3 reps,
  temperature 0, seed 7.
- **Conditions:** run under the gpu lease. Every card was at its default power cap (303/303/300 W, checked by the
  script before it ran). The CPU was quiet. The order was A1 TP2, B1 TP3, B2 TP3, A2 TP2.
- **Logs:** `vtp14/logs/window-n0-2354` in the task scratch folder.

**Results** (medians of 3 reps; aggregate is tokens over the batch's whole makespan, so prefill is included)

| streams | TP2 aggregate tok/s (A1/A2) | TP3 aggregate tok/s (B1/B2) | aggregate gain | TP2 step ms | TP3 step ms | step gain | engine decode tok/s TP2 → TP3 |
|---|---|---|---|---|---|---|---|
| 1 | 44.2 / 44.1 | 54.5 / 54.4 | +23.4% | 20.27 | 15.35 | -24.2% | 49.3 → 65.1 |
| 4 | 111.3 / 109.6 | 129.6 / 128.4 | **+16.8%** | 26.50 | 19.56 | -26.2% | 151.0 → 204.5 |
| 6 | 130.7 / 130.4 | 147.0 / 147.4 | **+12.8%** | 31.38 | 23.54 | -25.0% | 191.2 → 254.9 |
| 8 | 141.0 / 140.6 | 159.3 / 158.7 | **+12.9%** | 37.37 | 27.76 | -25.7% | 214.1 → 288.2 |

**What this shows**

- Drift is small: the A arms agree with each other within 1.5%, and so do the B arms.
- The decode step is about 25% faster at every concurrency. The aggregate gain is smaller (13–17%) because prefill
  counts in it, and TP3's TTFT is somewhat higher (1.46 s against 1.24 s at 4 streams).
- **This meets the VTP14 throughput goal** (beat dual-XTX TP2 at 4, 6 and 8 streams) for the matched configuration.

**What is still open**

- Speculation is off on both arms, because TP3 refuses a drafter under a profile. Production TP2 with DFlash2 depth
  3 is faster than both (about 97 tok/s at one stream on v1.16). The next slice is the drafter under an ownership
  profile: replicate it per rank.
- KV capacity and context (N5).
- Quality evidence against an oracle (KLD/top-1, N7).

## DFlash2 drafter under the profile, 2026-11-12 00:06Z (rad0019, build 7b7cbf55def2d4dd19991f3f3c8bbcdf)

The drafter now runs under the 6/6/12 profile. Its heads split 8/8/16 query and 2/2/4 KV, and it takes the target's
feed-forward rows.

**Correctness:** with depth 3 on MXFP4 and the other N0 settings unchanged, the answers are correct. Draft acceptance
is 2.95 tokens per step at one stream, against 3.12 on TP2.

**Throughput** (stream-bench, aggregate tok/s, 3 reps; lease held, every power cap at its default)

| streams | TP3 6/6/12 + drafter, **exact** wire | TP2 + drafter, **wht6** wire (production) |
|---|---|---|
| 1 | 92.5 (step 22.8 ms) | 87.0 (step 29.4 ms) |
| 4 | 164.4 (34.7 ms, 10.8 tok/step) | 201.0 (35.7 ms, 12.0 tok/step) |
| 6 | 171.6 (50.1 ms) | 207.3 (56.1 ms) |
| 8 | 176.4 (60.6 ms) | 227.4 (56.8 ms) |

**This pair is not matched.** TP2 uses the lossy wire, which sends about 2.4× fewer bytes per all-reduce, and TP3
uses the exact one. TP3 also loses on prefill: TTFT is about 40–50% higher. That is consistent with the R9700 sitting
on a PCIe Gen4 x4 link (`link-below-max`: 0000:09:00.0 at 16 GT/s x4), which carries every host-staged all-reduce. Its
per-step decode is still faster at 1, 4 and 6 streams.

**Next:** the matched pair (TP3 on wht6, which is the tiered int8 wire at 3 ranks, and TP2 on the exact wire, both
with the drafter). After that, prefill: `--tp-attn-split rows`, and the R9700 link. The link is a hardware or BIOS
matter, not code.

### Matched pair with speculation on (exact wire on both), 2026-11-12 00:14Z

Same build, MXFP4, DFlash2 at depth 3, aggregate tok/s:

| streams | TP3 6/6/12 + drafter, exact | TP2 + drafter, exact | TP3 vs TP2 | step ms, TP3 / TP2 | tokens per step, TP3 / TP2 |
|---|---|---|---|---|---|
| 1 | 92.5 | 83.4 | **+11%** | 22.8 / 29.4 | 2.95 / 3.12 |
| 4 | 164.4 | 184.2 | -11% | 34.7 / 37.0 | 10.8 / 12.6 |
| 6 | 171.6 | 178.2 | -4% | 50.1 / 57.1 | 17.3 / 18.4 |
| 8 | 176.4 | 200.8 | -12% | 60.6 / 59.1 | 22.2 / 23.5 |

With speculation on, the TP3 step is still faster up to 6 streams, but **its drafts are accepted less often**. Per
stream, acceptance is 2.7 against 3.15 at 4 streams. The gap grows with concurrency, so it needs investigating
before speculation counts as supported under the profile: suspect the drafter's per-rank state or spans.
TTFT is also higher. Without speculation TP3 wins at every concurrency (N0 above).

**Lossy wire at three ranks (`--tp-wire wht6`, the tiered int8 two-shot) faults on TP3.**

- **Symptom:** a GPU memory fault on the R9700 during the 8-stream bench. `r4d_attn_decode_kernel<2,…>` received an
  argument of `0x3d800000` and another 1 MiB past a host-mapped pool. Those look like corrupted values, not an
  attention bug.
- **Contrast:** the exact wire is clean on the same build.
- **Suspect:** the 3-rank ti8 all-reduce, which libr3 has built only since the Nrank fix and which is unvalidated on
  gfx1100, or its mixed-library scratch layout.
- **Until this is fixed:** TP3 runs the exact wire only.

### Draft-acceptance fix, 2026-11-12 00:24Z (build e783077b6879d6d060379d55aa157d7c)

**Cause:** `vocab_head_sharded` required an equal vocabulary split. Under the ragged 82816/82816/82688 split the DFlash2
draft head was replicated, so it **skipped `logit_rerank`** against the target's exact head, and its candidates came
from the 2-bit head alone. That head's top-1 agrees with the exact head on 76% of draft rows.

**Fix (rad0019):** a contiguous ragged split in whole 16-row tiles now counts as sharded, and the draft head spans its
own rows. The merge reads global-id pairs formed from `vocab_off`.

**Result** (matched pair, exact wire on both arms, MXFP4, DFlash2 depth 3; one pass per arm, so differences of a few
percent are within noise):

| streams | TP3 6/6/12 + drafter | TP2 + drafter | TP3 vs TP2 | step ms TP3 / TP2 | tokens per step TP3 / TP2 | TTFT max TP3 / TP2 |
|---|---|---|---|---|---|---|
| 1 | **104.9** | 83.1 | **+26%** | 22.3 / 29.5 | 3.42 / 3.12 | 0.39 / 0.33 s |
| 4 | 178.9 | 181.5 | -1% | 35.1 / 37.2 | 13.3 / 12.8 | 1.50 / 1.29 s |
| 6 | 174.5 | 176.4 | -1% | 47.0 / 57.4 | 16.7 / 18.3 | 2.20 / 1.90 s |
| 8 | 186.0 | 196.8 | -5% | 57.2 / 61.3 | 23.2 / 24.0 | 2.95 / 2.53 s |

**What this shows**

- Acceptance recovered to TP2's level or better. The decode step is faster on TP3 at every concurrency.
- The remaining aggregate gap at 6 and 8 streams is prefill: TTFT is 15–17% higher. The R9700 is on a Gen4 x4 link,
  and the host-staged all-reduces of a prefill chunk are bandwidth-bound there.

**Next**

- Prefill: try `--tp-attn-split rows`, and look at the R9700 link (hardware or BIOS).
- An ABBA on the spec-on pair.
- The ti8 lossy-wire fault (under external review).

### Spec-on ABBA, 2026-11-12 00:37Z (build e783077b6879d6d060379d55aa157d7c)

**Setup:** MXFP4, DFlash2 at depth 3, exact wire on both arms, all other settings as for N0. Run under the lease with
every card at its default power cap. Order: TP2, TP3, TP3, TP2. Logs: `vtp14/logs/window-n0spec-0037`.

| streams | TP2 aggregate tok/s (A1/A2) | TP3 aggregate tok/s (B1/B2) | TP3 vs TP2 | step ms TP2 / TP3 | tokens per step TP2 / TP3 | TTFT max ms TP2 / TP3 |
|---|---|---|---|---|---|---|
| 1 | 83.1 / 83.2 | 104.8 / 104.4 | **+25.8%** | 29.4 / 22.3 | 3.12 / 3.42 | 331 / 395 |
| 4 | 179.5 / 182.6 | 175.7 / 175.6 | -3.0% | 37.0 / 34.9 | 12.55 / 12.71 | 1288 / 1500 |
| 6 | 174.3 / 176.5 | 176.9 / 177.4 | +1.0% | 57.2 / 47.6 | 18.15 / 17.76 | 1920 / 2255 |
| 8 | 200.1 / 199.9 | 188.5 / 186.4 | -6.3% | 62.0 / 56.6 | 25.00 / 23.57 | 2523 / 2982 |

**Status against the goal**

- **Without speculation (N0 above):** TP3 beats TP2 by +17%, +13% and +13% at 4, 6 and 8 streams.
- **With speculation:** TP3 leads at one stream and is at parity at 4 and 6 streams, but 6% behind at 8. Its decode
  step is faster at every concurrency (−6% to −17%). What it loses is prefill: TTFT is 17–18% higher.

**What would close the 8-stream gap**

1. **Faster prefill all-reduces.** The R9700's upstream root port 0000:00:06.0 is limited to Gen4 x4, while the card
   is Gen5 x16. A CPU-attached slot would fix this; it is a hardware change, not code.
2. **The lossy 3-rank wire.** Tiered int8 sends about half the bytes, but it currently faults (under review).
3. `--tp-attn-split rows` is not an option: the qwen35_fp8 plugin refuses it for this model.

### Lossy-wire fault: independent review and fail-closed gate, 2026-11-12

Reviewer: reviewer-gpt-agent, session ses_2150e61a7b824108. The full assessment is
`vtp14/gpt-assessment-ti8-fault.md` in the task scratch folder.

**Findings**

- **The `+120 = 0x3d800000` argument is the correct softmax scale** (0.0625 = 1/√256), not corruption. The fault
  dump's address annotations also tag scalar words.
- **The bad value is `+8`, the attention KV-cache base pointer.** It lies 1 MiB past the 809 MiB host pool, outside
  every logged allocation. The fault address is 8 KiB below it, which is consistent with a −1 block id on KV head 1.
- **Static reading finds no ti8 defect:**
  - no out-of-bounds write;
  - no staging-size mismatch;
  - no libr3/libr4d divergence (the ti8 kernel and codec are byte-identical across the two libraries);
  - no fused-op overflow (the fused AR norms refuse world ≠ 2).
- **The cause is unproven.** The bad KV pointer only correlates with the lossy wire.
- The review independently identified the replicated draft head that skipped the rerank, which is already fixed
  above and confirmed on cards.

**Action (rad0016):**

- A `--rank-devices` world now refuses `--tp-wire wht6` at startup, naming `--tp-wire exact`.
- The exact wire is never substituted silently.
- Production sources are unaffected.

**Before lifting the gate**, instrument a run on hardware:

- KV operand provenance (resolver → `fill_paged_row`: rank, group, layer, base, and the extent of the allocation it
  resolves to);
- the block-table entry and KV head of the faulting tile;
- ti8 mailbox ordering.

## N5 first pass, 2026-11-12: context and resident KV (build a0979ca2eb38035b67618a59795a3103)

**Setup:** MXFP4, `--kv-cache-dtype fp8`, DFlash2 at depth 3, `--max-model-len 262144`. TP2 uses the wht6 wire
(production) and TP3 the exact wire with 6/6/12. Run under the lease; the run is not timed.

**Context.** Both configurations start at a 262,144-token context and answer a 26,624-token prompt correctly.

| measure | TP2 | TP3 |
|---|---|---|
| TTFT, 26,624-token prompt | 12.7 s | 20.0 s |
| decode after that prompt | 68.9 tok/s | 93.4 tok/s |
| KV per card for one 262k session | 4.16 GiB | 2.08 GiB on XTX ranks 0 and 1 |

The XTX ranks need half TP2's KV per card because each holds one KV head.

**Resident KV capacity** ("full-length sessions" as reported by the engine; tokens carved by TP3):

| sequences × 262k | TP2 | TP3 | TP3 tokens carved |
|---|---|---|---|
| 4 | 3.4 | **3.7 (+9%)** | 958,023 |
| 8 | 3.3 | **3.5 (+6%)** | 920,015 |

**What this shows**

- **The R9700 is the bottleneck.** One block table serves every rank, so the carve is the smallest rank's share:
  - the XTX ranks could page about 1.96M tokens each;
  - rank 2 holds 2 of the 4 KV heads, which is twice the KV per token, on 1.33× the VRAM and alongside the
    largest weight share.
- **Increased capacity:** met (modest: +6–9% resident KV).
- **Increased context length:** both reach 262k.

**Levers not tried yet**

- `--host-pool-mib`, so the cache can use host memory.
- A KV-light ownership profile. With 4 KV heads over 3 ranks, a split other than 1/1/2 needs KV replication.
- Long-context prefill is about 57% slower on TP3. This is the same x4-link limit seen in N0.

## N7 quality evidence, first pass, 2026-11-12: KLD against an exact TP2 reference (build a0979ca2eb38035b67618a59795a3103)

**Setup**

- **Corpus:** `rad-kld-corpus.jsonl` (24 documents, 42,761 scored positions).
- **Model:** the MXFP4 27B, `--max-model-len 8192 --max-num-seqs 4`, run under the lease.
- **Reference:** recorded on TP2 (two XTX, exact wire) with `--kld-record --kld-top 64`.
- **Runs scored with `--kld-ref`:** TP3 6/6/12 on the exact wire, and production TP2 on the wht6 lossy wire.
- **Outputs:** `vtp14/kld/{tp3,tp2wht6}.json`, with every position in `.rows`.

| against the TP2 exact reference | KLD mean | median | p99 | p99.9 | p99.99 | max | top-1 | PPL (ref 8.5620) |
|---|---|---|---|---|---|---|---|---|
| **TP3 6/6/12, exact** | **0.00307** | 0.00136 | 0.0227 | 0.084 | 0.445 | 3.62 | **96.80%** | 8.5564 (×0.9993) |
| TP2, wht6 (production) | 0.00391 | 0.00168 | 0.0281 | 0.111 | 0.980 | 6.41 | 96.57% | 8.5581 (×0.9995) |

**What this shows**

- TP3 changes the model's output distribution less than the production TP2 lossy wire does, at every percentile.
- The residual divergence is the expected effect of a different reduction order and partition over three ranks. The
  weights and kernels are the same MXFP4 container.
- This is equivalence evidence for the TP3 split, not a quality claim about MXFP4 itself.

**Still to do:** a BF16 reference (the original checkpoint, `qwen3.8-27b-bf16.rad`) to score MXFP4 TP2 and TP3
against original quality.

### N7: MXFP4 against the original BF16, 2026-11-12

**Setup**

- **Reference:** `qwen3.8-27b-bf16.rad` (the original checkpoint, 51 GiB), recorded on TP3 6/6/12. It fits only on
  three cards, not on two XTX.
- **Corpus and flags:** the same corpus and flags as above, with `--kld-top 64`.
- **Runs scored:** MXFP4 on each configuration, exact wire.

| MXFP4, against BF16 | KLD mean | median | p99 | p99.9 | top-1 | PPL (BF16 8.3366) |
|---|---|---|---|---|---|---|
| TP3 6/6/12 | 0.04351 | 0.02031 | 0.317 | 1.16 | 87.07% | 8.5564 (+2.64%) |
| TP2 | 0.04363 | 0.02029 | 0.314 | 1.26 | 87.09% | 8.5620 (+2.70%) |

**What this shows**

- MXFP4's divergence from the original model is the same on TP2 and on TP3. The split adds nothing measurable on top
  of quantisation (compare the 0.003 TP3-vs-TP2 KLD above with 0.044 for quantisation).
- **TP3 also makes original-quality BF16 serving possible**, at 23 tok/s single stream in the N4 smoke. The dual-XTX
  TP2 cannot hold it.

## N2 rebalance: per-rank timing, 2026-11-12 (build f1c9b765dbe4139fb17791b62313fa82)

**Question:** would a different split offset the timing mismatch between the three cards?

**Method:** `--profile-ops` on MXFP4 TP3 (one 256-token generation, DFlash2 at its default depth 7). The profile
sums device time per op and rank. Compute is everything except `all_reduce` and `all_gather`. The collective rows
include the time a rank waits for the slowest peer.

| profile | XTX rank 0 compute | XTX rank 1 compute | R9700 compute | R9700 in collectives |
|---|---|---|---|---|
| 6/6/12 | 3468 ms | 3452 ms | 2263 ms | 1702 ms |
| bandwidth-weighted (GDN 6/6/4, FFN 6656/6656/4096) | 3820 ms | 3821 ms | 1745 ms | 2438 ms |

**What this shows**

- Under 6/6/12 the XTX cards are the bottleneck, not the R9700. The R9700 holds about 45% of the weights, finishes in
  65% of the XTX time, and then waits in the all-reduce.
- The MXFP4 GEMMs are `mxfp4a8` (fp8 activations). gfx1100 has no fp8 and emulates it, so the XTX is compute-bound
  and the R9700 does about 2.5× the work per unit of time. Weighting by memory bandwidth, which moves work to the XTX,
  is the wrong direction (XTX compute +10%). That profile is dropped.
- That single-stream, depth-7 profile overstated the R9700's lead. The sweeps below decide the split.

### Split sweep, 2026-11-12 (build 751f23dc3bddff8c327f9dcb6e74d373)

**Setup:** custom splits `qwen38-27b:K0,K1,K2:F0,F1,F2` (delta-net key heads, value heads 3×, and feed-forward rows
per rank; attention 6/6/12 throughout).

**Profiled, 4 concurrent streams, speculation off.** Per-stream tok/s and per-rank compute (profiled timings are not
comparable with unprofiled runs, only with each other):

| split (K / FFN) | tok/s a stream | XTX compute | R9700 compute | R9700 in collectives |
|---|---|---|---|---|
| 4,4,8 / 5120,4864,7424 (6/6/12) | 39.3–40.1 | 5808 ms | 5304 ms | 1775 ms |
| 3,3,10 / 4608,4608,8192 | 37.8–38.5 | 5303 ms | 5939 ms | 1597 ms |
| 3,3,10 / 4096,4096,9216 | 37.3 | 5149 ms | 6126 ms | 1562 ms |
| 3,3,10 / 3584,3584,10240 | 35.5–36.0 | 4987 ms | 6351 ms | 1592 ms |
| 2,2,12 / 3584,3584,10240 | 35.1 | 4696 ms | 6727 ms | 1559 ms |

**Unprofiled, stream-bench (512-token prompts, 128 new tokens, 2 reps).** Decode step in ms (lower is better), with
aggregate tok/s in brackets:

| split (K / FFN) | off, 4 streams | off, 8 streams | depth 3, 4 streams | depth 3, 8 streams |
|---|---|---|---|---|
| 3,3,10 / 4096,4096,9216 | 21.51 (120.9) | **27.35** (160.3) | 37.35 (163.8) | 59.27 (181.2) |
| 4,4,8 / 5120,4864,7424 (6/6/12) | **19.55** (127.9) | 27.75 (157.4) | **34.72** (175.7) | **56.59** (187.0) |
| 4,4,8 / 5632,5632,6144 | 20.56 (126.4) | 29.24 (157.4) | 35.84 (172.7) | 58.67 (189.3) |
| 5,5,6 / 5632,5632,6144 | 20.57 (126.3) | 29.41 (156.9) | 35.11 (177.9) | 57.70 (194.4) |
| 6,6,4 / 6656,6656,4096 | 22.18 (123.6) | 31.85 (154.9) | 35.73 (180.3) | 62.71 (190.8) |

**What this shows**

- 6/6/12 has the shortest step in three of the four regimes. It is 1.4% behind the best in the fourth, within noise.
  No split is faster in a way that holds across regimes. With speculation on, aggregate tok/s also moves with draft
  acceptance (each split rounds slightly differently), so the step time is the cleaner comparison.
- In batched decode, the R9700's compute about equals the XTXs'. It also carries about 1.6 s of collective time that
  does not shrink as its share shrinks: that is the x4 link (PCIe Gen4 x4 root port) in the 129 all-reduces a step.
  Moving work either way trades one bottleneck for the other.
- The bandwidth-weighted and R9700-heavy named profiles are dropped; 6/6/12 stays the profile. The custom form stays
  for formats with different per-card costs (w8a8 runs int8 natively on both cards).
- The remaining lever is the link: an x8/x16 CPU-attached slot for the R9700, or fewer all-reduce round trips per step
  (fusing the attention/feed-forward pairs).

## N6: int8 (w8a8) on TP3, 2026-11-12 (build 751f23dc3bddff8c327f9dcb6e74d373)

**Setup:** `qwen3.8-27b-w8a8-df2.rad` (`w8a8_fwht128`, served by `qwen35_w8a8`, which is `qwen35_fp8.cpp` built with
`QWEN35_W8A8`). rad0019's opt-in and spans cover it without new code: the int8 linear takes row spans as is and column
spans in whole 128-column scale groups, which every split here keeps. Smoke: TP3 6/6/12 and TP2 give the same greedy
answer.

**Split sweep (stream-bench, 512-token prompts, 128 new tokens), aggregate tok/s:**

| split (K / FFN), w8a8 | off: 1 / 4 / 6 / 8 streams | depth 3: 1 / 4 / 6 / 8 streams |
|---|---|---|
| 4,4,8 / 5120,4864,7424 (6/6/12) | 36.1 / 101.9 / 127.4 / 142.2 | 68.7 / 151.0 / 166.6 / 176.8 |
| 5,5,6 / 5632,5632,6144 | 39.1 / 112.1 / 137.6 / 157.3 | 81.8 / 164.0 / 173.6 / 186.2 |
| 6,6,4 / 6656,6656,4096 | 38.2 / 111.1 / 138.6 / 160.7 | 80.7 / 174.6 / 164.9 / 189.7 |
| 3,3,10 / 4096,4096,9216 | 31.8 / 91.0 / 113.4 / 129.1 | 71.5 / 143.0 / 159.7 / 165.3 |

For int8 the best split is the opposite of MXFP4's: more delta-net and feed-forward work on the XTX cards. `i8a8` runs
natively on gfx1100 and gfx1201, so memory bandwidth (960 against 640 GB/s) and the R9700's link set the balance.
6/6/4 with 6656/6656/4096 is +13% at 8 streams with speculation off and +7% with it on, over 6/6/12.

**Matched ABBA (6/6/12 split), aggregate tok/s, mean of the A and B pairs:**

| arm | off: 1 / 4 / 6 / 8 | depth 3: 1 / 4 / 6 / 8 |
|---|---|---|
| w8a8 TP3 6/6/12 | 36.0 / 101.6 / 127.7 / 143.6 | 68.8 / 149.4 / 168.1 / 177.0 |
| w8a8 TP2 | 32.9 / 99.6 / 129.3 / 149.9 | 65.6 / 173.7 / 168.9 / 194.8 |
| MXFP4 TP3 6/6/12 | 54.5 / 128.9 / 147.2 / 158.4 | 104.3 / 173.6 / 176.3 / 185.6 |

With its own best split (single sweep run, not an ABBA), w8a8 TP3 is +16% / +12% / +7% / +7% over w8a8 TP2 with
speculation off, and +23% / 0% / −2% / −3% with it on. MXFP4 stays faster per token, carrying half the weight bytes.

**Quality against the original BF16 (same corpus and reference as N7):**

| model, split | KLD mean | median | p99 | p99.9 | top-1 | PPL against BF16 |
|---|---|---|---|---|---|---|
| w8a8 TP3 6/6/12 | 0.00199 | 0.00031 | 0.00646 | 0.0430 | 97.32% | +0.03% |
| w8a8 TP2 | 0.00126 | 0.00030 | 0.00620 | 0.0431 | 97.33% | +0.03% |
| MXFP4 TP3 6/6/12 | 0.04351 | 0.02031 | 0.317 | 1.16 | 87.07% | +2.64% |

w8a8 is about 20 times closer to the original model than MXFP4. The TP3 mean is above TP2's only through a few positions
in the far tail (p99.99 2.64 against 0.44); the median, p99 and p99.9 match.

## Pipelined prefill at TP3 (rad0020), 2026-11-12 (builds 751f23dc without, a541fb4a with)

qwen35_fp8 pipelines a prefill step of at least 1024 rows of one sequence: two pieces, each block's all-reduce for one
piece on the second lane beside the other piece's work. Upstream enables it at two ranks only. rad0020 enables it at
three under an ownership profile. MXFP4, 4096-token prompts, 32 new tokens, 2048-token chunks, ABBA:

| arm | TTFT, 1 stream | TTFT max, 4 streams |
|---|---|---|
| TP3 6/6/12, serial (without rad0020) | 1392 / 1398 ms | 5769 / 5892 ms |
| TP3 6/6/12, pipelined (rad0020) | 1324 / 1329 ms | 5763 / 5655 ms |
| TP2 (pipelines upstream) | 901 / 903 ms | 5088 / 5069 ms |

- Single-stream TTFT is 5% shorter with the pipeline. With several sequences in a step the pipeline does not engage
  ("more than one sequence"), on TP2 as on TP3.
- TP3 prefill stays 47% slower than TP2. Prefill moves about 10 MiB per all-reduce, and the R9700 receives two partial
  sums over its Gen4 x4 link. Overlap hides part of that; a reduce centred on the XTX cards (the R9700 sending and
  receiving once) or a compressed wire would cut the bytes themselves.

## Generic split spec and pipelined-prefill options (smoke)

The named profiles (`qwen38-27b-6-6-12`, the int8 profile and the `qwen38-27b:K:F` form) are gone. `--tp-ownership`
now takes a spec, one count a rank: `kv=..;ffn=..[;q=..][;lk=..;lv=..]`, checked against the loaded model at build
(rad0018/rad0019). No model or split is built into the patches. The pipelined prefill is configuration:
`--pipelined-prefill auto|on|off` and `--pipelined-prefill-min-rows N` (rad0020).

Smoke on the cards (three ranks: XTX, XTX, R9700; one greedy request each):

| arm | spec | result |
|---|---|---|
| w8a8 | `kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24` | served; output byte-identical to the named 6/6/12 profile |
| w8a8 | `kv=1,1,2;ffn=6656,6656,4096;lk=6,6,4;lv=18,18,12` | served, coherent |
| MXFP4 | 6/6/12 spec, `--pipelined-prefill auto` | served, coherent |
| w8a8 | ffn shares summing to 16992 | refused: shares must sum to the model's heads and 17408 rows |
| w8a8 | lv 12,16,20 (sum right, V:K wrong) | refused: each rank's value heads must be 3 a key head |

Layouts are not added to `layouts.toml`: run definitions with model paths are to be external (pending the user's
decision on the external-layout change).

## One build, configuration as data

- Layouts are run-time data (`--layouts` / `BIGCHERRY_LAYOUTS`; the repository's examples in `engines/radiance/layouts/`),
  never compiled into `bc-serve`, and `layout.name` has no default. A layout's engine may be a list of `{ arch, count }`
  groups in rank order (a mixed world), and a `[tuning]` table sets engine tuning fields.
- Contract amendment 18: `tuning.tp_ownership`, `tuning.pipelined_prefill`, `tuning.pipelined_prefill_min_rows`, and
  `device.rank_devices` (derived from the engine groups of a mixed world).
- The TP3 setups are layouts, `qwen27b-tp3` (MXFP4) and `qwen27b-tp3-int8` (w8a8): on this host `bc-serve` resolves them
  to the cards, rank order, kernels and split used by hand above.
- The VTP14 patches (rad0016-rad0020) are one patch set in the main `radiance` build; the separate source is gone.

Inertness of the fold (no VTP14 option given), ABBA on two XTX, `--tp 2 --tp-wire wht6`, spec off, main `radiance` lane at
the commit before the fold (base) against the fold:

| model | greedy text | 1 stream tok/s base / fold | 4 streams tok/s base / fold |
|---|---|---|---|
| MXFP4 | identical (4 arms) | 45.05 / 44.96 (−0.2%) | 117.39 / 118.14 (+0.6%) |
| w8a8 | identical (4 arms) | 33.37 / 33.41 (+0.1%) | 105.48 / 105.63 (+0.1%) |

On the same main build, the TP3 smoke repeats: w8a8 6/6/12 output byte-identical to the original TP3 run, the int8 split
and MXFP4 serve, and a spec that does not match the model is refused.

## TP3 from the layout alone (launcher, manifest)

`launch-flashnext-best.sh start --layout qwen27b-tp3` (and `qwen27b-tp3-int8`): the launcher resolves the layout, `bc-serve`
writes the unmeasured manifest, the engine's measuring start plans memory, `bc-serve` writes the measured manifest (a second
round with the 3449 MiB host pool it derived), and the engine starts from the manifest alone. No engine flag was written by
hand. The engine's option report shows every TP3 setting arriving from the manifest: `--tp 3`, `--kernels libr3,libr4d,libref`,
`--rank-devices` (03, 06 gfx1100; 09 gfx1201), `--tp-ownership` (the layout's settings), `--pipelined-prefill auto`.
Greedy output is byte-identical to the hand-flag runs: MXFP4 6/6/12 and w8a8 6,6,4.

Two refusals on the way, both in the measured-manifest path and blocking every dense engine-only layout (TP2 included),
fixed generically: the P1 producer required routed experts (C23), and the retention target had to be positive though an
engine-only layout has no host KV tier (C24, contract amendment 19).
