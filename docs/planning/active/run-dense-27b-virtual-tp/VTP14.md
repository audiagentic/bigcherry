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
