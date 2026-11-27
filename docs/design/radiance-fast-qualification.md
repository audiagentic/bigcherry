# Radiance fast scheduler and model-layout qualification (VTP60)

Status: implementation design, 2026-11-27. VTP59 owns the current 2.2 regression and production tuning; VTP60 owns the reusable fast qualification integration. Related: [kernel group qualification](radiance-kernel-group-qualification.md), [bump integration](radiance-bump-integration.md), RCS25–28, VTP42, MRC08 and MEN17. Do not modify competing plan items, add a second scheduler, fork a second workload catalogue or introduce a second evidence store. The existing release gate remains authoritative.

## 1. Evidence and constraints

VTP59 host-burst proof (1c8af7595) runs 62 scenarios/18,157 checks in 0.08 s after ~5.4 s compile/link against an existing HIP-off host build. For eight 4096-token prompts and 128 outputs, auto:1 takes 201 decode-containing steps with peak five decoders; 0 + 250 ms takes 79 with peak eight. Both complete 1024 output tokens. Synthetic timing reproduces a structural failure mode, not measured tok/s. Real TP2/TP3 2.2 measurements are in docs/evidence/VTP59-scheduler-20261127/. Faster decode can coexist with worse inter-event latency or lower end-to-end throughput. No current configuration is fully qualified.

The proof includes sched_test.cpp and links an existing archive via compile_commands/link.txt. Its source/test/archives are hashed but not proven to be one fresh composed build. Its hard-coded timings, fake KV, ideal draft acceptance, request geometry and explicit policies are insufficient for release acceptance. It remains a lab diagnostic until the gates below are implemented. No production config or default is changed by this design.

## 2. Delivery slices and ownership

| Slice | Existing seam | Deliverable / exit |
|---|---|---|
| S1 | upstream tests/sched_test.cpp, sched_fakes.h, rad_test.h, tests/CMakeLists.txt; BigCherry patch/build/compose workflow | Native host target built from exact composed core with fixture; no copied scheduler; test-only patch or supported CMake inclusion; candidate identity verified |
| S2 | native_fleet_config.cpp/test, manifest_writer_v3_test, core/config.cpp, core/engine_bringup.cpp; workloads.toml/workloads.py | Shared effective default resolver and versioned workload descriptors; deterministic scenarios with hard invariants and comparative budgets |
| S3 | unit_harness.py/unit_families.toml, compose_check, RCS26–28, rad-kbench, kbench_step.py, rum09_phase_timeline.py | Affected family and exact row selection; measured cost replay inputs with uncertainty and coverage |
| S4 | core/run_records.py, calibration_records.py, bench/workloads.py, MRC08 export, campaign/acceptance.py, release/pin_bump.py | Typed evidence receipts, fail-closed comparisons and routing to existing bump gate |
| S5 | upstream sched/runtime/config tests, existing campaign and serving harness | Mutation controls, baseline/candidate repeatable screens, held-out hardware correlation, final integrated checks |

S1/S2 are independent of MRC08 merge and GPU availability. S3 uses existing RCS routes only; uncovered forms stay explicit. S4 consumes the MRC08 contract when available; no parallel exporter while that work is pending. S5 consumes existing MEN17 acceptance, not a new engine qualification workflow. VTP59 can use the lab screen now; completion of broad future coverage is not a prerequisite for individual kernel changes.

## 3. Native CPU gate

Promote the workload cases into the native scheduler test target using the smallest supported test-source composition seam. Build only sched_test/config_test and their actual dependencies from the composed source with RAD_WITH_HIP=OFF, RAD_BUILD_TESTS=ON, test-only isolated scratch. The production recipe keeps RAD_BUILD_TESTS=OFF. Preserve upstream compiler flags and weak-fake checks; never link a pre-existing archive just because its pathname matches. Reuse exact verified build outputs; cache key covers source pin, composition, recipe, compiler, flags, ABI, fixture, dependency digests and target. A clean source or archive mismatch invalidates the receipt. Do not parse generated Makefiles as the permanent build API; the lab launcher is temporary. Assert the device fakes are linked, GPU visibility hidden, no weight/container loads, bounded steps and zero vacuous cases. Select tests by explicit case names; a missing or skipped required test is nonpass.

Use the real Scheduler::step/plan/commit and the existing note_step_cost and note_mixed_step_cost hooks, mirroring step_finished semantics and order. The latter reads the still-valid plan; test that the timing observation is attributed to the completed step exactly once. Feed simulated time only into existing host seams (age/cancellation require injectable monotonic clock or deterministic existing interface, not sleep). Record per-step phase, sequence ids local to scenario, context, prefill/decode rows, n_spec, accepted output and terminal reason. Check each sequence's input/output conservation, bounded progress/no deadlock, no duplicate admission, slot reuse, cancellation cleanup, no unintended preemption and peak/mean decode occupancy. Separate policy properties from performance comparisons: fixed synthetic controls include explicit auto:1 and interleave 0, and the measured 250 ms candidate; a good relative result alone cannot qualify.

## 4. Effective policy contract

Native config parsing records cfg.given; core/engine_bringup.cpp later sets omitted interleave to auto:1 except deterministic/KLD/generator paths. A YAML diff or config_test alone cannot detect that drift. Extract the minimal pure resolver from actual bring-up (or exercise it with existing fakes), return value, origin (explicit, inherited, automatic), rationale and input capabilities. Call this same resolver in production and host tests; keep no Python duplicate default table. Test omitted, explicit zero, fixed, auto shares, deterministic/KLD/generator, invalid value, model capability refusals and precedence. Keep native manifest provenance, compare fully resolved effective values and explain/actual launch readback. Cover interleave, mixed-step target, busy chunk, max batch, max sequences/context, checkpoint/quantum, prefix/resume, speculative policy/depth, pipeline/min rows, KV dtype, wire and collective policy; dynamic hardware-dependent fields remain unresolved until observed, not silently defaulted. Classify expected changes with reviewable rationale; unexplained drift blocks candidate approval even if tests pass. Do not force one global default.

## 5. Workload selection

Extend existing versioned bench/workloads.toml + workloads.py schema/validator with scenario attributes and a host adapter; keep current serving definitions backward-compatible. Same workload family key but explicitly different realization/evidence kind for synthetic token ids vs tokenizer-calibrated text, so not comparable as real measurements. Derive exact B-1/B/B+1 from actual program/selected kernel geometry and scheduler quantum/page/checkpoint/budget; no fixed 128/2048 assumption for all models. Keep a curated sparse screen plus affected boundaries rather than full Cartesian product. Explicit invalid requests test refusal, not throughput. Use actual per-request context capacity and output reserve.

| Class | CPU first | GPU confirmation |
|---|---|---|
| Baseline | 1/2/4/6/8 streams; 64/128/256/512/1024/2048/4096/6144 prompts; 32/128/512/1024 outputs, sparse pairwise | 512/128 and 4096/128+512 representative control |
| Geometry | B-1/B/B+1, mixed decode and prefill, ragged rows, per-rank selected geometry | affected unit/family replay with actual rows and operands |
| Arrival | burst, staggered, short arriving behind long, sustained decode, cancellation and completion | selected mixed arrival and steady latency |
| Capacity | max sessions+1, near context, preemption, shared/independent prefix, cold/warm, slot reuse | targeted real allocation and memory pressure |
| Long | 8k to 100k within admitted max, FP8/BF16/BF16i8, checkpoint and age control | existing VTP42 policy controls; no unqualified change |
| Model/layout | IQ8R TP2 WHT6/exact, TP3 exact/pipeline, IQ6R V4, FlashNext IQ4R/HCBF16 helper/native, single-card control | only affected compatible class and mandatory integrated witnesses |

Use fixed seed, exact synthetic lengths and independent token sequences, measured or bounded acceptance patterns (zero/partial/full), explicit depth and cache state. Arrival uses simulated timestamp not scheduler wall time; stable tie-break. Screening must expose worst-case and distribution, not just one mean. CPU fakes do not attest real KV sizing/layout, arithmetic, driver, DMA, graph replay, helper placement or output quality. Capture real end-to-end tests for these.

## 6. Cost replay and grouping

Use canonical r3_groups.tsv/classifier, composed dependency closure, origins.toml, unit_families.toml and existing model_graph_scope/resolve_rows. Join exact provider, arch, layout, geometry, rank, phase, draft route, and actual issue evidence; declared /graph is not actual execution. Expand changed shared headers and unknown coverage conservatively; compile scope may widen, missing oracle is nonpass. Support-unit compilation is not qualification. No new group taxonomy. RCS26–28 own coverage and same-ABI reference, independent byte/oracle checks, resource and cold/warm replay.

Define a typed cost input referenced by content digest in existing run-record provenance: phase (pure prefill, pure decode, mixed), rank/layout/topology and runtime identity; actual rows, number of sequences, context band, KV and speculation, cache state, trace mode, clock/power, samples and bounds. Prefer uninstrumented measured step costs or native scheduler diagnose; phase traces identify critical path but perturb timings. Existing kbench_step.py aggregates row time from graph and timed kernels, not full step latency. Use only as component-cost attribution; never sum overlapped kernels/ranks and call it real step time. Communication/helper/host/launch costs and mixed overlap must be accounted for by measured step brackets or explicitly unknown. Exact cost cells first; bounded interpolation only within validated geometry/provider/phase, never through dispatch changes, no extrapolated qualification. Unknown/uncertain cost -> diagnostic/incomparable, request smallest missing measurement. Use median and dispersion, test sensitivity at lower/upper cost and acceptance bands. Reuse unaffected costs only with identical relevant identity and dependency closure. A changed policy may alter which rows are reached; resolve again, no old-trace forced replay as sole coverage proof.

Separate deterministic structural invariants (hard), cost-model screening rankings (diagnostic), isolated GPU oracle/replay (qualified for its scope), and integrated real serving (release authority). Hold out measured serving cohorts before fitting, compare direction, decode overlap and per-request TTFT/gap distributions. If model fails to discriminate known bad auto:1 from tuned control on held-out cohort, mark cost screen uncalibrated. No fabricated speedup predictions. Rank only nondominated choices across throughput, TTFT, gaps and fairness; if rank flips within uncertainty, narrow measurement. Do not change production settings automatically.

## 7. Receipts and bump integration

Use existing platform run-record store, metric specs and MRC08 allowlisted export. Native CTest logs/receipts remain build evidence until the run-record schema can represent host screening without a fake model/build or fabricated throughput. Use explicit kind/validity, scenario digest and verified source/build/fixture/cost-input digests; store raw trace in existing content-addressed blob store, portable summary only. Schema v1 requires real identity/model/layout and decode/prefill/TTFT groups; do not manufacture missing values. Add a backwards-compatible typed extension or separate existing evidence role via MRC08 owner; no second DB/record writer. Comparability uses same scenario realization and cost assumptions, runtime identity changes are expected comparison axis, not a license to match different topology, power, workload or quality. Missing identity, required cases, samples or schema -> INCOMPARABLE/FAIL, never pass. Failed/aborted evidence preserved and exports retry idempotently. No host paths, raw text, device locator or URLs in tracked summary.

MEN17 currently calls acceptance.accept(repo_root, layout, None, out_dir) without explicit candidate; fix there and require independently verified baseline, not same-run self-reference. Add fast gate to existing pin-bump compile/CPU gate before expensive hardware, reuse exact candidate artifact identities and required scope from existing triage. Run all scheduler/config invariants cheaply for every bump; affected families only for ordinary kernel changes. Full release build, existing CPU smoke, representative installed end-to-end performance and independent quality remain at promotion; no broad full-engine builds for isolated tests. Explicit skipped/deferred gates recorded as not qualified. Reuse existing lease, work-root, artifact and recovery rules; CPU compile nice, bounded GPU leases only. No power-state changes here.

## 8. Validation and acceptance

1. Native tests compile from clean candidate source and run on HIP-off host. Required cases cannot disappear or skip; test fakes linked. Test source/core archive mismatch, stale cache, changed compile flags, missing and duplicate result, malformed/negative cost, unrecognized geometry and incomplete coverage. Prove no GPU or model launch. Target steady-state host runtime < 2 s for curated suite (initial evidence 0.08 s; not a hard speed gate on loaded hosts); track compile separately.
2. Negative controls: omitted auto:1 drift vs explicit policy; known 8x4K overlap loss (201 vs 79 steps in PoC); injected stalled admission, duplicate slot, wrong acceptance/count, cancellation leak and missing boundary; demonstrate failure using targeted mutation and positive control, restore fixture. Check per-request progress, work and correctness, not only aggregate; predeclare class-specific allowable overhead and latency budgets before A/B. Include good and bad cases so fastest but unfair scheduler cannot pass.
3. Cost join tests: exact key, incompatible ABI/provider/rank/cache/shape, missing phase, duplicate row, unsupported and unseen geometry, trace-vs-untraced mismatch, overlap double-count, noise interval crossing. Require calibrated held-out cohorts and evidence of selected row; no invented fallback.
4. Plan-driven selection tests: affected kernel with valid oracle -> only family; missing owner/unknown header -> widen or block; scheduler-only -> native host tests and targeted witnesses, not full kernel fleet; new model/layout -> scope and unresolved receipt. Test unsupported device/reference, skipped replay and failed independent oracle.
5. Record and bump integration: identity mismatch, stale build, failed/skipped/synthetic-only and incomplete cohort reject; export privacy, idempotence, interrupted and retry; candidate exact runtime and independent baseline verified; prove no GPU launch on failing host gate. Compare historical same model/layout where available; if unavailable retain explicit blocker. Use matched fresh-start ABBA for finalists, mixed arrivals, longer output, quality controls and real latency. MRC08 and MEN17 remain owners.
6. Demonstrate the workflow on current released 2.2 and 2.3 candidate with at least one affected IQ8R TP2/TP3 cohort, IQ6R and FlashNext representative classes plus affected long-context controls. VTP59 owns wider production qualification. Keep previous releases and rejected arms; close only validated slice, do not mark VTP59 complete based on host screen.

## 9. Review disposition

The review found stale-core provenance, default resolution after parsing, unrealistic cost aggregation, declared-versus-executed row confusion, insufficient workload diversity, and missing candidate binding. Sections 3–8 explicitly address these. Still open: implement native integration, validate exact build/clock and cost ingestion, independent evidence and run-record contract, actual model quality and final serving. This is a reviewed design, not implementation approval. Runtime cost fitting and final latency budgets must be measured; do not promise quantitative performance or fixed time savings beyond PoC evidence.

## 10. Native-first consolidation and deterministic tuning (owner direction)

Supersedes the choice of exploratory harness in sections 2, 3 and 6, not their qualification requirements. Source reviewed: Radiance 2.3 pin 3b13fd32663ef52252a7f0be255523480aa22ffe, composed slice 90bca121c9dcee4cac6cf2b41a8d7b3e.

| Capability | Native authority | BigCherry responsibility |
|---|---|---|
| Kernel variant tuning | tools/rad_tune.cpp; Program::tune_requests; format/tunecache.h | Resolve model/layout and affected groups, capture medians and dispersion, qualify and retain native cache |
| Correctness and replay | rad-kbench, kfixture, kbench_core | Independent reference, family selection, real coverage and resource gates |
| Graph and config inspection | rad-info, rad-yaml, config_test, native bring-up | Exact effective settings, model/layout and provenance comparison |
| Real step and serving costs | radiance --diagnose, existing step/phase trace | Capture matched phase/rank/topology, workload and power; correlate with serving |
| Scheduler simulation | tests/sched_test.cpp: sim_burst_of_long_prompts | Minimal extension, deterministic policy search, regression invariants |
| Evidence and promotion | Existing RCS26–28, MRC08 and MEN17 | Selection, records, independent reference and release decisions |

rad-tune is not a serving-policy tuner: it searches declared legal kernel variants, not interleave, admission or mixed latency. Keep isolated replay/independent correctness even when a native tune cache selects a faster variant. Do not assume declared shapes were issued. Verify native tool capability for each quantization, KV, helper and architecture combination; uncovered forms require explicit scoped evidence, not a new generic tuner.

Native sim_burst already supplies seeded acceptance, burst/pre-existing decoder inputs and configurable linear costs. It calls note_step_cost but does not currently feed note_mixed_step_cost or an injected age clock. Extend that existing test path, not the scheduler logic in a second simulator. Add structured result/trace output, a bounded complete-run invariant, input validation, per-request deterministic acceptance keyed by request and generation position, and explicit phase costs. Use an injectable monotonic clock only where scheduling decisions depend on time; do not replace real production telemetry. Inject each observation exactly once at the native completed-step boundary. Preserve existing SIM_* compatibility and upstream tests. Prefer a small upstreamable change; if native upstream adds equivalent support, retire the local extension after parity.

Freeze scenario, seed, cost table digest, policy, geometry, capacity and arrival order. Deterministically enumerate a bounded set of legal policies with stable ordering and tie-breaks. Hard reject invalid requests, missing cost coverage, deadlock, starvation, conservation/quality failures or latency-budget violations. Compare nondominated choices on aggregate throughput, TTFT, gap tails and fairness; uncertainty bands and held-out measured cohorts determine whether a policy merits hardware confirmation. Record schedule digest over ordered steps, per-sequence IDs/progress/acceptance/completion and controller decisions; exclude wall-clock/log paths. Require repeated identical digests and metrics for the same inputs, including with different host load. Changing the cost/acceptance profile is a separate cohort. Native simulator estimates are not measured tok/s and cannot promote production settings.

Implementation status: 724ebe40c replaced the old manual link launcher with attested snapshots and native CTest. Seven cases/18,221 checks in 0.21 s on restored 2.3 host build, 31 Python tests and default mutation rejected. Existing scheduler assertions remain until upstream equivalents prove same coverage. Fresh combined launcher validation, complete profile audit, deterministic simulator extension, cost ingestion and MRC08/MEN17 integration are open. No production configuration changed. See docs/evidence/VTP60-native-host/summary.json.

## 11. Native tuning on heterogeneous TP and KV formats

Reviewed the composed 2.3 candidate (slice 90bca121c9dcee4cac6cf2b41a8d7b3e).
rad0071_native_kv_tools shares core/mem/packed_kv.h with both native tools;
it compiles but is not yet promoted. Existing KV layout descriptors in
abi/rad_builder.h are the authority, not a separate list of quantisation
formats. Reuse ENG04/ENG06 format and retrieval gates and RSC12 profiles.

**Do not publish a TP3 cache from an unqualified rad-tune run.** Three
concrete gaps in the composed source:

- tools/rad_tune.cpp declares the requested TP width but omits the engine's
  ownership split, pipeline/min rows, FP8 attention product selector and
  effective max context. core/engine_bringup.cpp Engine::build_ctx is the
  actual authority. Factor out its pure config-derived fields rather than
  replicating them. Preserve runtime-only fields (slot, calibration,
  controls, streaming capabilities) as explicit inputs/refusals. Prefer the
  existing native fleet/config reader for the model selection; compare
  effective declared graphs and source provenance.
- rad-tune iterates progs[0].tune_requests, and noncollective launches use
  rank zero. For the XTX/XTX/R9700 deployment, that can miss different
  rank geometries and kernel providers. Enumerate the union of per-rank
  Program::TuneRequest records; measure noncollective work on its actual
  rank, and group collective requests by the existing instance and
  compatible geometry. A collective lacking a matching peer is uncovered,
  not a valid timed case. Use native rad-kbench recorded fixtures for
  independent correctness on every affected rank/provider/geometry.
- core/format/machine_key.cpp identifies device 0 and visible device count,
  not the whole heterogeneous topology; TuneCache currently keys on kernel,
  plugin and geometry. A shared provider/key can collide if two ranks
  require different optimal axes. Initially keep caches isolated, and
  admit a shared row only when every rank that consumes it is validated
  for that choice. For real collisions extend the native key and lookup
  together (backward-compatible reader, explicit ambiguity refusal),
  never invent a BigCherry tune cache. Validate reordered devices,
  homogeneous vs heterogeneous topology, identical and differing
  geometries, pin precedence (RADIANCE_TUNE / tune-pin), stale source and
  plugin changes. Only publish after exact cache readback in engine and
  independent quality/serving gates. Model format and runtime identity
  belong in existing MRC08 evidence, not in an unrelated store.

Performance attribution: use matching prompt hashes and actual token
counts, repeat fresh starts in ABBA order, report both step latency and
tokens per step (draft acceptance can change independently), wall output
rate, TTFT, gap tails and actual prefix hits. Preserve cold prefill vs
cached workload as separate cohorts. Measure 16/8 against FP8/BF16 over
context lengths that physically fit, then I8/V4/K6V4 with quality and
retrieval controls. Compare memory at a common page/token budget before
claiming capacity gains. Reject unsupported geometries before timing.

Packed KV correctness is a separate required gate, not supplied by accepting
a CLI string. The current libref/ref_attn.cpp explicitly returns
RAD_E_UNSUPPORTED for packed layouts; ref_shapes.cpp's paged recipe
uses native 2*hd, and tools/opshapes.cpp has the same native fallback.
libr4d/r4d_shapes.cpp already computes actual packed storage extent via
rad_kv_group_row_elems. Prevent reference-shape fallback from passing an
incorrect native extent to a packed kernel. Reuse that ABI row descriptor
for physical extent, build finite legal packed fixtures using independent
known source K/V and validated packers, and compare decoded attention
against the independent dense/native reference (not the tested GPU
packer/dequant as its own oracle). Inspect the existing device selftests
and packed replay fixtures before adding a new helper. Record uncovered
formats explicitly if an independent oracle/fixture does not exist.
A tune candidate agreeing with its own default is insufficient quality
evidence. Native tools must refuse or skip these unsupported cases with
coverage accounting; no random opaque packed bytes as numerical evidence.
