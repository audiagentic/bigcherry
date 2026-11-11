# BigCherry code-reduction architecture audit — 2026-11-11

**Scope:** offline, card-free research only; no GPU, lease, vendor modification, code change, or generated-file edit.

**Snapshot:** `origin/main` at `c625a9b8805695de11a234987491a2f0725aefd5`. All line numbers refer to this commit, *not* an in-flight RSC11 branch.

**Decision:** remove *duplicated behaviour* before splitting large files. A module split is not a line reduction; a central abstraction that conceals patch drift or GPU launch semantics is worse than repetition.

## Measurement and reproducibility

Counted tracked files using `git ls-files -z` and physical lines using `splitlines()` (equivalent to `wc -l` for these newline-terminated sources). Scanned `tools/bigcherry/`, `engines/radiance/`, and `engines/llamacpp/`; excluded vendored, third-party, generated, and fixture content, including the serving generated registry. The survey covered **736 tracked source files** in those roots (5,078 tracked files repo-wide). The largest relevant hand-written sources are:

| File (all paths relative to repository root) | Lines |
| --- | ---: |
| `engines/llamacpp/overlay/ggml/src/ggml-cuda/hip-autotune-tuner.cu` | 4,095 |
| `tools/bigcherry/experiment/contract.py` | 3,252 |
| `engines/llamacpp/overlay/ggml/src/ggml-cuda/hip-autotune-dispatch.cu` | 2,713 |
| `tools/bigcherry/patch/campaign/producer.py` | 2,530 |
| `tools/bigcherry/release/pin_bump.py` | 2,480 |
| `tools/bigcherry/cli/main.py` | 2,375 |
| `tools/bigcherry/tuning/inventory.py` | 2,169 |
| `engines/radiance/components/serving/contract/registry.cpp` | 1,928 |
| `tools/bigcherry/tuning/catalog.py` | 1,878 |
| `engines/radiance/components/moe-remote/moe_remote_gpu.h` | 1,848 |
| `tools/bigcherry/patch/evidence.py` | 1,788 |
| `engines/radiance/components/moe-remote/moe_remote_epoch_test.cpp` | 1,747 |
| `tools/bigcherry/patch/rebase.py` | 1,736 |
| `engines/llamacpp/patches/1361_prefill_stagger_pair/patch.py` | 1,731 |
| `engines/radiance/components/serving/serving_test.cpp` | 1,723 |

Used `ast` to measure Python function line spans, imports, direct named calls and patch constructors; compared Python functions of at least 20 physical lines / 70 tokens with **12-token normalized shingles** (comments/whitespace omitted, names and literals normalized; candidate threshold at least 12 matching shingles and overlap >=0.58 of the smaller set). These are **candidate clones**, not proof of semantic equivalence. Used **eight-line exact whitespace-normalized noncomment C++/CUDA shingles**, rejecting adjacent same-file windows, to find conservative clones. Python AST import resolution is syntactic only; dynamic imports, C++ macros, conditional compilation and runtime dispatch are not resolved.

AST facts illustrating *mixed responsibilities*, rather than merely big files: `cli/main.py:100-2098` is a **1,999-line** `build_parser` (81 `add_parser`, 363 `add_argument`, 71 `set_defaults` calls across the file). `tuning/inventory.py:879-1821` is a **943-line** `load_measurements` combining JSONL parsing, trust/attestation, row resolution and SQLite writes. `experiment/contract.py:953-1465` is a **513-line** `parse_contract`, directly calling `_required_string` 14 times and `_table` 13 times. `pin_bump.py:1970-2351` contains a **382-line** `_run_phases`. `patch/campaign/producer.py` imports build, core, experiment, patch activation, benchmark, scaffold, trace and validation owners in lines 15-55 and has a 171-line `_run_validation_producer` at 2295-2465.

Specific clone candidates with locations: `patch/evidence.py:1697-1732` vs `:1735-1766` (normalized shingle overlap **0.92**, qualification loops); `:260-309` vs `:338-407` (**0.87**, campaign identity digests, but *different input-binding semantics*); `experiment/contract.py:1553-1583` vs `:1586-1622` (**0.91**, known-ID registry parsing); `patch/rebase.py:237-293` vs `:334-408` (**0.69**, selector shape vs resolved snapshot). The cross-file `cli/main.py` / `inventory.main` clone suggestion was rejected as generic argparse scaffolding. For C++, there are **34 overlapping eight-line same-file matching windows** in `moe_remote_gpu.h` (sample `:789-796` vs `:811-818`, repeated kernel launch arguments), **7** in `hip-autotune-dispatch.cu` (sample `:1644-1654` vs `:1746-1754`, launch-context assembly), **2** in the tuner and **1** in the serving registry. No qualifying *cross-file* eight-line exact clones were seen among the six sampled C++/CUDA files. The window counts are *not* additive removable lines.

## Ranked top ten: **net** lines worth removing

Estimates are engineering ranges, *not measured deletions*. They discount relocated lines and generated outputs; avoid summing repeated opportunities in the same file. Test names are gates to preserve, not a claim that a card-free run proves GPU correctness. **Total: 1,370–2,360 net handwritten lines (planning midpoint ~1,865)** across these ten; only the first item concerns the full patch fleet.

| # | File/area, current lines and evidence | Recommended pattern | Net removable; risk | Protecting tests / acceptance |
| ---: | --- | --- | --- | --- |
| 1 | **Patch authoring:** llama.cpp `patch.py` **38,570**, Radiance **3,254**; 974 + 80 direct `Edit` constructors, repetitive anchor/guard/rationale/expectation syntax; `1361_prefill_stagger_pair/patch.py:12-22` has a local `_edit` helper | Small **canonical** `bigcherry.patch` authoring builders for one-match edits and typed target specs; retain explicit guards and rationales; declarative metadata, never another patch engine | **600–1,000; medium–high** (anchor drift) | `tools/tests/patch/test_patch_rebase.py`, `test_patch_validation_evidence.py`, `test_0300_mmq_forced_j.py`, `test_1263_prbe41_ssm_conv_channels_major.py` |
| 2 | **Experiment contract** `contract.py` **3,252**; `parse_contract:953-1465`, `evaluate_promotion_gate:2711-2970`, `aggregate_contract_effects:2020-2229`; `known_source_ids:1553-1583` and `known_model_ids:1586-1622` structurally alike | Declarative field/registry specs plus isolated evaluation stages; one strict schema/error-path owner, no optional legacy acceptance | **120–200; medium** (contract identity) | `tools/tests/campaign/test_experiment_contract.py`, `test_experiment_contract_va12_resource_acceptance.py`, `test_experiment_contract_cli.py` |
| 3 | **Tuning measurement ingest** `inventory.py` **2,169**; `load_measurements:879-1821` **943** lines, `build_database:536-724`; mixed JSONL, canonical signature, provenance, SQL | Typed pipeline `parse → attest → resolve → persist` with one transaction and explicit trust state; factor repeated row inserts, do **not** duplicate runtime canonicalization in Python | **120–200; medium–high** (identity/SQLite) | `tools/tests/integration/test_re09_schema_v4.py`, `tools/tests/tuning/test_dispatch_db_capabilities.py`, `test_hip_producer_capabilities.py` |
| 4 | **Patch evidence + rebase** `evidence.py` **1,788**, `rebase.py` **1,736**; qualification clones `evidence:1622-1766`, digest shape `:260-407`, selector paths `rebase:237-408` | Common record traversal and reason accumulation; **status-specific strict predicates**; one selector serializer/digest, not merged acceptance rules | **100–180; medium** (fail-closed status) | `test_patch_evidence_va08.py`, `test_patch_evidence_va18.py`, `test_patch_rebase.py`, `test_evidence_withdrawals.py` |
| 5 | **HIP autotune** `hip-autotune-tuner.cu` **4,095**, `hip-autotune-dispatch.cu` **2,713**; duplicate launch-context construction `dispatch:1644-1654,1746-1754`; tuner same-file shingle matches `:3113-3120,3433-3440` | One typed launch-context assembler and constexpr/template strategy per kernel family; keep native/fused eligibility different | **100–180; high; GPU revalidate** | `tools/tests/tuning/test_dispatch_abi_agreement.py`, `test_dispatch_safety.py`, `test_candidate_coverage.py`; compile plus hardware parity/perf |
| 6 | **Remote MoE launch wiring** `moe_remote_gpu.h` **1,848**; 37 `hipLaunchKernelGGL` call sites, 34 same-file eight-line matches around `:789-840`; `moe_remote_epoch_test.cpp` **1,747** | Typed `LaunchArgs` and templated variant dispatch; factor common gu/dn row arguments, preserve specialization/shape guards; parameterize identical epoch test fixtures separately | **90–150; high; GPU revalidate** | `moe_remote_epoch_test.cpp`, `combine_epoch_test.cpp`, `moe_backend_test.cpp`, `tp_exactly_once_test.cpp` |
| 7 | **CLI assembly** `cli/main.py` **2,375**; `build_parser:100-2098` owns 1,999 lines, though `cli/<domain>.py` handlers already exist | **Per-subcommand parser registration**, lazy-load implementation at invocation, keep one top-level help/parser contract. Moving 1,500+ lines alone is **not** a reduction | **70–130; low–medium** (CLI/help parity) | `tools/tests/core/test_cli_layering.py`, `test_cli_tooling_surface.py`, `tools/tests/cli/test_cli_tuning.py`, `test_cli_source_pull.py` |
| 8 | **Patch campaign producer** `campaign/producer.py` **2,530**; `_run_validation_producer:2295-2465`, `execute_validation_producer:1156-1304`, `CampaignProducerRuntime.build_pair:176-309` | Plan stages with typed request/result and shared artifact/evidence persistence; keep producer plugin boundaries already documented in TOOLING.md | **60–110; medium** (receipt semantics) | `test_validation_producer_execution.py`, `test_producer_session_aggregation.py`, `test_patch_validation_campaign_paired_bench.py` |
| 9 | **Pin bump orchestration** `release/pin_bump.py` **2,480**; `_run_phases:1970-2351`, `run_phase_compile:1189-1274` | Declarative phase registry with one checkpoint/receipt driver; extract only structurally identical phase transitions; preserve resume and rollback semantics exactly | **60–110; medium–high** (releases) | `tools/tests/release/test_pin_bump.py`, `test_pin_bump_resume_inputs.py`, `test_bump_gates.py` |
| 10 | **Tuning catalogue** `tuning/catalog.py` **1,878**; 43 functions, `build_manifest:932-1050`, `render_mmvq_instances:1526-1607`, `enumerate_mmq:429-494` | Table-driven **validated family descriptors** generating enumerator/renderer inputs; one eligibility definition, not independent dispatch logic | **50–100; medium** (candidate semantics) | `tools/tests/tuning/test_hi09_catalog_completeness.py`, `test_catalog_resource_blacklist.py`, `test_candidate_coverage.py` |

**Deferred, deliberately not counted:** `engines/radiance/components/serving/contract/registry.cpp` (1,928), `serving_test.cpp` (1,723), `serving.cpp` (1,295), `manifest_writer.cpp`, and anything in `serving/generated/` until **RSC11 S13** is merged. Then consider table-driven v2 contract parsing and parameterized fixtures *without* duplicating the generator. `moe_remote_epoch_test.cpp` is a secondary test-only candidate, not an extra 1,747 lines to remove. `1361_prefill_stagger_pair/patch.py` is a high-value patch **payload** and has 39 helper-constructed edits; relocating its 1,314 known upstream text lines does not by itself reduce them.

## Reusable pattern catalogue (apply with strict boundaries)

1. **Typed field registry:** *before* `parse_contract()` repeatedly calls `_table(doc, key)` / `_required_string(table,key)` (953-1465); *after* a single list of required field descriptors is consumed by a shared strict reader, with existing per-field validators and identical dotted error paths. Apply to genuinely uniform tables, **not** different promotion policy.
2. **Pure phase pipeline:** *before* `load_measurements()` parses JSONL, attests signatures, creates build/candidate rows and writes winners in one 943-line function; *after* `ParsedRun → AttestedRun → ResolvedRows → transactional_insert`, with a single fail-closed commit boundary. Use where input/output identities can be explicit; never permit partially ingested runs.
3. **Shared traversal + explicit strategy:** *before* `verify_ported_benched_patch` (1697-1732) and `verify_deferred_hardware_patch` (1735-1766) repeat record loops; *after* `qualify_records(records, required_policy)` where policy is a **required** enum and each rule still has its own status-specific predicate. Do not weaken a benchmark qualification into a deferred-hardware rule.
4. **Parser family registration:** *before* `build_parser()` registers `pull`, `audit`, `check`, etc. inline (100-2098); *after* `cli/source.py:register_parsers(sub)`, `cli/patch.py:register_parsers(sub)` returning their parsers, with lazy handler imports only at execution. Apply to one isolated command family at a time; verify help, argv and exit codes.
5. **Explicit phase state machine:** *before* `_run_phases()` manages many resumable bump decisions (1970-2351); *after* a fixed tuple of `Phase(name, run, acceptance, resume_key)` evaluated by one transition driver. Only combine phases with the same failure/receipt guarantees.
6. **One descriptor, several outputs:** *before* `enumerate_mmq()` and `render_mmvq_instances()` use separate procedural candidate/format rules; *after* typed family descriptors produce an enumeration and renderer input through one validation path. Apply only if the descriptor expresses *all* family-specific eligibility without new magic numbers.
7. **Typed GPU variant dispatch:** *before* `moe_remote_gpu.h:789-840` passes the same sorted/offset/shape arguments through multiple gu variants; *after* `launch_rows<Variant>(args)` with compile-time variants and distinct guards. Likewise assemble `ggml_hip_launch_context` once (dispatch.cu:1644-1654,1746-1754). Use only after source-level proof of identical argument evaluation order; require real-device parity and performance gates.
8. **Safe anchored patch builder:** *before* each `Edit(id, anchor, mode, text, guard, expect_matches=1, ...)`; *after* a canonical `unique_guarded_replace(target, anchor, replacement, guard, reason)` producing the **same Edit contract** and computing a span limit from the anchor. Use for one-match/static edits; keep complex regex/rewrite edits explicit and all idempotence checks mandatory.
9. **Owned hook + parameterized tests:** *before* substantial upstream code is embedded in `rad0007_hc_pf2_fp8_down/patch.py` and similar edits; *after* BigCherry-owned `bc_*.h/.cpp/.hip` blocks, with upstream one-line include/call hooks (existing model: `rad0002` and `rad0009`). Replace identical `moe_remote_epoch_test.cpp` fixtures with case tables only when failure diagnostics remain per-case. Apply **after** the active engine patch lanes settle; externalization primarily reduces upstream drift, not net LOC.

## Patch-extension accounting and hook recommendation

| Static measurement | llama.cpp | Radiance |
| --- | ---: | ---: |
| Directories with tracked `patch.py` + `patch.toml` | **151** | **13** |
| Total `patch.py` physical lines | **38,570** | **3,254** |
| `FilePatch` constructors / `create=True` | 343 / 22 | 34 / 11 |
| Direct `Edit` calls | 974 | 80 |
| Additional calls through local `_edit` helper | 39 (patch 1361) | 0 |
| Explicit `anchor=` keywords (including indirect calls) | **1,013** | **80** |
| Explicit `guard=` / `expect_matches=` keywords | 1,012 / 550 | 80 / 80 |
| Static resolved `Edit.text` payload in upstream `FilePatch` (lines) | 17,748 | **770** |
| Static resolved `Edit.text` payload in created files (lines) | 4,289 | **1,118** |

The payload counts measure **lines of supplied replacement/insert/create text**, **not net added code**: `replace` may repeat original upstream lines; unresolved dynamic expressions (19 llama, 2 Radiance edits) are excluded. Radiance uses **23 upstream `FilePatch` declarations / 11 created-file declarations**, **80 edits** (79 statically resolvable anchors), and 1,888 resolved lines of replacement/created-file text; of its **770 upstream payload lines**, **467** occur in `rad0007` alone. The biggest Radiance anchor concentrations are `rad0002` **33**, `rad0010` **11**, `rad0007` **7**, `rad0009` **6**, and `rad0011` **6**. For llama.cpp: `1263` **47**, `1361` **39**, `1337` **36**, `1202` **35**, `1281` **34**. These counts describe *authoring sites*, not unique upstream hunks after composition.

Metadata dependency declarations are already centralized in `patch.toml`: **75/151 llama.cpp** patches declare **108 `requires` edges** (11 declare conflicts); **2/13 Radiance** patches declare **2 `requires` edges**. These are not duplicated Python constants and should not move into `patch.py`.

**Do not build a second patch system.** `tools/bigcherry/patch/{apply,patchset,registry,rebase}.py` already owns loading, anchored apply, idempotence and source-bound reports; `bigcherry.patcher` is an existing compatibility facade (TOOLING.md). A tiny authoring-only builder *inside the canonical patch package* could collapse repeated `Edit` keyword boilerplate and standardize one-match, span and guard requirements. Keep `patch.toml` as the dependency/metadata authority, `EnvDoc` as the existing environment contract, and the actual patch IDs/digests stable until migrated and revalidated. Do not add a parallel declarative loader, defaults that accept old syntax, or hand-maintained generated registries. 150 llama patch files currently import the old facade; migrate **all** consumers over bounded slices, explicitly track remaining imports and remove the facade only after the existing retirement checks.

**Radiance recommendation:** prefer owned hook headers/code for *material new functionality* where the upstream call-site ABI is narrow; this strategy is already proven structurally by `rad0002` (652 static created-file payload lines vs 45 upstream edit payload lines) and `rad0009` (324 vs 16). For `rad0007`, consider moving its large inline FP8 kernel implementation into a BigCherry-owned include/translation unit with thin hook-in calls, **only** after confirming include visibility, symbol linkage, licensing and kernel registration. Retain tiny 1–2 line upstream guards where moving them would require exposing private types or multiply hook count. `rad0013` has 125 lines of upstream replacement payload behind one anchor and merits a separate boundary review. A hook-based split is not a performance optimization on its own.

**Anchor-drift dry-run (future patch slice; not executed in this read-only audit):**

1. Record pinned Radiance/llama upstream refs, BigCherry commit, patch/overlay digests, per-patch anchor counts and before/after exact file hashes. Capture the current source-bound `bigcherry patch-rebase-check` report in an **isolated detached source worktree**, never the production vendor tree.
2. For one candidate patch at a time, introduce an owned block and minimal upstream include/call, then run `PYTHONPATH=tools python -m bigcherry patch-rebase-check --help` to select the pinned source and produce a new revision-bound report; require all formerly-clean edits to remain `CLEAN` or explicit `CLEAN_NOOP`, not silently `NOT_APPLICABLE`, `UPSTREAM_ABSORBED` or `FAILED`.
3. Apply old and proposed patch stacks to two *isolated* clones of the **same** pinned source, compare modified upstream file list, line diff, created-file hashes, exported signatures and idempotence on a second application. Probe the next upstream pin with the same report machinery. Fail on any altered anchor-match count, unexpected guard match, extra edit, or changed dependency closure.
4. Run existing Python patch tests and CPU-only Radiance `bc_serving_test` / `moe_remote_epoch_test` where applicable, then native compile/link for all affected card architectures. **Hardware re-validation owned by the lead** is mandatory for new HIP/FP8/GEMM/dispatcher paths: gfx1100/gfx1201/gfx1030 applicable cases, numerical parity/KL, real throughput/latency (including concurrency), fallback paths and no regressions with opt-in features off.

## Sequenced reversible slices

Each PR changes **at most 400 total lines** (added + deleted, including tests); split any slice that exceeds the cap. Sequence the card-free Python work first; avoid touching `serving.cpp`, `manifest_writer.cpp`, `serving/generated/`, or v2 contract generator while RSC11 S9/S10/S13 work is live. Do not move a source merely to shrink its apparent line count.

| Order | Slice, max changed lines | Acceptance and conflict/lease constraint |
| ---: | --- | --- |
| 1 | **CLI source/pull parser extraction**, ≤300 | Exact `--help` and parser/exit parity against frozen fixtures; one family only, lazy dispatch; card-free. |
| 2 | **Evidence qualification common traversal**, ≤300 | VA08/VA18, withdrawal and stale-record tests unchanged; no policy downgrade; card-free. |
| 3 | **Experiment known-ID resolver**, ≤220 | Source/model duplicate registry parsing removed; contract digest/errors unchanged; card-free. |
| 4 | **Tuning ingest header/row typed stage**, ≤350 | Schema-v4 and capability/attestation tests unchanged; next stage separately; card-free. |
| 5 | **Campaign producer receipt stage**, ≤300 | Exact artifact digests, cancellation and failure receipts; no new producer framework; card-free. |
| 6 | **Pin-bump one resumable phase**, ≤300 | Resume/retry, pin consistency and failure injection parity; card-free. |
| 7 | **Catalogue one MMQ family descriptor**, ≤280 | Candidate list, manifest digest and eligibility byte-for-byte equal; card-free. |
| 8 | **Patch-authoring canonical helper + a few llama patches**, ≤350 | Same `Edit` values, anchor match counts, repeat-apply idempotence, metadata and dependency closure; coordinate with patch owners. Repeat per stable patch cohort. |
| 9 | **HIP launch-context builder**, ≤280 | CPU static/ABI tests and native compile before merge; **separate hardware performance/quality acceptance**; no GPU in code slice. |
| 10 | **Remote-MoE gu-rows argument factoring**, ≤350 | Compile/component tests plus real gfx1100/gfx1201/gfx1030 relevant-layout parity, epoch and concurrency checks; schedule GPU with lead. |
| 11 | **Radiance hook pilot (rad0007 or rad0013)**, ≤400 each | Only after active Radiance patch owners are clear; anchored dry-run against both pinned/next refs, compile and GPU gates; require fewer fragile upstream edits, not just relocation. |
| 12 | **Serving v2 fixture parametrization / registry review**, ≤350 | **After S13 lands**; v2 golden manifest/digest and native `bc_serving_test` exact, no generated edits; only then evaluate whether a new reduction is justified. |

No slice introduces compatibility shims, silently optional parameters, hard-picked profile constants or an independent runtime/patch registry. Production consumers and tests migrate together; generated output is regenerated solely from its canonical input. The RSC11 integration lead owns conflict-sensitive files and the hardware acceptance queue.

## Plan items to create with ag-planning

The planning tools were unavailable in this task; **none of these items was created or marked complete**. Proposed groups are discipline-prefixed and can reuse an existing matching group if the lead prefers. Sizes are per small PR (S/M), not total programme size.

| Proposed title | Group | Size | Completion/acceptance criterion |
| --- | --- | --- | --- |
| Extract CLI parser family while preserving help | `run-cli-architecture` | S | One family migrated, help/argv/exit snapshots identical, no eager handler imports |
| Unify patch evidence record traversal | `patching-evidence-maintenance` | S | VA08/VA18/status/withdrawal strictness unchanged, all tests pass |
| Deduplicate experiment registry-ID validators | `run-experiment-contracts` | S | Source/model identity and digest/error fixtures unchanged |
| Factor measurement ingest into typed phases | `tuning-measurement-ingest` | M | Schema-v4, provenance and strengthened-attestation transaction parity |
| Extract validation-producer evidence stage | `patching-validation-producer` | S | Receipt/failure/cancellation identity parity |
| Replace duplicated pin-bump phase wiring | `build-pin-bump` | S | Resume/retry and release gate fixtures identical |
| Make tuning family selection descriptor-driven | `tuning-kernel-catalogue` | S | Candidate enumeration, eligibility and manifest digest parity |
| Consolidate guarded patch authoring on canonical API | `patching-authoring` | M | Each small cohort passes anchor/rebase/idempotence checks, all consumers migrated before retiring facade |
| Factor HIP autotune launch-context duplication | `tuning-hip-autotune` | S | Host ABI/static tests, native builds, separate lead-run numerical/throughput AB |
| Factor remote MoE GPU launch variants | `run-radiance-moe-remote` | M | Epoch/TP/component gates, lead-run per-card parity and concurrency |
| Trial owned-hook externalization for bulky Radiance edits | `patching-radiance-hooks` | M | Two-source anchor drift, link/parity tests, lead-run FP8 correctness/perf before activation |
| Review serving v2 validator/test duplication after S13 | `run-serving-configuration` | S | S13 merged, no generator hand edits, v2 golden digest/component parity |

**Bottom line:** the near-term low-conflict work is Python validation/CLI/ingest consolidation; ~**1.4–2.4k net lines** are plausibly removable from the ranked ten over small reviewed PRs. A wider source-file split or conversion of large patch payloads into owned code is valuable primarily for **maintainability and upstream-bump resilience**, not counted as LOC savings.
