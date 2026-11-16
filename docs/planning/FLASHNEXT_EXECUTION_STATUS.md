# Flash-Next four-GPU plan reconciliation — 2026-11-10

Host: two gfx1100 RX 7900 XTX head helpers, gfx1201 R9700 engine and gfx1030 RX 6900 XT tail. This is a docs-only investigation: no GPU, no lease, no new benchmark. Scope evidence: plan item Validation/Acceptance/Notes, merged git history, layouts and calibration records, release ledger, and actual results under ${BC_WORK_ROOT}/runs. Use the linked plan items as the action queue, not the obsolete standing to-do list.

**State discipline:** State in file is the actual frontmatter. Partly means implementation or measurement exists but not all Acceptance Criteria have been demonstrated. A pending file with partly shipped work is *progress-stale*, not automatically completed. A merged slice alone cannot justify a terminal state.

## Top eight genuinely open performance actions

| Priority | Item(s) | Specific action | GPU lease? |
| --- | --- | --- | --- |
| 1 | RHP18, RHP19, RHP20 | Explicit on/off/not-applicable/degraded placement state and bounded calibration; safety across helper roles and all supported architectures | No for fake/card-free tests, yes for hardware acceptance |
| 2 | RHP22 | Disposition forced-churn p99 KLD breach on same stable reference before default-on | Yes |
| 3 | ENG01 | Obtain policy-on engine per-op profile (cold/document prefill, decode) without profile_ops startup OOM | Yes after card-free harness repair |
| 4 | ENG03 | Per-stage 1/2/4/6/8-stream decode scaling after ENG01 | Yes |
| 5 | ENG02, FMTP07 | One current-build 1/6-stream MTP depth and acceptance sweep | Yes |
| 6 | ENG04, RHP12 | KV format, context capacity, retained-session baseline and quality at 262K concurrent / 1M single | Yes |
| 7 | RHP21, RHP04, RHP09 | First-cold-after-doc adaptation A/B, then cost-aware solver only if measured path improves | Replay no, A/B yes |
| 8 | RHP01, RHP02, RHP08 | Final histogram coverage and tail route parity; retire duplicated diagnostic only after proof | Card-free no, coverage A/B yes |

**Not queued:** RHP10 thrash root-cause and 1M tail matrix; RHP11 live-migration build; TK01/TK02/TK04 bit-identical improvements; TK03 rejected experiment; HL04 low-benefit CPU combine push.

## Tail sizes and combine: actual evidence and safe decisions

The layout table in engines/radiance/layouts.toml records XTX budgets of 22528 MiB each. Budget values below are helper residency, not whole-card VRAM. RAD23 is the old baseline and RAD46 reports later build-sensitive quality changes. No universal tail optimum was demonstrated.

| Layout/arm | Tail MiB; combine | KLD evidence | Speed evidence | Verdict |
| --- | --- | --- | --- | --- |
| flashnext-2xtx | 0; off | RAD23 two-XTX mean 0.05518, p99.9 1.642 | Natural 25/14/72K prefill 3297/3053/2998 tok/s, old same-build sweep | Old short-prompt baseline |
| flashnext-2xtx-combine | 0; CPU combine | RAD18/RAD23 combine gate; not new current pin | RAD23 combine +7.6/+6.3/+1.5/-0.5% at increasing natural lengths | Earlier fastest short-prompt configuration, not universal |
| flashnext-2xtx-rdna2-tail | 2048; off | RAD23 2G mean 0.05555, p99.9 1.752: clean pass | Natural 25/14/72K 3094/2819/2941 tok/s; ~1.7 GiB engine pool freed; tail-sweep run | Conservative named 2G tail; capacity trade |
| experimental 4G | 4096; off | RAD46 v1.17.0 p99.9 3.478 vs 2.067: **FAIL +1.41**, two repeats identical (tail4g-kld-117) | RAD23 context sweep at 34/147/332/663K: prefill 2951/2870/1950/1259, decode 127/95/97/70 | Measured speed but NOT qualified on current-build gate; not optimal |
| experimental 6G | 6144; off | RAD23 p99.9 2.414 vs 1.642: **FAIL +0.77** | Old natural 25/14/72K 2084/2845/2804; RHP10 1M experiment at 362K/710K 2572/2169 | Not qualified for default |
| experimental 8G | 8192; off | Tail-sweep-kld ran, but no documented accepted current gate | tail-sweep engine short decode 109.36 tok/s vs 2G 113.08 and 4G 121.24 (two reps, acceptance confounding) | Measured, no demonstrated winner |
| flashnext-2xtx-rdna2-tail-combine | 2048; CPU combine | **2G off KLD is NOT evidence of 2G+combine parity** | RAD23 4G+combine early comparison showed no win over no-tail CPU combine; rad23-combine-layout exists | No isolated current-build 2G combine quality+speed qualification established |
| flashnext-4gpu | 14336; CPU combine | RAD23 old pass +0.43 p99.9; RAD46 v1.16.0 pass -0.13; **v1.17.0 FAIL +0.94** (3.003 vs 2.067). Position 117 is an engine/helper rounding-sensitive token, not proved tail-only (RAD49) | Old 8-session ~1.04M fill 2369 vs 1353 tok/s no tail (+75%). RHP10 710K prefill 2307 vs 1244 and host-to-GPU 613 vs 1392 GiB, decode broadly KV-bound | Best tested very-long-prefill/capacity result in RHP10, but not unconditional quality-qualified |

**Measured versus missing:** RHP10 already tested 0/3/6/14 GiB at actual 361,979 and 709,649-token positions at the 1M setting, one run per arm. Do not rerun that matrix. A *matched current-build factorial* of 2G/4G/14G each both combine on/off, with live-policy stable quality and speed, was **not established**; neither was a 2G+combine gate that can be substituted from the 2G off run. The 4G gate **was** measured (failed), so it must not be called never tested. Tail latency and fitted/static heat differences affect comparisons. Treat RAD23 old pass and RAD46 newer failure as distinct build-reference results, not a contradiction to erase.

Sources: docs/planning/active/patching-radiance-rdna3/RAD23.md and RAD46.md; docs/planning/completed/patching-radiance-heat-placement/RHP10.md; engines/radiance/layouts.toml; ${BC_WORK_ROOT}/runs/tail-sweep, tail-sweep-kld, tail4g-kld-117, tail14g-kld-116, tail14g-kld-117, rhp10-tails3, rad23-combine-layout, and HL04 lane-trace results.

## Per-item status and exact next steps


### Heat placement (RHP)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [RHP01](active/patching-radiance-heat-placement/RHP01.md) | Prove complete asynchronous Radiance routing evidence under long prefill | pending | partly; progress-stale | RHP01: 3696/3696 prefill credited, zero overwrites at chunk 2048; rank 0 only | Test 4096/8192 chunks, verification batch and host run-ahead; close only if coverage acceptance passes |
| [RHP02](active/patching-radiance-heat-placement/RHP02.md) | Single Radiance heat observation feed to libr3moe; remove duplicate route collection | pending | partly; progress-stale | PRs #500/#502 pinned histogram is interim, not the single shared feed | Build versioned observer/parity gate before deleting duplicate route diagnostics |
| [RHP03](active/patching-radiance-heat-placement/RHP03.md) | Extend Radiance telemetry and persist one placement snapshot, not a heat database | pending | open | RHP03 epoch-journal design; policy logs are not full snapshots | Bounded epoch/rollback/coverage snapshot with restart tests |
| [RHP04](active/patching-radiance-heat-placement/RHP04.md) | Placement policy: stage 1 heat/bottleneck balancing, stage 3 calibrated critical-path solver | in_progress | partly | #597 live target-map policy and policy-live-8 A/B; stage 3 not built | Calibrate critical-path device cost solver after RHP09 and RHP12 |
| [RHP05](completed/patching-radiance-heat-placement/RHP05.md) | Request-count or timed heat reevaluation with restart-safe ownership activation | superseded | superseded | #501 drift watch retained; restart activation discarded | Do not rebuild restart activation; reuse trigger in RHP07 |
| [RHP06](active/patching-radiance-heat-placement/RHP06.md) | Stage 1: live helper expert migration - slot manager, async host->VRAM upload, retirement | pending | partly; progress-stale | #542/#557/#558/#572 live moves and #733 paced 131 swaps with passing KLD | Finish full RHP12 8-session/6-stream/42K latency plus heterogeneous-move gate |
| [RHP07](active/patching-radiance-heat-placement/RHP07.md) | Placement controls: decay, hysteresis and change budgets (stage 1), then adaptive refinement (stage 5) | pending | partly; progress-stale | #597/#693/#708 controls and calibration-backed values; adaptation remains | Evaluate adaptive triggers/hysteresis and rollback under warm/cold shifts |
| [RHP08](active/patching-radiance-heat-placement/RHP08.md) | Runtime tail-heat collector | pending | partly; progress-stale | #807 and HL04 trace show tail routes; no shared feed parity proof | Compare tail routes to coherent histogram on cold/docs/long context; close or document gap |
| [RHP09](active/patching-radiance-heat-placement/RHP09.md) | Stage 2-3: topology discovery, default-on calibration and the critical-path cost model | pending | partly; progress-stale | RHP16 shipped derived policy settings; topology-cache and cost model unproven | Benchmark topology-keyed costs and cache timings and remove literal resource budgets |
| [RHP10](completed/patching-radiance-heat-placement/RHP10.md) | Long-context thrash experiment | completed | done | #512/#613/#807; RHP10 0/3/6/14 GiB 362K/710K matrix, retained thrash fixed | None; preserve results as RHP12 retained-session cohort |
| [RHP11](active/patching-radiance-heat-placement/RHP11.md) | Stage 1: placement-epoch mailbox protocol v23 (live helper ownership changes) | in_progress | partly | #542/#557/#558/#572 epoch protocol and forced real moves; #733 tests | Cite complete RHP12 timing/quality gate before closure; no new slice G |
| [RHP12](active/patching-radiance-heat-placement/RHP12.md) | Every stage: per-layout baselines and the no-regression gate (return to a known layout without degradation) | pending | partly; progress-stale | RHP12 retained-session failure; rhp16-prod-ab2 has paired variants | Record per-layout baselines including KV retention, fresh/return state and resource gate |
| [RHP13](active/patching-radiance-heat-placement/RHP13.md) | Stage 4: one allocator for engine, host and helper residency (managed radiance core/place patch) | pending | open | RHP10 loan and retained-KV issue; RSC03 budget plan | One allocator balancing engine KV, experts, helper VRAM, host pool |
| [RHP14](active/patching-radiance-heat-placement/RHP14.md) | Parked: overlap helper round trips across MoE layers (cut fixed per-layer latency on short steps) | pending | parked | RHP14 cross-layer overlap proposal, no measurements | Park until ENG01 identifies fixed helper round trip as critical |
| [RHP15](active/patching-radiance-heat-placement/RHP15.md) | Admission covers media encoder passes (encoder-only passes need no remote MoE ticket) | pending | open | RHP15 media encoder admission gap | Card-free media-only ticket tests, then card-backed admission check |
| [RHP16](active/patching-radiance-heat-placement/RHP16.md) | Calibrate live-placement observability and move triggers from measured quantities (no hand-picked policy constants) | in_progress | partly | #620/#621/#666/#693/#708 derived policy; rhp16-prod-ab2 evidence | Finish keyed store/enablement/quality gates and RHP18 explicit state |
| [RHP17](completed/patching-radiance-heat-placement/RHP17.md) | Device-speed and phase-aware expert placement: minimise each layer's slowest-device finish time (offline simulation first) | deprecated | deprecated | #806 closed offline device-speed simulation; successor RHP04/RHP09 | No rerun of rejected simulator |
| [RHP18](active/patching-radiance-heat-placement/RHP18.md) | Live placement observability: the manifest, the launcher and the helper say whether the policy is on, why not, and what a calibration is waiting for | pending | partly; progress-stale | #798/#802/#804 observability and policy-off reasons; #790 invalidated calibration key | Preflight on/off/not_applicable and degraded status on all layouts; requalify matching record |
| [RHP19](active/patching-radiance-heat-placement/RHP19.md) | Calibration that always terminates with a usable record: bounded loop, periodic waiting report, and a deliberate tail-lane move to measure the tail-lane quantities | pending | partly; progress-stale | #801 newline fixed; ~30-minute run had COMPLETE record | Bounded lifecycle, independently measured passive values and partial/failed records |
| [RHP20](active/patching-radiance-heat-placement/RHP20.md) | Live expert swapping and its observability verified on every card type and layout (gfx1201, gfx1100, gfx1030; head, tail, TP and engine-resident placement) | pending | partly; progress-stale | #805 blocks incompatible arrangements; last-helper tail assumption still flawed | True placement roles and supported/unsupported gfx1201/gfx1100/gfx1030 move matrix |
| [RHP21](active/patching-radiance-heat-placement/RHP21.md) | First cold prompt after a documents-only period: shorten the live policy's adaptation lag | pending | open | RHP21; RHP16 policy-on doc-to-cold experiments | Measure first cold prompt adaptation penalty and control migration cost |
| [RHP22](active/patching-radiance-heat-placement/RHP22.md) | Live placement quality disposition: the forced-churn KL p99 breach (+0.234 against +0.2) | pending | open | policy-kld forced-churn p99 increase +0.234 vs gate +0.2 | Same-reference repeat and stable-position quality disposition before default-on |

### Engine profiling and scaling (ENG)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [ENG01](active/tuning-radiance-engine/ENG01.md) | R9700 engine per-op attribution with the live policy on: prefill (document, cold) and decode, with an unprofiled reference | pending | open | engine-profile-policy-on attempt did not deliver per-op attribution | Profile policy-on cold/doc prefill and decode without profile_ops startup OOM; control |
| [ENG02](active/tuning-radiance-engine/ENG02.md) | Current-build MTP depth and acceptance sweep on flashnext-4gpu | pending | open | ENG02 prior MTP evidence is not a current-build depth sweep | Matched 1/6-stream MTP depth/acceptance A/B; share FMTP07 evidence |
| [ENG03](active/tuning-radiance-engine/ENG03.md) | Multi-stream decode scaling decomposition: matched 1/2/4/6/8-stream runs with work per step | pending | open | Prior 1/6-stream rates have no stage scaling decomposition | Matched 1/2/4/6/8 stage-time scaling after ENG01 |
| [ENG04](active/tuning-radiance-engine/ENG04.md) | KV format and context-capacity matrix on the current build: fp8 / fp8_v4 / k6_v4, 262K concurrency, single-session 1M | pending | open | ENG04; QFN old llama.cpp memory ceilings not Radiance | Current-build fp8/fp8_v4/k6_v4 KV quality/capacity/speed at 262K/1M |

### Tail kernels (TK)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [TK01](completed/tuning-radiance-tail-kernel/TK01.md) | RX 6900 XT tail quad kernel: register-permute W4 decoder instead of the LDS table (bit-identical) | completed | done | #794; tk01-e2e-ab and tk01-replay-hash: exact and combined GEMM -9% | None; merged W4 register decoder |
| [TK02](completed/tuning-radiance-tail-kernel/TK02.md) | RX 6900 XT tail quad kernel: fold the quad DPP broadcasts into v_add_f32_dpp (bit-identical) | completed | done | #794; rad49-tail-quadopt; exact DPP-add reduction | None; merged together with TK01 |
| [TK03](completed/tuning-radiance-tail-kernel/TK03.md) | RX 6900 XT tail quad kernel: occupancy-specialised row batches and no padded-row work | deprecated | done no-go | tk03-split: 40 captures match; 1.7% kernel improvement below 5% bar | Do not integrate marginal split experiment |
| [TK04](completed/tuning-radiance-tail-kernel/TK04.md) | RX 6900 XT tail quad kernel: stage each activation row once per expert, not once per N-tile CTA | completed | done | #797; tk04-bypass and 40 captures; packed E4M3 widening -5.7% GEMM | None; N64 branch rejected by 0.8 ms go gate |
| [TK05](active/tuning-radiance-tail-kernel/TK05.md) | Experimental branch: INT8 dot4 tail kernel (W4NL codebook as int8, activations re-quantised) under its own quality gate | pending | open low-priority | TK05 exact INT8 two-pass cost no gain, lossy gate headroom absent | Offline representation error study only; no default production kernel |

### Host links (HL)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [HL01](completed/run-radiance-host-link/HL01.md) | Diagnose primary PEG-half GPU host-transfer contention | completed | verified slot-swap 2026-11-13 | Matching PCIe/IOMMU; before/after XTX card swap and partial-population matrix; ~400 us TP3 reduce skew | Diagnosis closed; HL02 owns transfer-method mitigation |
| [HL02](active/run-radiance-host-link/HL02.md) | Shared contended host-link matrix: non-coherent mailbox, true CPU push to one/all, R9700 qualification | pending | new gates untested | HL01 swap diagnosis, RAD40 physics, RAD50 DMA-BUF serial/live receipts | Gate A non-coherent; Gate B simultaneous 320 KiB–10 MiB CPU push; Gate C GPU-live safety, then VTP38 handoff |
| [HL03](active/run-radiance-host-link/HL03.md) | CPU-push the helpers' activation copies instead of each helper pulling them (prefill activation round) | pending | open | HL03 design, depends on HL02 | Bench activation plane CPU push, only if link cost is critical |
| [HL04](completed/run-radiance-host-link/HL04.md) | Prefill-only CPU-combine push into R9700 VRAM: is the combine on the flashnext-4gpu critical path? | completed | done no-go | hl04-combine-trace: combine ~0.1-0.2 ms exposed, doc steps engine-bound | No R9700 BAR push integration until >0.3 ms exposed |
| [HL05](active/run-radiance-host-link/HL05.md) | Host-memory all-reduce via CPU relay push for tensor-parallel layouts (27B on two XTX) | pending | open low-priority | HL05 TP all-reduce relay design; TP performance open | Lab two-XTX relay with end-to-end benefit before implementation |

### Workspace lifecycle (WSL)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [WSL01](active/run-workspace-lifecycle/WSL01.md) | Standard work-root layout and one run-directory template | pending | partly; progress-stale | #311 workspace.py/run_manifest.py foundation; producers not all migrated | Migrate producer run manifests incrementally |
| [WSL02](active/run-workspace-lifecycle/WSL02.md) | Harvest: extract what we keep, then mark the raw output expendable | pending | open | WSL02 harvest spec and evidence standards | Implement validated harvest of stream, acceptance and profiling data |
| [WSL03](active/run-workspace-lifecycle/WSL03.md) | Retention and managed cleanup (`bigcherry clean`, dry run first) | pending | partly; progress-stale | #311 clean dry-run skeleton; quarantine/apply not accepted | Lease-safe quarantine after WSL02 |
| [WSL04](active/run-workspace-lifecycle/WSL04.md) | Build and source cache GC keyed by plan id | pending | open | WSL04 build/source cache GC plan | Implement keyed and lease-safe cache retention |
| [WSL05](active/run-workspace-lifecycle/WSL05.md) | Experiment templates: config, launcher, analysis and evidence in one pattern | pending | open | WSL05 experiment template plan | Standardize configuration, launcher, summary, evidence |
| [WSL06](completed/run-workspace-lifecycle/WSL06.md) | Adopt and clear the legacy work roots on dev | deprecated | deprecated | WSL06 legacy-root plan in terminal completed tree | No deletion or re-run |
| [WSL07](active/run-workspace-lifecycle/WSL07.md) | Lab worktree hygiene: lint ad-hoc worktrees in lab scripts; slice status/prune see lab-host worktrees | pending | partly; progress-stale | Slice status/prune present; lab lint not independently accepted | Confirm lab-worktree lint and remote-host inventory |
| [WSL08](active/run-workspace-lifecycle/WSL08.md) | The gpu lease covers the lifetime of the serving processes a launcher detaches | pending | open | WSL08; RSC11 serving supervisor skeleton being built | Card-free lease-held-through-process-exit tests then production acceptance |

### Related Radiance items (RAD)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [RAD23](active/patching-radiance-rdna3/RAD23.md) | moe-remote: helper count and topology as runtime values (no fixed 2), selectable combine methods | pending | partly; progress-stale | RAD23 tail topology/KLD/bench; layouts.toml; no auto/GPU-combine proof | Split delivered N-helper/CPU combine from still-open selectable auto/GPU modes |
| [RAD39](active/patching-radiance-rdna3/RAD39.md) | Helper-layout KLD gate judges against the measured noise band, not one control arm | pending | partly; progress-stale | RAD39 reference noise band; rad49-sens-kl instability | Resolve stable-reference numerics with RHP22 gate |
| [RAD46](active/patching-radiance-rdna3/RAD46.md) | RDNA2 tail helper degrades Flash-Next tail quality: KLD p99.9 +1.17 over the two-XTX helpers on 2.3% of routes | pending | partly; progress-stale | RAD46 tail p99.9 v1.16 pass, v1.17 4G/14G fail | Investigate numerical outliers and disposition current-build tail gate |
| [RAD49](active/patching-radiance-rdna3/RAD49.md) | flashnext-tp2xtx: dense/attention TP2 on the XTX pair, R9700 + RX 6900 XT hold the experts (Strata-informed) | pending | partly; progress-stale | #770/#771/#777/#794/#797 TP and fused helper slices | Profile full-model TP critical path and decide performance winner |

### Serving budget (RSC)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [RSC03](active/run-serving-configuration/RSC03.md) | One derived start-up resource plan: host RAM, pinned memory and VRAM budgeted together | in_progress | partly | RSC03 derived host memory budget PR #821 pending CI fixes | Finish no-literal budget derivation with v1 compatibility |

### Flash-Next MTP (FMTP)

| ID | Title | File state | Evidence verdict | Evidence path and finding | Exact next action |
| --- | --- | --- | --- | --- | --- |
| [FMTP07](active/patching-flash-next-mtp-pipeline/FMTP07.md) | Qualify Flash-Next MTP verification overlap on the AMD deployment topology | pending | open | FMTP07 MTP pipeline plan | Run one current-build MTP acceptance sweep shared with ENG02 |

## Adjacent Flash-Next plans in other run groups

These are **llama.cpp/GGUF experiments or release workflow items**, not evidence of performance on Radiance's gfx1201-engine/remote-MoE layout. File states are unchanged; several items have measurements but lack complete acceptance.

| ID | Title | State in file | Evidence verdict / source | Exact next action |
| --- | --- | --- | --- | --- |
| [QFN01](active/run-qwen-flash-next/QFN01.md) | GGUF Flash-Next layout, MTP and ngram speculation | pending | **Partly, progress-stale**: item records 192K + MTP and short-context 48-50 tok/s decode on llama.cpp, not Radiance | Separate GGUF results from Radiance layout decision; validate only outstanding GGUF criteria if backend remains supported |
| [QFN02](active/run-qwen-flash-next/QFN02.md) | Frequency-aware expert split in llama.cpp | pending | **Open, not equivalent to RHP11**: no llama.cpp expert-split runtime acceptance | Decide if backend-specific expert-split work remains in scope; if yes, require oracle, speed and VRAM gate |
| [QFN03](active/run-qwen-flash-next/QFN03.md) | GGUF compute-buffer/KV fit | pending | **Partly**: 192K HIP graph-instance OOM addressed by 1302 eviction, not proof of all proposed savings | Prove remaining GGUF capacity/headroom only on its backend; do not use as Radiance KV ceiling |
| [QFN04](active/run-qwen-flash-next/QFN04.md) | GGUF 248K prefill collapse | pending | **Root cause diagnosed, closure metadata stale**: R9700 within ~150 MiB VRAM caused paging; v2 240K ceiling (QFN04 Notes). Acceptance Criteria empty | Write explicit acceptance and complete ag-planning disposition, not a redundant collapse rerun |
| [MEN26](active/run-multi-engine/MEN26.md) | Radiance production release matches Flash-Next serving | pending | **Partly**: current-pin two-XTX KLD done in ${BC_WORK_ROOT}/runs/men26-kld; tail assessed separately by RAD46 | Audit promotion, patch verification and actual release build serving; close only on full acceptance |

Other tuning-hip-autotune (e.g. THA32) and run-xtx27-prefill items target a different llama.cpp kernel or dense 27B model, not the Radiance four-GPU Flash-Next benchmark queue.

## Ledger and source boundaries

- Git history shows #542/#557/#558/#572 live move implementation, #597 policy, #620/#621/#666/#693/#708 calibrated derivations, #790 key invalidation, #798/#801/#802/#804/#805 observability gates, #512/#613/#807 context closure, #794/#797 kernel wins, #795 link tool and #806 dispositions. See docs/releases/CURRENT_RELEASE_LEDGER.ndjson and the plan items for acceptance, not just commit titles.
- docs/reference/FINDINGS.md holds general guidance, but detailed tail and live placement outcomes live in RAD23, RAD46, RHP10, RHP16, TK01-05 and HL01-04 plus retained run result files, which this audit checked.
- QFP/QFN llama.cpp bring-up and XDP dense 27B are separate workloads; do not treat their KV ceilings or throughput as Flash-Next Radiance measurements. RSC03 and FMTP07 are included as direct dependencies.
- Ag-planning state transitions and review closures require the planning MCP tool and proof of each Validation and Acceptance Criteria; the manual audit deliberately **does not fabricate state changes**. An ag-ledger change event must include plan-item-ids.
