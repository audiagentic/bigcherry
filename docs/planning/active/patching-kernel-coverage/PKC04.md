---
id: PKC04
order: 0
plan: patching-kernel-coverage
state: done
created-at: '2026-09-09T10:51:56.429711+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# Vulkan attention/transport lifecycle: no separate control plane

## Authoritative scope disposition — 2026-10-13 (b11552)

**Decision-only item DONE: no new Vulkan lifecycle, candidate registry, scheduler, cache, allocator, provider SPI, or experiment authority.** This closes PKC04's *ownership question*, not the still-pending Vulkan implementation, driver qualification, provider correctness, replay or hardware work. The 2026-09-10 user-directed Vulkan implementation pause in RRVP02 remains authoritative. This audit changes documentation only and does not queue GPU work.

**Branch/source evidence:** BigCherry `feat/rad03-gpu-lanes` at independent HEAD `8681a4879f93d1781842c169e6ffd2d98286ff53`; pinned llama.cpp b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`). PKC04 last substantive independent update was 2026-09-10; 2026-10-08 changed line endings only. No independent PKC04/Vulkan-lifecycle work or queued qualification was found in the preceding 12 hours. Active Radiance GPU-mailbox/R9700 helper, patch-prefix/release and hardware work is excluded and untouched. Recent BCOP189/188/186 and other scheduled slices do not own this decision.

## Actual implementation boundaries and duplication decision

| Semantic boundary | Existing owner / source | What exists and what remains |
| --- | --- | --- |
| Vulkan `MUL_MAT` / `MUL_MAT_ID` pipeline and preparation | TRVP12, TRVP14/15; `ggml-vulkan.cpp::ggml_vk_get_mul_mat_mat_pipeline_map`, `ggml_vk_matmul_id`, `ggml_vk_mul_mat` | Native pipeline selection, Q8_1 conversion and split-K/reduction exist. Strict BigCherry force/actual-bound-route receipt and replay qualification are **not** implemented; no 1247 package. No second selector. |
| Vulkan dense/sparse attention | Native `ggml_vk_flash_attn`; TRVP12 for provider capability; PRVP02 for separate CM1 safety | Pinned b11552 already gates sparse FA by `n_kv_max`, K/V types, coopmat2 8-byte stride alignment, shape and mask semantics. Merged upstream #30003 extends coopmat2 sparse gather. No separate PKC04 attention lifecycle or transplant; QFP25 owns a different HIP QSA selected-index kernel. |
| Vulkan TP collective/transport | PRVP03/1290; Meta `ggml-backend-meta.cpp::allreduce_fallback`; TRVP16 and RRVP05 qualification | Evaluated default-off `BIGCHERRY_VK_ALLREDUCE=host-f32` SPI proves wiring, not a fast transport. Pinned Meta zero-fills `!GGML_TENSOR_FLAG_COMPUTE` ranks before generic fallback; 1290 reads every rank. PRVP03 owns a **prewrite inactive-rank reject**; returning false after any output mutation is unsafe because Meta then runs fallback. Upstream #25051 is a competing mapped-host/timeline implementation, not merged/qualified. No second provider SPI. |
| Runtime stack/device/ICD identity | RRVP02, TRVP12; `tools/bigcherry/backend/identity.py::ResolvedStackIdentity` | Canonical software/provider fingerprint exists. RRVP02's Vulkan probe/RuntimeCapabilitySnapshot and real process-bound ICD/device receipt remain incomplete/paused. Device ordinal is not physical identity (two gfx1100 XTXs). |
| Vulkan tuning candidate identity | TRVP14/15; `tools/bigcherry/vk_autotune_types.py` | `vk_hardware_digest` includes device traits, driver, extensions, limits and shader toolchain; `vk_signature_digest` is operation-specific. Module explicitly says data-model only: **no runtime dispatch, writer or patch**. These digests are **not** `ResolvedStackIdentity.fingerprint()`, nor proof of an executed shader/collective. Keep candidate keys and stack identity separate; never copy capability fields into the software-only stack key. |
| Contract, execution attestation, campaign/replay | EC16/EC19, PKC03, RRVP02/RRVP05, TRVP13, `experiment/contract.py`, `experiment/attestation.py` | Contract hash expresses intent, not selected execution. Existing attestation verifies real ROCm process/device, but cannot be claimed to verify Vulkan ICD/actual pipeline until its producer exists. TRVP13 owns replay/inspection; PKC03 owns any cross-backend collective-evidence representability decision. No PKC04 evidence namespace. |

**Source-level failure boundary:** Meta calls `comm_allreduce` and, when it returns false, immediately executes `allreduce_fallback`. A provider's requested setting, an attempted route or a process-once activation marker cannot certify a completed collective. 1290 synchronizes/reads every rank and lacks the inactive-rank guard; this is a known **source-confirmed correctness gap**, not a reproduced GPU failure. Leave PRVP03 untouched under its own ownership; do not expand PKC04 into a transport patch.

**Identity boundary:** `backend/identity.py` uses schema `bigcherry/resolved-stack-identity/v1` and fingerprint domain `bigcherry/backend-stack/v1`. The Vulkan candidate helper uses separate `llama-vk-hw` / `llama-vk-tune` digest domains and currently has no runtime integration. A device/ICD change must invalidate unsafe replay, but the candidate hash alone cannot attest which physical GPU or pipeline executed. TRVP12/13 must consume the shared identity and process-observed capability evidence rather than invent another global identity.

## Measured evidence and external mechanism decision

RRVP05's historical 2026-10-02 single-screen dual-XTX 27B Q8_0 RADV results: Vulkan layer split 882 pp / 20.8 tg versus HIP 1086 / 23.8; Vulkan tensor split 494 pp / 24.2 tg versus HIP 1479 / 38.5. These are unmatched backend+collective paths, **not** an isolated Vulkan kernel regression or a transferable speedup. PRVP03's historical 1290 provider ON 259 pp / 20.1 tg versus stock fallback OFF 496 pp / 24.6 tg; host-F32 provider was slower. No new GPU result.

- [llama.cpp #30003](https://github.com/ggml-org/llama.cpp/pull/30003), merged 2026-10-08: inspected actual Vulkan sparse FA shader, 8-byte coopmat2 alignment guard and backend-op tests. Already present at b11552. **Use as native attention correctness baseline**, not a new PKC04 kernel or RDNA performance claim.
- [llama.cpp #29998](https://github.com/ggml-org/llama.cpp/pull/29998), merged 2026-10-11: compact `MUL_MAT_ID` expert tile prepass. Inspect on a future pin via TRVP12/14; no PKC04 scheduler.
- [llama.cpp #25051](https://github.com/ggml-org/llama.cpp/pull/25051), open, head `99becee6`, updated 2026-10-08: mapped-host/timeline-FD collective provider; actual diff changes Vulkan context semaphores and communication path. **Wait/compare under PRVP03**, check inactive ranks, driver support and post-submit failure semantics before adoption. External NVIDIA benchmark is not AMD evidence.
- vLLM ROCm's transport/attention paths are HIP/Triton rather than a Vulkan implementation; portability of identity/negative controls is useful, but no direct Vulkan patch or NVIDIA speed ratio transfers to Brutus.

## Bounded closure and downstream gates

**PKC04 is terminal `done` for ownership:** every attention, collective, transport, identity, campaign and replay responsibility has an existing owner. Reopen only if a *concrete, executed* Vulkan semantic cannot be represented by those owners and the existing contract/evidence fields; show one minimal counterexample and assign it to an existing owner before proposing a new surface.

**No Vulkan implementation/hardware queue from this audit.** When RRVP02's pause is independently lifted, the next cheapest existing-owner gate is: (1) source/driver/ICD/physical-device receipt from the measured process; (2) requested-versus-actual shader/provider and fallback receipt, including negative unsupported/changed-ICD controls; (3) PRVP03's prewrite inactive-rank and no-partial-fallback correctness fixture; (4) only then existing RRVP05 matched controls, multi-request/ubatch, long-context, logits/KLD/greedy/MTP, graph/descriptor lifetime and all-reduce byte/work accounting. Fail closed on missing proof; retain stock Meta fallback. Promotion thresholds belong to TRVP15/PRVP03, not PKC04.

**Audit validation:** 14/14 actual-repository-source assertions, 13/13 disposable host scope/receipt tests and Python compilation passed. No BigCherry repository pytest, Vulkan/SPIR-V compilation, GPU execution, benchmark, or new performance measurement. Host model is not a device correctness proof.

## Historical provenance

Supersedes KC04; inherited RV108/RV113/RV124. The 2026-09-10 successor originally deferred this scope decision until TRVP12–15; those owner boundaries are now explicit even though their runtime implementation/qualification remains pending. Historical drafts and migration steps remain in Git history.

## Change Log

- 2026-09-09T10:51:56.429711+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:24.078150+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.048940+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.683688+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:50.713702+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.031122+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:47:46.421161+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.200144+00:00 (updated-by): Updated: section:ledger-events
