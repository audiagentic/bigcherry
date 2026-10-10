---
id: PNRO03
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-09T10:52:12.743676+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Correctness-first HIP P2P transport provider for internal AllReduce

## Description

**Terminal disposition: closed (2026-10-09).** Patch 1252 was rejected at b11474 on 2026-10-08 after the 2026-10-07 hardware fault. Do not enable, repair, benchmark or queue this direct-P2P AllReduce provider on the current no-P2P topology. Exact-F32 host-staged/RCCL routing (PGC09/PGC12) remains the production control.

The stored `evidence/validation.json` reports bit-identical full-vocabulary outputs but **failed activation** and `NRO03-ALLREDUCE-P2P passed=false`. This proves fallback equivalence, **not** P2P correctness or performance. Historical `ab-27b-p2p.json` has no independent positive route-completion proof.

## Steps

1. Keep `1252_nro03_allreduce_p2p_provider/patch.toml` rejected and default-off. Do not queue additional tests on the failed topology.
2. Preserve historical negative evidence and production host/RCCL fallback.
3. Reopen only if a materially new topology/driver and supported production workload justify a fresh correctness campaign.

## Detailed Solution & Technical Design

In `ggml/src/ggml-cuda/allreduce.cu`, as introduced by patch 1252, `ggml_cuda_ar_p2p_probe()` uses fresh `cudaMalloc` buffers and four fixed copy sizes, synchronizing and comparing host readback. Actual `ggml_cuda_ar_allreduce_p2p_impl()` uses `p->dev_tmp`, source-owned streams, cross-device `p->ev_pool`/`p->p2p_done` events and ring slots. Startup probe success therefore does **not** establish production collective safety. This is a source-derived coverage gap, **not** a demonstrated cause of the hardware fault. The permanent `test_p2p_copy_correctness.py` uses `hipMemcpyDefault`, not 1252's production peer-async copy.

**Conditional re-entry (not queued):** prove both directed copies through actual production scratch, streams and event slots at each dispatch threshold ± one element; bind topology, driver, ROCm, runtime hashes; require positive provider-completion marker, multi-request same-process, graph replay, no crash, full-vocabulary bit identity and exact work accounting. Only then compare matched P2P against host-stage/RCCL for prefill/decode/MTP. Fail closed on any missing activation or incorrect byte. Do not add a second scheduler, selector, allocator or transport.

## Files

`engines/llamacpp/patches/1252_nro03_allreduce_p2p_provider/`; `tools/tests/hardware/test_p2p_copy_correctness.py`; `docs/evidence/lab-run-summaries/ab-27b-p2p.json`; PNRO18; PGC09/PGC12.

## Validation

Ten pinned-source/evidence static assertions and sixteen disposable fail-closed admission fixtures passed. No repository test suite, HIP compilation, GPU execution or new benchmark ran.

## Effort & Risk

Terminal on existing hardware. Capability bits and isolated copy success do not qualify production P2P.

## Standards

Preserve rejected state, negative evidence and exact-F32 host/RCCL fallback. Upstream llama.cpp #27825 enables native HIP internal AllReduce, not direct-P2P qualification. #21648 documents missing-peer-access corruption. External vLLM/dual-R9700 P2P results do not transfer automatically to the no-P2P mixed RDNA3/4/2 topology.

## Acceptance Criteria

PNRO03 and PNRO18 done; patch 1252 rejected; no new campaign; no fallback equivalence mislabelled as P2P success.

## Notes

Supersedes: NRO03
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro03

Supersedes: NRO03
Inherited semantic scope: preserve source-current push direction, probe/fallback, no direct peer reads, and correctness-first acceptance.
Migration: capability-rebaseline-v3-2026-09

REAL FINDING 2026-09-12: found the identical compile-breaking anchor bug fixed in patches/1250 (NRO01) this same session, by inspection (same author/batch, same pattern: anchor ending at '=' mid-statement). Fixed proactively and re-verified with a real isolated gfx1100 build (1001+1252 composition): clean build, BUILD_OK. Closes this draft's 'applies cleanly, builds HIP' bar for the first time. All other acceptance criteria (bidirectional probes, scratch ownership, fallback, content-checked performance campaign) remain entirely unstarted.

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1252_nro03_allreduce_p2p_provider exists, state=untested. Prior session (2026-09-12) fixed the same anchor/compile bug class as PNRO01 and confirmed clean gfx1100 build (1001+1252 composition). No upstream equivalent. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1252 only has an env flag and a bare cudaMemcpyPeerAsync helper -- no probe/scratch/selector/fallback/marker exists. Added the required peer-probe/enable step in ggml_cuda_ar_pipeline_init(), source/destination stream+scratch ownership, the actual P2P-selection gate, a required activation marker (none existed), and removed the stale 1001 requires composition.

2026-09-25: IMPLEMENTED (commits bd2fc067/45219a0d). patches/1252 is now a real port of 7c5bb5cb adapted to this item's invariants: per-direction streams/events on the SOURCE device, set_device(source) before every cudaMemcpyPeerAsync, no issuer, ggml_cuda_ar_p2p_probe (4 sizes x 2 directions x 2 passes, byte compare) gates GGML_CUDA_AR_P2P. Marker patch=1252_nro03. Rebase CLEAN at b11126. Next: build + hardware arms per TESTING.md (P2P on vs off, same binary, 2x gfx1100).

2026-10-08: TERMINAL REJECTION at b11474. Patch 1252_nro03_allreduce_p2p_provider is rejected: P2P AllReduce faults on the lab topology (re-tested 2026-10-07). The ar-p2p experiment and spent queue-p2p-accuracy.sh were removed. Historical evidence above is retained.

## Change Log

- 2026-09-09T10:52:12.743676+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:41.098537+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.061978+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.701391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:09.609126+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.291162+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T03:47:57.075297+00:00 (updated-by): Updated: section:notes
- chg_20260912_034828_confirmed-on-real-hardware-tha_4451
- 2026-09-12T03:48:28.669796+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:08.096956+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:48:10.768762+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- chg_20260925_111345_real-ports-of-the-nasone-allre_4524
- 2026-09-25T11:13:54.348499+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-25T11:14:00.720422+00:00 (updated-by): Updated: section:notes
