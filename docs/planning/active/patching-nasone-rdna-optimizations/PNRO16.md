---
id: PNRO16
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-11T04:32:00.691672+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Offline expert-placement compiler — superseded, no new implementation

## Disposition (2026-10-10)

**Close PNRO16 without a new tool or patch.** The original September proposal for eight modules under `tools/lab/moe-expert-placement/` has been overtaken by implemented, model-tested BigCherry mechanisms. The proposal's unresolved `GGUFReader` import, target-model selection and evidence-location questions are now answered by existing owners; implementing another inventory/compiler/replay tool would duplicate those owners. Historical design and earlier notes remain in Git history. This closure is planning-only, not a claim that a new placement algorithm was benchmarked.

## Implementation and evidence traced

- `tools/lab/flash-next/expert-place.py::{read_profile,place,main}` already imports `GGUFReader` from `vendor/llama.cpp/gguf-py`, consumes the STRP ranked `(layer,expert)` profile, and deterministically emits `--plan-only` JSON containing `caps`, `traffic`, `n_layer`, `n_expert` and per-layer `order`. Its full-copy path reorders the three expert-weight tensors and router rows by whole byte blocks; it does not requantize. It is the existing MET08/MET11 oracle.
- `tools/lab/strata/routing-balance.py` already replays per-token selected expert IDs from Strata routing traces and reports device-load imbalance for contiguous and usage-placed partitions. Do not create a second replay estimator.
- `engines/llamacpp/patches/1281_moe_mul_mat_id_range` and `1283_qwen4exp_expert_parallel` provide range-aware device-local execution and delayed single-AllReduce semantics. MET04 owns their correctness and hardware qualification; MET08 owns placement performance; MET11 owns future load-time permutation and old-to-new ID translation; MET09 owns numerical drift; MET07 owns cache/profile; RPL01 owns cross-capability topology cost evidence.
- MET08 reports a real UD-IQ4_XS 128/128/256 placement fitting at ctx 245760 after 1341, with 98K prefill 975.6/974.6 versus 956.5/988.2 t/s and decode 60.7/60.6 versus 59.2/58.9 t/s. Acceptance differed. These are first-party observations, **not** a PNRO16 speedup or a controlled placement-policy win. The 6900 XT remains an auxiliary, non-P2P GPU; rejected 1328 auxiliary transport is not a placement dependency.

## Remaining gaps — transferred to MET11, not a new PNRO16 implementation

The existing `--plan-only` contract has no GGUF/shard/model identity and only expert-count capacities; STRP stores **rank order**, not per-expert frequency weights. The tool's admission check `sum(caps)==n_expert` plus `min(traffic)>0` fails to reject negative capacity and non-finite traffic inputs. A host reproduction accepted `caps=[-1,7]` for six experts and `traffic=[NaN,1]` with caps `[3,3]`. A six-expert synthetic ranking produced per-device frequency totals 181/161 under equal-share rank-only placement versus 171/171 for a separate weight-aware greedy control; this is a **host-model counterexample**, not a GPU benchmark or a replacement algorithm.

MET11 must own fail-closed profile/model/shard binding, finite-positive traffic and nonnegative-capacity validation, complete permutation/inverse mapping, real per-device byte/fit accounting, and comparison with the existing `--plan-only` oracle before any opt-in loader patch. The proposed `1343_moe_expert_placement` ID is unavailable: `engines/llamacpp/patches/1343_mtp_nextn_rereserve/patch.toml` already reserves 1343 (state superseded). Allocate a fresh ID only if MET11 reaches implementation admission.

No new scheduler, runtime placement solver, cache, registry, CLI, GGUF mutation, or hardware experiment is authorized by this closure. Do not modify recently active Radiance, Flash-Next accuracy or Meta dispatch work.

## Acceptance / terminal condition

Terminal: duplicate ownership eliminated; PNRO16 has no independent implementation remainder. Reopen only if MET11 and the existing routing/placement tools demonstrably cannot represent a required **measured** topology or format, with an explicit non-overlapping owner and a >=3% plausible end-to-end benefit. Otherwise keep closed.

## Validation and provenance

Source-static inspection of `expert-place.py`, `routing-balance.py`, 1283, MET08/MET11, 1338 and patch-1343 metadata; three deterministic Python host cases for ordinary, negative-capacity and non-finite-traffic admission, plus a synthetic rank-only versus weight-aware load comparison. No repository pytest, GGUF model run, HIP build, GPU execution or hardware benchmark occurred. Upstream llama.cpp #29887 merged 2026-10-07 (host-expert GPU cache), #29963 remained open 2026-10-09 (host-weight pipeline); neither is a replacement for MET11's load-time map. vLLM EPLB `balanced_packing` uses weighted load rather than rank alone; its dynamic expert replication and NCCL assumptions are not portable to no-P2P RDNA without separate qualification.

## Change Log

- 2026-09-11 to 2026-09-24: original offline placement compiler proposal and readiness reviews (preserved in Git history).
- 2026-10-10: superseded by MET04/MET08/MET11 and existing tools; terminal consolidation recorded in BCOP101.
