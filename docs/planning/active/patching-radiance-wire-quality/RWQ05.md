---
id: RWQ05
order: 5
plan: patching-radiance-wire-quality
state: pending
created-at: '2026-10-15T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Hardware A/B: quality, MTP, PCIe bytes and latency for wire candidates

## Description

No codec change is good merely because CPU rel-L2 falls. Qualification must directly measure model logits KL, top-1, PPL, decode acceptance and Brutus throughput, with common frozen configurations. The current WHT6 failure is a 0.0746 mean KL / 87.18% top-1 Flash-Next self-reference result over 42,761 positions; expected prefill upside over exact on Flash-Next is only about 5-6%, not 27B's 38%.

## Steps

1. Preregister reference and candidates: exact (A), WHT6 (B), WHT8 (C), policy-mixed family/layer variants (D), optionally residual (E) only if device-correct. All same .rad, same model load, same serving flags aside from wire settings; use pinned source and explicit release identity. On same model repeat A/A for deterministic floor.
2. Record 42,761-position reference corpus IDs and token position masks, plus held-out code, tool calls and long-context transcripts. Use identical raw tokens and scoring positions; log stable per-position KL, percentiles, top-1, PPL, decoded greedy divergence, number of positions changed, and max tail. Never mix model-weight KL from BF16 with runtime wire KL.
3. Prefill: 2.6K, 7.2K, 24K+ where model fits; batch/chunk widths 64,512,1024,2048,4096; separate no-MTP and MTP3 256-token greedy decode, MTP acceptance, context depths 8K/32K/long configured; multiple prompts including sparse/outlier-rich. Test Flash-Next and 27B separately and stratify by operation families.
4. Paired ABBA/BAAB, >=3 complete pairs after warmup on same GPU reservation; collect median and spread of prefill tok/s, per fused collective latency and payload, actual active GPU utilization, memory use, host pinned scratch, stream stalls and device errors. Do not combine profiled sync timing with unprofiled throughput. Record GPU clock and contention checks.
5. Candidate initial adoption gate (owner may tighten, do not silently weaken): Flash-Next mean KL <=0.005 relative to exact-wire reference; top-1 >=99%; PPL within ±0.5%; p99 KL <=0.05 with p99.9 and worst tail reviewed; >=3% repeated prefill throughput over exact with no material decode/MTP regression. Also compare per-corpus categories, not only aggregate. If a numerically safer mode yields no speedup, default remains exact. 27B has independent gates.
6. Compare speed against payload ceiling: WHT8 wire payload is 51.5625% exact and 1.32x WHT6; because both per-op compute and transport matter, measured speed may be lower/higher than byte-only expectations. Inspect whether fused post-ops lose dispatch coverage on exact fallback.

## Detailed Solution & Technical Design

Use existing engine-specific profile and ab-benchmark plus engine_quality.py: quality traces are first-class gate, activation markers prove that candidate is reached. Profiles must describe only changed values; do not create a profile copied verbatim from defaults. For synthetic proxy use RWQ01 only as preparatory evidence. Keep code and environment hashes for all arms, including auxiliary R9700 helper enabled/disabled; cross-rank helper implementation itself has known independent quality sensitivity and must not be changed within a wire experiment. Freeze --tp-wire exact as Flash-Next production control. The existing measurement tools can score KL; the real CLI path must be checked on Brutus before automation. Keep raw logs in artifacts/RWQ05-*, compact receipts in docs/evidence/RWQ05.

## Code Samples & Guidance

Experiment axes: model × wire × context-depth × chunk × operation-family-policy × MTP × helper. To control combinatorial explosion: phase 1 exact/WHT6/WHT8 one representative depth; phase 2 isolate sensitive family; phase 3 run full ABBA only on survivors. Publish per-arm support matrix, not just winner.

## Files

tools/lab/radiance/wire-quality/ hardware wrapper only if existing ab-benchmark config cannot express it; engines/radiance/recipes.toml experiment profiles; tools/bigcherry/experiment/engine_quality.py only if engine-neutral quality metrics need extending; docs/evidence/RWQ05/; engines/radiance/README.md after measurements.

## Validation

Checks: exact rerun KLD exactly zero on fixed build; exact post-op checker tests; verified chosen wires through log markers and device traces; rank-order consistency; speed separation; long-running serving stability. A/B campaign should fail on missing evidence rather than accepting no candidate activation. Report mean/median/p99/p99.9/max KL and top-1 both overall and by dataset.

## Effort & Risk

High GPU-time and confounding risk: helper remote results, MTP, quantised KV and native kernels can each change output. Enable one independent factor per A/B. KLD threshold is provisional; no invented results or unearned promotion.

## Standards

- Target Radiance 1.3.0 pinned at 89cee7ce414486c717aa2085c99e1f522e4c0ea6; no unpinned edits.
- Engine edits are explicit Radiance patch packages; gfx1100 kernels and settings belong to the libr3 component. Keep opt-ins OFF unless independently promoted with quality evidence.
- Use engine-specific recipes, source/build identities, runtime profiles containing only settings different from declared defaults, and existing bigcherry ab-benchmark/engine_quality evidence facilities. No second campaign or loader.
- Follow docs/reference/tooling/TOOLING.md, docs/reference/engines/RADIANCE_RUNBOOK.md, patch mechanics, relevant device checker, and main-branch code review requirements.
- Changes must preserve both-rank numerical agreement, graph replay and no-P2P mapped-pinned-host protocol; fail closed on missing kernel support or disagreement.

## Acceptance Criteria

- Controlled complete hardware evidence for exact, WHT6 and every attempted candidate; full provenance.
- Flash-Next selected winner clears stated quality and speed gates; otherwise document exact as winner without forced adoption.
- 27B independent result prevents changing its production WHT6 policy based solely on Flash-Next.
- Scores, worst tails, MTP acceptance and link bytes shown together.

## Notes

Depends RWQ01 baseline and RWQ02 candidate, plus RWQ03 if selective policies are tested; monitor device reserve and host scratch memory. Do not equate model's original BF16-to-4bit publisher KL with wire transport KL.

## Change Log

- 2026-10-15T00:00:00+00:00 (created-by): Initial plan; unimplemented GPU work remains pending.
