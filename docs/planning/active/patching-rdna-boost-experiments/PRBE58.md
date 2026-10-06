---
id: PRBE58
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:31.760072+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-MTP-002: Remove four-copy Vulkan MTP handoff

## Description

TODO, corrected per GPT review (req_b43762f844fb40b3). Source/mechanism was wrong: 1fcc05da does NOT implement the proposed P2P/host-bounce mtp_handoff paths (that commit is PRBE57's/RD71's Vulkan placement change) -- the actual backend-resident hidden-state transfer commit is 41a8ca78. The prior draft invented an implementation rather than porting the cited fork commit. Relevance unchanged: no existing patch for RD72 (grep = no hits).

## Steps

1. Base PRBE58's handoff work on 41a8ca78 (not 1fcc05da -- that belongs to PRBE57), rebased after PRBE57's package if preserving fork commit order. 2. Concrete b11126 anchors to verify before use: src/llama-batch.h immediately after `std::shared_ptr<data_t> data;`; src/llama-graph.h constructor `llm_graph_input_embd_h(int64_t n_embd);`; src/llama-graph.cpp::llm_graph_input_embd_h::set_input at the hidden-state `if (ubatch->embd)` block; src/llama-context.cpp::set_embeddings_nextn; Qwen35/Qwen35MoE `auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);` and `res->t_h_nextn = cur`. 3. Re-port 41a8ca78 SEMANTICALLY, not verbatim -- its verify_pos/pending_pos/process_enabled speculative-decode state does not exist in b11126 in that shape; the port must adapt to b11126's actual speculative-decode state machine, not assume 41a8ca78's structure carries over unchanged. 4. Verify every anchor above against the live b11126 mirror before writing patch.py (none independently re-verified this session -- same caveat as before, now attached to the correct commit).

## Detailed Solution & Technical Design

This item is explicitly downstream of / coupled with PRBE57 (RD71): a copy-reduction design chosen independent of NextN's placement decision could contradict it (e.g. optimizing away a copy that RD71's chosen placement would have made moot anyway, or vice versa). This plan therefore treats PRBE57 as a co-requisite to finalize alongside, not a separate independent task -- whoever implements should read both plans together and likely land them as one coordinated change (or two patches with an explicit `requires` relationship in patch.toml, per this project's dependency-declaration convention) rather than two fully independent packages.

## Code Samples & Guidance

No real anchors available (fork not locally mirrored, same as PRBE57). Once fetched, follow the same provenance-dict/PATCHES-list convention as patches/1203_.../patch.py or patches/1215_.../patch.py.

## Files

To be determined after fetching the fork commit; likely Vulkan backend buffer-transfer code and src/llama-context.cpp's speculative-decode loop. New package patches/<order>_rd72_mtp_copy_reduction/ (only after fetching source; consider making it depend on / be sequenced with PRBE57's package via patch.toml `requires`).

## Validation

Fetch prerequisite (step 1) blocks further validation design. Once designed: hidden-state/output bit-identity vs the 4-copy baseline (mandatory, zero tolerance), copy count/bytes/time + effective TG + PP + VRAM across path variants/draft depths/contexts on dual RDNA4 Vulkan (Brutus, not run here), single-GPU controls.

## Effort & Risk

Unscored by item; set to M -- correctness bar is strict (bit-identity) but the change is confined to transfer/copy mechanics, not compute; real effort risk is coordination with PRBE57's coupled decision, not code complexity.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; coupled RD71/RD72 analysis required (explicit item requirement); keep host-staging/four-copy fallback if correctness or VRAM/PP cost changes for the worse.

## Acceptance Criteria

Promote only if copies fall without hidden-state/output changes, PP/VRAM cost increase, or acceptance loss; retain fallback otherwise.

## Notes

Supersedes: RD72
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd72

2026-09-24 relevance at b11126: no existing patch for RD72 (grep = no hits); no functional overlap found. External fork source (commit 1fcc05da) not locally available -- same blocker as PRBE57, documented as mandatory step 1. GPT design request req_83fbfa0995034d2d (covering this + PRBE57/59/101) was in progress when this plan was authored; check for its response and merge/reconcile if it landed with fork-specific detail.

2026-09-24 GPT req_83fbfa0995034d2d COMPLETED. Its design: a context-owned `mtp_handoff` struct with 4 prioritized paths (same-owner-GPU alias/0 copies; P2P direct/1 copy; host-bounce/2 copies; existing baseline fallback), explicitly contingent on PRBE57's placement choice (matches this plan's own coupling requirement), persistent/reused buffers (no per-token transient Vulkan staging allocation -- a concrete improvement over this plan's vaguer 'reduce copies' framing), and patch.toml `requires=["<PRBE57 package id>"]`. Test bar: byte-identical hidden state and temp=0 tokens, forced-path tests (force-direct/force-bounce/force-baseline), Vulkan copy-command + byte/token tracing targeting <=2 (ideally 0/1). Prefer this design when implementing; anchors still unverified against the real fork diff (same caveat as PRBE57).

2026-09-24 GPT review req_b43762f844fb40b3 applied: corrected commit attribution (this item is 41a8ca78/backend-resident hidden-state handoff, not 1fcc05da which belongs to PRBE57); added concrete b11126 anchor candidates (llama-batch.h, llama-graph.h/.cpp, llama-context.cpp, Qwen35(MoE) graph-build sites) and flagged that 41a8ca78's speculative-decode state shape does not exist in b11126 and requires semantic re-porting, not verbatim copy.

## Change Log

- 2026-09-09T10:57:31.760072+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:43.820256+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.389761+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.197952+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:26.102767+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.510818+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:52:14.738266+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:53:38.311212+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:09:30.649556+00:00 (updated-by): Updated: section:description, section:steps, section:notes


## 2026-10-07 optimisation audit: measure the handoff before porting it

PRBE58 remains the owner of **embedded-MTP hidden-state handoff traffic only**. The older four-path design is too broad for the current tree and hardware: on the production topology normal GPU peer access is unavailable, so a generic P2P branch is not a useful first implementation, while same-device aliasing can only help if target hidden-state production and the MTP consumer are already co-resident. PRBE57 owns placement; PNRO10/1261 owns ctx_other scheduler-backend coverage for genuinely different draft-model device lists. Do not duplicate either mechanism here.

Current upstream evidence further narrows the boundary. llama.cpp PR #26636 adds ctx_other model backends to a speculative context so shared pre-allocated tensors can be scheduled when target and draft device lists differ. BigCherry's PNRO10/1261 is the same class of mechanism, and its 2026-09-30 evidence says it does **not** activate for embedded MTP on a tensor split because the draft shares the main model devices. Therefore adding scheduler backends is not a hidden-state transfer optimisation for the production embedded-MTP lane. Upstream #26827/#28252 also show that multi-ubatch MTP has a real mutable-state ordering constraint: do not overlap or alias handoff storage across unfinished ubatches merely to reduce copies.

### Cheapest discriminator

Before any 41a8ca78-derived port, add observation-only accounting at the existing target-hidden-state -> MTP input boundary. For each speculative round record:

- producer backend/device and consumer backend/device;
- hidden-state row bytes and rows transferred;
- number and direction of backend copies plus host-visible staging copies;
- copy submit/completion time and synchronization attributable to the handoff;
- whether source/destination storage aliases, remains device-resident, or crosses host memory;
- MTP depth, accepted/drafted tokens, context depth, and ubatch count.

Run MTP depth 1/3/7 at shallow, ~80K and >=160K context on gfx1100/gfx1201 production ordering, plus single-GPU and MTP-off controls. Include a prompt that forces multiple internal ubatches. This is an attribution lane, not a performance claim.

### Decision / implementation gate

1. If measured handoff copies + waits are **<3% of MTP decode wall time**, close the copy-reduction sub-slice; do not port 41a8ca78.
2. If >=3% and producer == consumer backend with source lifetime covering the MTP graph, prototype only a borrowed device-resident view/reference. Ownership stays with the producing context; the consumer may not free/reallocate it. Generation/ubatch identity must prevent reuse while work is in flight.
3. If >=3% and producer != consumer, first let PRBE57 choose the home backend from measured topology. With no normal P2P on the production machine, use the existing backend-copy/staging primitive and persistent reusable buffers; do not add a private P2P transport. A direct device path may be architecture-gated only if runtime capability is actually reported and measured.
4. Preserve the existing baseline handoff as fallback. No new scheduler, allocator, placement table, or ctx_other backend enumerator belongs in PRBE58.

### Correctness and promotion

Require greedy/token identity, byte-identical handoff rows for a deterministic fixture, unchanged MTP accepted/drafted counts, multi-request same-process, forced partial rejection/rollback, and multi-ubatch long-context coverage. Count expected versus observed bytes/copies so a speedup caused by missing handoff work fails immediately. Promotion requires CI95-low-positive **>=3% end-to-end MTP decode improvement** on at least one production architecture with **<=1% regression** on the other production architecture and no prompt-throughput regression >1%. Otherwise retain baseline and close the sub-slice.

External references inspected for this audit: llama.cpp #26636 (ctx_other backend coverage; open, updated 2026-09-18), #26827/#28252 (MTP multi-ubatch serialization/order correctness), and MrLordCat/llama.cpp-rdna-lab current performance evidence showing MTP is valuable on dual-RDNA4 but strongly context/backend dependent. These are mechanism/correctness evidence, not BigCherry handoff measurements.
