---
id: VTP39
order: 0
plan: run-dense-27b-virtual-tp
state: completed
created-at: '2026-11-14T06:57:52.821543+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Long prompts beside other work: decode share, prefill concurrency, short requests (-long configs)

## Description

With 262,144-token sessions a prefill lasts about a minute and the scheduler's defaults were set for 8192. Measured on `qwen27b-tp3-int8-556-long-bf16` (single runs, 2026-11-14): four decoding streams get 5.0 chunks/s while a 100k prompt prefills, with gaps up to 1.55 s; `prefill_interleave 2` gives 13.5 chunks/s for +8% on the long prompt; `busy_prefill_chunk 512` caps the gap at 0.68 s for +46%. Eight concurrent 100k prompts into a 624k-token cache all completed (6 preemptions, 4 forced; over-length prompts get a clean 400, a full admission queue a clean 429) but saw first tokens at 377-564 s where one at a time would average ~245 s; four short requests waited ~330 s for a slot and then got one token per ~1 s shared step. Recompute after preemption was <= 7% of the steps: the cost is round-robin and slot starvation. A prefilling runner is never a preemption victim, so request priority alone cannot free a slot. Every lever is a manifest field; none names a model.

## Steps

1. Settings only, on the interference and full-cache clients: `prefill_interleave` 2 / 4 / 8; request `priority` on short requests; long prompts submitted at most 1 or 2 at a time (emulating a cap). 3x ABBA for the chosen values.
2. Decode share by time rather than count (managed scheduler patch): after a prefill chunk, decode-only steps until a configured fraction of the chunk's measured duration has been spent.
3. A cap on sequences prefilling at once (manifest field, default = today's behaviour): the rest wait in arrival and priority order; decoding sequences keep their slots; KV reserved when a prompt starts.
4. After RAD52: `busy_prefill_chunk` by step time (chunk sized so the step stays under a configured duration at the sequence's depth).
5. Named variants in `configs/qwen27b.toml` for the chosen settings (RSC12 rules).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Interference client (long TTFT, decode chunks/s, gap p50/max) and full-cache client (per-request TTFT, preemptions, forced preemptions, steps); card-free scheduler tests for new fields; int8 stress.

## Effort & Risk

M. Steps 2-3 are scheduler patches; a prefill cap trades the long prompts' fairness for mean latency (owner's call, hence a field).

## Standards



## Acceptance Criteria

- [ ] With 4 streams decoding beside a 100k prefill: decode at least 3x base chunks/s with the long TTFT within +10%, and a recorded gap max. **Not met by one setting:** interleave 2 gives 2.7x at +9.8% (3x costs about +12%); gap max 1.15 s at every setting. Owner accepted interleave 2.
- [x] Eight 100k prompts: mean TTFT at least 35% lower than round-robin with no forced preemption: cap 1 -52%, cap 2 -38%, 0 preemptions.
- [x] A short request arriving beside long prefills gets its first token within a few steps: 3.5-5.6 s with the cap (5.5 min without).
- [x] Defaults unchanged until ABBA and the owner's choice: owner chose interleave 2 and cap 2 for the `-long` configs (2026-11-15, #974).

## Notes

Created 2026-11-14 (`docs/evidence/2026-11-14-tp3-long-context/README.md`). Step 4 depends on RAD52. Related: VTP36 step 9 (lone-prefill wide steps), RSC12.

Step 3 implementation: managed engine patch `rad0034_prefill_concurrency` introduces `--max-prefill-seqs N` (default 0, unlimited) and an optional inactive serving-v2 tuning field. Only unfinished RUNNING long prefills count; existing runners continue, preempted requests retain the whole-recompute gate, decoders are unaffected, and a short request which completes its prompt in the admission step can bypass a full cap, including after a prefix hit. The prefill-interleave hold, busy-chunk reserves, lone-prefill span and hard sequence/card limits keep their existing semantics. Patch state remains untested: next are G4 current-pin requalification, card-free scheduler and contract checks, then the full-cache eight-long-prompt ABBA of N=0/1/2 with short-request probes and the owner decision on a default; do not enable it in a serving config beforehand.



**2026-11-15 hardware (`docs/evidence/2026-11-15-vtp39-prefill-concurrency/README.md`, -long-bf16, two starts an arm).** Step 1 interleave: 4 decoders beside a 100k prompt 6.0 -> 16.0 / 24.4 / 37.5 chunks/s at 2 / 4 / 8 for +9.8 / +19 / +39% on its first token; 7 decoders at 2: 2.6x for +14%; a lone prompt and eight at once unchanged. Owner chose interleave 2 as the `-long` default. Step 3 cap: eight 100k prompts mean first token 434 s -> 208 s (cap 1, -52%) / 265 s (cap 2), preemptions 6 (4 forced) -> 0. A fix in rad0034: prompts whose remainder fits one shared step bypass the cap at any share (the first version made ~450-token requests wait ~45 s; now 3.6-4.5 s). Criteria: 1 not met by one setting (2.7x at +9.8%), 2 and 3 met with cap 1; cap default is the owner's decision. Step 2 (decode share by time) not built: the share/cost trade is linear, so it would move along the same line. Corrected: patch state `untested` needs no G4 record to merge (G4 applies when changing a patch that has evidence).



**2026-11-15 close.** Owner chose a cap of 2 for the `-long` configs (two long prompts prefill at once rather than strictly in turn). Cap 2 on the fixed build (`fullcache3`, PQQP): mean first token 268 s (-38%), first / last 82 / 443 s, short requests 4.6-5.6 s, 0 preemptions; adding interleave 2 leaves it at 268 s. `configs/qwen27b.toml` `[config.qwen27b-tp3-int8-556-long.tuning]` now sets `prefill_interleave = 2` and `max_prefill_seqs = 2` (merged in #974; evidence update #976). Steps 2 and 4 not built: the measured share/cost trade is linear, and the worst gap (1.15 s, one 2048-token step at depth) is the only remaining stall; reopen as a new item if step-time sizing is wanted.

## Change Log

- 2026-11-14T06:57:52.821543+00:00 (created-by): Created by agent
- 2026-11-14T06:58:44.043856+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-15T05:59:26.035868+00:00 (updated-by): Updated: section:notes
- 2026-11-15T07:40:43.322564+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-15T07:40:47.387247+00:00 (state-transition): State: pending → completed
