---
id: VTP39
order: 0
plan: run-dense-27b-virtual-tp
state: pending
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

- [ ] With 4 streams decoding beside a 100k prefill: decode at least 3x base chunks/s with the long TTFT within +10%, and a recorded gap max.
- [ ] Eight 100k prompts: mean TTFT at least 35% lower than round-robin with no forced preemption, or the owner's decision to keep round-robin recorded.
- [ ] A short request arriving beside long prefills gets its first token within a few steps.
- [ ] Defaults unchanged until ABBA and the owner's choice.

## Notes

Created 2026-11-14 (`docs/evidence/2026-11-14-tp3-long-context/README.md`). Step 4 depends on RAD52. Related: VTP36 step 9 (lone-prefill wide steps), RSC12.

## Change Log

- 2026-11-14T06:57:52.821543+00:00 (created-by): Created by agent
- 2026-11-14T06:58:44.043856+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
