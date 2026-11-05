---
id: RAD46
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-03T02:29:34.417938+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# RDNA2 tail helper degrades Flash-Next tail quality: KLD p99.9 +1.17 over the two-XTX helpers on 2.3% of routes

## Description

Found by the MEN26 current-pin quality capture (2026-11-03). Production Flash-Next serves the tail layout: R9700 engine, two 7900 XTX helpers, plus the 6900 XT (gfx1030, RDNA2) tail helper. engines/radiance/archs.toml gives gfx1030 'no matrix instructions', plugin libr2.so, formats w4nl64 only, rows-kernel. Against a freshly recorded engine-local reference (TP2 exact over the two XTX, radiance 1.14.0, origin/main 2f5fc39df, corpus rad-kld-corpus-4.jsonl, 8240 positions, deterministic: three captures bit-identical), measured as change vs the R9700-alone control:

| arm | KLD mean | median | p99 | p99.9 | top-1 drop | ppl ratio |
|---|---|---|---|---|---|---|
| two XTX helpers (devices 0 1) | +0.003 | +0.0006 | -0.034 | +0.41 | 0.36 pts | +1.2% |
| production tail (devices 0 1 3, MIBS 22528,22528,14336, PLACEMENT tail) | +0.007 | -0.0005 | +0.146 | +1.58 | 0.46 pts | +1.2% |
| limit (tools/lab/radiance/helper-kld.json) | 0.02 | 0.01 | 0.2 | 0.5 | 1.0 | 5% |

The two-XTX arm passes every limit; the tail arm fails only p99.9. The 6900 XT served 5,312 requests and 42,729 routes, about 2.3% of the roughly 1.8M routes each XTX served, yet adds about +1.17 nats to the worst 0.1% of tokens. The worst positions are the same hard tokens in every arm, clustered in corpus document kld-docs-02 (positions about 5000-5500, e.g. 5022-5026, 5409, 5456-5458, 6096): the tail amplifies existing error, it does not create new error. Evidence: runs/men26-kld/ (work root) {capture,capture2,capture3,capture-isolate} (run.json quality block, per-position *.kld.json.rows, helper logs).

## Steps

1. Attribute: re-run the tail arm with R3_MOE_WORKER_TRACE / per-request route capture over the outlier positions, and map each outlier token's routes to (layer, expert, helper). Confirm the outliers' tail-routed experts carry the extra error (compare per-layer partial outputs vs the XTX path using the existing shadow-compare diagnostics, rad0004 / moe-remote-diag, on the same requests).
2. Isolate the kernel: run the gfx1030 libr2 expert path for the identified experts against the reference fp32 expert computation (moe_fp8_ref_test-style, card-free where possible) and measure per-expert relative L2. Candidates: w4nl64 dequant precision, accumulation type or order in the rows kernel (no matrix instructions), FP8 activation handling on a card without native FP8, and down-projection reduction order.
3. Fix in the helper kernel (managed component code, never an upstream rewrite), or, if the RDNA2 path cannot reach parity, constrain which experts may be tail-owned (e.g. exclude the layers or experts that carry the error) and derive that exclusion from measured per-expert error, not a hand-picked list.
4. Re-measure with the MEN26 capture config (kld-isolate.json, three arms); the tail arm must pass all six limits.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

MEN26 KLD capture against the same recorded reference: the production tail arm passes all limits (p99.9 increase <= 0.5 vs the R9700-alone control), deterministic across two captures; plus a decode/prefill A/B showing the fix costs no tail-helper throughput beyond noise (decode and prefill reported separately).

## Effort & Risk



## Standards



## Acceptance Criteria

Production tail layout passes the helper-kld.json quality limits at the current pin; root cause documented with per-expert error numbers; no regression in decode or prefill.

## Notes

Interim production choice is the owner's: keep the tail layout (more expert capacity and context, known tail-quality cost) or serve on the two XTX helpers only (passes quality, less capacity). rad0002's promotion (MEN26) rests on the two-XTX evidence and is not blocked by this item: the issue is in the gfx1030 helper kernel path (moe-remote component and libr2), not in the engine hand-off patch.

2026-11-04 experiment archive: ten Codex variants (outputlane, rb16, groupfold, stageprobe, pairdecode, pairdpp, quadadd, quadasm, quaddpp, attribution) are archived as inert patches in experiments/radiance/RAD46-*; the session's original notes and reviews are in experiments/radiance/RAD46-rad46/. Outcomes: docs/research/radiance-history/RAD46-rows-kernel-accuracy-2026-11.md. Where to resume: quad-DPP ordered-assembly variant (quadasm/quaddpp) - passes hashes, gate/up 1084 us vs 710 us original; needs full KLD and the matched pp1024/tg512 A/B. Output-lane passes quality but costs 31.5% prefill. Evidence copied to runs/rad46-evidence/ (work root), one folder per variant.

2026-11-04 CORRECTION to the archive note above: the resume point is not just quad-DPP. Production (feat/rad46-accurate-rows, 90ecda12f) selects per stage by request rows: output-lane RB8 at or below derived crossovers (gate/up 309 rows, down 922), quad-DPP RB16 above. KLD on the 6900 XT tail passes, decode at parity, 4k prefill about 5-10% slower, large gate/up about 1.4x slower than the old kernel in isolation. Unmeasured: row-owned rowlane16/rowlane8 gate/up candidate (Codex report 3) - only if large-prefill speed matters. PR still needs r3_rad46_rows_policy_test declared in component.toml and tools/tests/release/test_bump_gates.py. Evidence runs rad46-speed2 under runs/ belong to the PR; do not touch.

2026-11-05 KLD of the 6900 XT tail on the v1.16.0 and v1.17.0 builds, same corpus and reference (8,240 positions; runs tail14g-kld-116, tail14g-kld-117, tail4g-kld-117 under the work root runs/):

| Build | Tail | Control p99.9 | Tail p99.9 | Tail minus control (limit +0.5) | Mean, p99, top-1, perplexity |
|---|---|---:|---:|---:|---|
| v1.16.0 | 14 GiB | 2.376 | 2.245 | -0.13, pass | inside limits |
| v1.17.0 | 14 GiB | 2.067 | 3.003 | +0.94, fail | inside limits |
| v1.17.0 | 4 GiB | 2.067 | 3.478 | +1.41, fail (two repeats bit-identical) | inside limits |

Per position (the .rows files of each arm): (1) position 117 is a persistent defect: tail minus control is +5.34 on v1.16.0 and +5.55 on v1.17.0, so it is not from the bump and it is the residual tail error this item exists for; the runs are deterministic, so it can be traced to a token and the experts the 6900 XT served. (2) The gate flip between builds is a cluster of numerically unstable positions (about 6180 to 6230 and 6504) whose KLD moves by 2 to 5 between builds in BOTH arms (position 6504: control 0.33 to 2.61; 6182: control 4.26 to 2.52), so the p99.9 of 8,240 positions rests on about eight chaotic positions and flips with any small engine change; the bulk (mean, median, p99, top-1, perplexity) is equal across builds. Not a bump regression on this evidence. Open: trace position 117; propose a gate that does not rest on a handful of positions (more corpora, or positions stable in the control), which is the owner's call because it changes a threshold.

## Change Log

- 2026-11-03T02:29:34.417938+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261103_022951_measured-that-offloading-exper_7122
- 2026-11-03T02:29:58.843319+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T06:15:41.758211+00:00 (updated-by): Updated: section:notes
- 2026-11-04T06:57:23.943415+00:00 (updated-by): Updated: section:notes
- chg_20261104_071139_the-rx-6900-xt-expert-helper-n_8892
- 2026-11-04T07:11:43.593853+00:00 (updated-by): Updated: section:ledger-events
