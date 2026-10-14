---
id: RAD02
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-13T04:42:45.717482+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Inverted expert-parallel layout: trunk on the R9700, routed experts on the two XTX

## Description

Target architecture agreed in conversation 2026-10-12: the R9700 runs the whole trunk (attention, delta-net, shared expert, heads, KV; ~7 GiB + KV) on native RDNA4 kernels, and each XTX holds whole routed experts (half the set each, ~75-80% resident vs ~67% today). Removes the cross-card trunk sum that bounds 27B and Flash-Next prefill. Exchange per layer through pinned host memory with GPU-side flags (no CPU relay; measured ~60 us exchanges, ~3 ms/decode step); prefill pipelined in micro-batches so transfers overlap compute. Stepping stone: the R9700 expert helper (slice 2b) proves the per-layer exchange cost. Implemented as a new engine patch (expert-parallel mode) plus persistent expert kernels on each XTX.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-10-14 progress (branch feat/rad03-gpu-lanes): piece 1 done -- r3_moe_helper runs on gfx1100 with libr3 (plugin chosen by device arch; R3_MOE_HELPER_PLUGIN override; helper built for gfx1100+gfx1201; gfx11 store drain s_waitcnt_vscnt). --compare on an XTX (R3_MOE_HELPER_MIB=20480, layer 20, 64 rows): rel_l2 6.9e-4 / 7.6e-4, prefix_ok, same as on the R9700. Quality bar for the inverted layout is the XTX re-chunking noise floor (KLD ~0.06 on the 4-doc corpus), not bit-exactness (see RAD03 notes). Remaining: (2) engine-side hook + mailbox ops in the R9700 engine -- they live in libr3 (gfx1100-only); plan a small separate plugin with just the mailbox rows and hook, built for gfx1100+gfx1201, loaded beside libr4d; (3) one engine rank to two helpers: one shm segment per helper with disjoint expert ownership, publish routes by owner, add merges both; (4) placement: R9700 keeps attention/dense/hot experts, XTX hold the rest; (5) Flash-Next end-to-end vs current best (prefill 1249 / decode 48.3 tok/s with the R9700 helper). 27B is dense: run it on the R9700 alone (3.1-3.2K prefill) rather than in this layout.

2026-10-14 late: piece 2a libr3moe plugin (mailbox rows + hook, gfx1100+gfx1201; runs use --kernels libr3,libr3moe,libref; regression byte-identical, decode 48.6). Piece 3a protocol v8: helper returns ONE weighted row a token (slot order, f32 acc, one bf16 narrow); engine merge/shadow use linear slot maps (route mark + inverse) instead of scanning the step; cap lifted 409 -> 4096 rows. --compare per token: XTX rel_l2 8.4e-4/5.1e-4, R9700 8.3e-4/4.2e-4. KL (4-doc) KLD 0.0665 / top-1 88.46% / PPL 8.987 -- at the noise-floor edge (0.058-0.062); extra bf16 narrow of the helper row is the likely +0.005; option: f32 result rows. Speed (KL-corpus prompt 8440 tok, not comparable to earlier runs): max rows 64 -> prefill 1026/1018, decode 57.4/58.4; max rows 4096 -> prefill 1060/1063 (+3%), decode 23.9 because the pre-queued GPU lane computes every request at the cap. Piece 3b protocol v9: small lane (pre-queued, R3_MOE_HELPER_SMALL_ROWS) + large lane per rank launched sized on arrival; under test (short29). Remaining: publish kernel copies activations with one block and scans routes with one thread (fix before 4K prefill timing); one engine rank -> two helpers (segment per helper); placement + R9700 engine runner; end-to-end Flash-Next.

2026-10-15 01:20 first R9700-engine runs (branch feat/rad03-gpu-lanes 42388e88; fn-r9700-engine.sh: engine on R9700 tp1 --kernels libr4d,libr3moe,libref; helper on one XTX, R3_MOE_HELPER_ENGINE_RANKS=1 full-width, MIB 20480). Memory: at tp1 the engine must hold every expert -- 47.7 GiB n-gram pinned in RAM + host pool does not fit 86 GiB RAM (44 GiB pool: n-gram refused; 30 GiB pool: 5.6 GiB experts did not fit; + --weights-disk-tier: 6.7 GiB relayouted experts cannot use the file tier). Ran with --ngram-placement disk + 70 GiB host pool (proof only; owner prefers n-gram in RAM). Results 7306-token prompt: R9700 alone prefill 526 / decode 8.2 tok/s; + one XTX helper (max rows 64) prefill 549 / decode 16.8 -- end-to-end works, helper doubles decode; both far below the two-XTX+R9700-helper layout (~1480 / ~68). Bottleneck: experts streamed from the host pool. Next: (a) engine skips placing helper-owned experts (planner 'remote' tier for (layer, expert) units the helper owns, via a hook query like the layer-map hook; valid only when max rows covers every step and no shadow/mask diagnostic; MTP + protected layers stay local) -- frees ~20 GiB per XTX helper so the n-gram returns to RAM and KV gets VRAM; (b) two XTX helpers (segment per helper); (c) load-balanced ownership by routing heat x card cost so engine and helpers finish a layer together; (d) measure the merge wait with --profile-ops. Quality of this layout not yet KL-checked.

2026-10-15 02:15 piece 4b working (e07de541): R3_MOE_HELPER_SKIP_OWNED=1 leaves helper-owned experts unplaced on the R9700 (planner pins them to the mapped tier, expert fill skips preplaced units, pre-plan footprint excludes them, bring-up publishes a never-read placeholder). R9700 engine + one XTX helper, max rows 4096, n-gram back in RAM, host pool 30 GiB: prefill 992.7 / 1004.3, decode 21.3 / 21.2 tok/s, output identical to the non-skip helper run (6a2c53d5) and across runs. vs R9700 alone 526 / 8.2. Still below the two-XTX + R9700-helper layout (~1480 / ~68). Remaining bottleneck: ~21 GiB of the R9700's experts still in the host pool. Next: second XTX helper (one shm segment per helper, disjoint ownership, publish routes by owner, merge both) so every expert is VRAM-resident on some card; then load-balanced ownership and the merge-wait profile.

2026-10-15 03:05 piece 5 working (2e64d47e; v10 protocol, two helpers share one segment, wake-all futex fix). R9700 engine (tp1, libr4d,libr3moe,libref) + XTX0 and XTX1 helpers (R3_MOE_HELPER_COUNT=2, 20 GiB each, ENGINE_RANKS=1), SKIP_OWNED=1, MAX_ROWS=4096, n-gram RAM, host pool 30 GiB, 7306-token prompt: prefill 1881.5 / 1880.4, decode 46.5 / 46.5 tok/s, identical output (6242c9b1). Helpers balanced: ~4450 requests each, ~870 us a request, 176 large-lane requests each. vs current best layout (two-XTX engine + R9700 helper) ~1480 / ~68: prefill +27%, decode -32% (different generated text, MTP acceptance not compared). Quality (KL) of this layout not yet measured. Next: helper per-request cost at decode (870 us is mostly fixed overhead), load-balanced ownership by routing heat x card cost, merge-wait profile, KL check.

2026-10-15 03:50 quality: R9700 engine + two XTX helpers (SKIP_OWNED, small lane 64, KL steps 64 tokens) vs the XTX-pair exact reference (4-doc corpus): KLD 0.0592 / top-1 88.48% / PPL 8.987 -- inside the XTX re-chunking noise floor (0.058-0.062). Speed (7306-token prompt): small lane 64 -> prefill 1878/1881, decode 46.5/46.6; small lane 8 -> prefill 1783/1808, decode 54.6/54.0 (identical output across all four). Helper decode request ~690 us at small 8 (fixed cost; full-width runs the GEMM chain twice). Two-stream halves hung the lane (shared libr3 MoE GEMM not safe concurrently on one card) -- reverted (563455f8). Open bug: KL run with small lane 8 + MAX_ROWS 64 (every step on the large lane) hangs at the first merge; speed runs with MAX_ROWS 4096 do not. Current best decode remains the two-XTX + R9700-helper layout (~68).

2026-10-15 04:45 full-width helper (9713602e): one stored full expert, one GEMM chain at 640 width. R9700 engine + two XTX helpers, small lane 8: prefill 1893 / 1943 (best), decode 52.8 / 53.1, identical output 305c5aff; helper decode request 484/466 us (was ~690). KL 0.0672 / top-1 88.53% / PPL 9.0885 (ref 9.1034) -- slightly above the 0.058-0.062 floor, PPL closer to reference; full-width rounding differs from the TP-split reference, re-check before default. Hardware-queue fix 486147bb (GPU_MAX_HW_QUEUES=8 in helper) removed the KL large-lane hang. --profile-ops on the R9700 engine (whole run): moe_mailbox_add (wait + merge) 55.5% (~1.5 ms avg), gemm_nt_q 11.9%, moe_gemm_q 7.3%, hc_read 6.6%, moe_mailbox_publish 6.3% (~170 us). Decode is bound by the per-layer helper round trip, not R9700 compute. Next: time each kernel in the helper chain and fuse the small ones; trim publish for decode-sized steps; ownership by heat (R9700 keeps hot experts, helpers cold) so decode routes mostly local.

2026-10-15 05:10 route-heat ownership (052a876d): R3_MOE_HELPER_HISTOGRAM profile run (KL corpus prompt, 8440 tok) -> heat.rank0; helpers with R3_MOE_HELPER_HEAT own the coldest experts first so the R9700 keeps the hot ones. Balanced A/B on the 7306-token prompt (R9700 engine + two XTX full-width helpers, small lane 8, SKIP_OWNED): heat prefill 2635.6 / 2638.6, decode 59.8 / 59.8 (fbb7cfab, helper 405 us); base prefill 1942.8 / 1944.3, decode 53.5 / 52.1 (305c5aff, helper 445-468 us). Prefill +36%, complete separation. KL with heat ownership: KLD 0.0665 / top-1 88.54% / PPL 9.118 (ref 9.1034, x1.0016) -- same band as every helper configuration. Best Flash-Next prefill on Brutus now 2,637 tok/s (from 1,180 two-XTX baseline, 1,480 two-XTX+R9700-helper); decode 59.8 vs the two-XTX+R9700-helper layout's ~68 (different text; acceptance not compared). Two R9700 reference ~6,500 prefill.

2026-10-15 05:35 stage-once (50ba9264: two helpers share slot 0 activation staging): heat ownership, helpers 20 GiB -> prefill 2768.7 / 2767.2, decode 58.2 / 60.0 (fbb7cfab). Helper budget A/B (balanced 22/20/20/22 GiB): 20 GiB prefill 2765.6 / 2767.2, decode 60.0 / 60.0 (fbb7cfab); 22 GiB prefill 2529.4 / 2458.1, decode 69.6 / 69.7 (bef41b47). More helper experts = fewer R9700 host-pool reads (decode up) but more prefill hand-off to the slower helpers (prefill down). At 22 GiB the layout beats every earlier layout on both axes: prefill 2.1x the two-XTX baseline (1,180) and above the two-XTX+R9700-helper layout (1,480 / ~68). Profile with heat: moe_mailbox_add 33% (745 us avg), local moe_gemm_q 21%, gemm_nt_q 15%, hc_read 8%, publish 7.5%. Still ~40% of two-R9700 prefill (~6,500).

2026-10-15 05:45 helper per-stage timing (203e2f8b/ff53c12d), small-lane decode requests, heat ownership, 22 GiB helpers: routes ~50 us, gate/up GEMM 242-259 us, gated quant 12-41 us, down GEMM 102-108 us, reduce 23 us (total ~455). The gate/up GEMM is ~4x its weight-bandwidth bound at decode size: libr4d/libr3 has a single w4 MoE row (moe_gemm_w4a8, WMMA 16/32-row tiles, NOTUNE) and no decode-sized variant; the R9700's own hot experts use the same kernel (moe_gemm_q 21% of its op time). Next lever: a decode MoE GEMV for w4nl64a8h (rows <= ~8 a expert) on gfx1100 and gfx1201, registered as a moe_gemm_q row banded to small M, used by the helper's small lane and by the engine.

## Change Log

- 2026-10-13T04:42:45.717482+00:00 (created-by): Created by agent
- 2026-10-14T10:52:41.321989+00:00 (updated-by): Updated: section:notes
- 2026-10-14T12:32:18.859149+00:00 (updated-by): Updated: section:notes
- 2026-10-14T14:17:31.118315+00:00 (updated-by): Updated: section:notes
- 2026-10-14T15:14:52.650873+00:00 (updated-by): Updated: section:notes
- 2026-10-14T16:03:29.249768+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261014_160331_flash-next-can-now-run-with-th_1934
- 2026-10-14T16:03:35.151883+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-14T16:49:59.962733+00:00 (updated-by): Updated: section:notes
- 2026-10-14T17:40:20.077523+00:00 (updated-by): Updated: section:notes
- 2026-10-14T18:03:04.872416+00:00 (updated-by): Updated: section:notes
- chg_20261014_180305_flash-next-prefill-on-brutus-r_5446
- 2026-10-14T18:03:08.961352+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-14T18:34:39.996377+00:00 (updated-by): Updated: section:notes
- 2026-10-14T18:51:25.390902+00:00 (updated-by): Updated: section:notes
