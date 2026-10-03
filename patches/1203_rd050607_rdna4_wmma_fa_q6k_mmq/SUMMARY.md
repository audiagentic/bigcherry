# 1203_rd050607_rdna4_wmma_fa_q6k_mmq: RDNA4 WMMA flash-attn and Q6_K mmq prefill performance work (RD05/RD06/RD07)

**Status:** rejected
**Plan item:** RD05/RD06/RD07

## What it does

Fixes a head-256 WMMA flash-attn combine race and a tile_Q reuse race, tunes and enables the WMMA flash-attn path up to head 576 on RDNA4 by default, and hoists/folds Q6_K mmq sub-scales into the row base-scale to remove an int-mul from the scale chain (fork: Q6_K mmq 40 -> 58 TFLOPS). Also adds op-timing instrumentation and test cases.

## Why

Targets specific RDNA4 kernel-level performance bugs and tuning gaps identified upstream in the fork's own commit series; the head-256 combine race and tile_Q reuse race are correctness fixes bundled with the performance tuning.

## Upstream / provenance

Ported (with adaptations for this project's own 1000/HI70 anchors) from stew675-rdna-boosts fork commit 1d525bd45 (https://github.com/stew675/llama.cpp), covering RD05/RD06/RD07. Not merged into ggml-org/llama.cpp master.

## DEMOTION (2026-09-23, PA39)

The sections above describe the original claim and are preserved unchanged.
This patch is `rejected` as a bundle: its declared real-hardware acceptance
(PA39 retry-1, Brutus, revision `a4d373fa`, pin `28ff0958`/b10901,
gfx1100 devices 0/1, gfx1201 device 2, gfx1030 device 3) completed and
failed one of its three contracts.

- RD05-WMMA-FA-CORRECTNESS-BARRIERS: backend-reference correctness and
  controls PASS.
- RD06-RDNA4-WMMA-FA-CONFIG: activation and backend-reference PASS;
  performance FAIL -- `target_kernel_gain_pct` point `-0.0154%`, CI95 low
  `-0.0745%`, required CI95 low `>= 0.5%`; `rd06-controls` fails through the
  same gate. The head-576 WMMA enablement gives no measurable gain.
- RD07-Q6K-MMQ-PREFILL-FOLD: activation, backend-reference, performance and
  controls PASS on gfx1100, gfx1201 and gfx1030.

Evidence: `producer-execution.json` SHA-256
`a6949b475ba84d453942cff1e929d6001d3b45f625c7db2a1bb3e0c61485ef5a`
(`eligible=false`); full record in
`docs/planning/active/patching-patch-system/PA39.md` and PA40's DoD matrix.

Decision (GPT lifecycle review `req_f34f50a25c6240fe`): reject this exact
composition unchanged rather than removing RD06 from it -- the failed
identity and its evidence stay together, and no threshold changes. PA41's
RD06 remediation deferral stands. The passing slices are not promoted by
reuse of this receipt: RD07 (and RD05, after its scope is separated from the
RD06-only config/softcap/dkq-gate/wmma-gating edits) may only return as new,
separately identified `untested` patches with fresh hardware evidence. Kept
in `[experiment.rd05-07-only]` so the failed receipt stays reproducible.
