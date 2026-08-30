# 1208_rd21_gfx1151_mmvq_nwarps_table: gfx1151 (RDNA3_5) MMVQ table with nwarps=2 for Q8_0 decode (RD21)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD21

## What it does

Adds a dedicated mmvq nwarps table entry for gfx1151 (previously grouped with RDNA2), giving nwarps=2 for Q8_0 decode matmuls; the batch-range condition is baked in at the fork's branch-tip fixed state so speculative-verify batches match decode consistently.

## Why

gfx1151 previously fell into the RDNA2 mmvq table; the fork's sweep found nwarps=2 gives ~+0.6% decode on Qwen3.6-35B-A3B Q8_0. The performance claim is hardware-deferred since this project's hardware (gfx1100/gfx1201/gfx1030) cannot exercise the RDNA3_5 branch.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 1818c3b37, with the branch-tip fix from follow-up commit 8cdf1ab08 baked in (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.
