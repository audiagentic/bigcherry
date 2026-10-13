---
id: BCOP183
order: 183
plan: patching-bc-optimizations
state: done
created-at: '2026-10-13T10:06:16Z'
created-by: agent
priority: P2
work: S
---

# PGC05: retire duplicate adaptive AllReduce threshold/bucket plan

PGC05's scalar `--allreduce-switch-bytes` is already implemented in 0860/0840 (96 KiB, context snapshot). No comparable evidence establishes a nonmonotone size crossover needing a second table. PGC09 owns runtime dispatch; PGC13 owns conditional offline scalar calibration; PGC12 owns actual provider completion/transition traces; PGC11 owns wire safety. Historical validated exact-F32 dual-gfx1100 tg128 +3.87%..+3.98%, pp512 +0.02%..+0.04%, bit-identical. Source checks 15/15; synthetic host selector 11/11. No GPU/build run or new performance claim. Close PGC05 and reopen only on matched execution receipts proving disjoint provider reversals and >=3% end-to-end ceiling. Recent Radiance work excluded and untouched.
