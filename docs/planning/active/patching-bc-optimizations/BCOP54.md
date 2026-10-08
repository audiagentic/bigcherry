---
id: BCOP54
order: 54
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-08T10:05:18+11:00'
created-by: agent
priority: P2
---

# PRBE65 / RD82: reject generic Vulkan FA occupancy scaling

## Discovery / change

Pinned b11474 and inspected upstream master retain the 64 KiB-only scalar FA occupancy limiter. It adds actual dummy LDS writes/barrier, affects RDNA prefill (n_rows >= 64, hsk <= 128), not normal RDNA MTP verify/decode. External 32 KiB proprietary R9700 measurements show scaled limiter pp512 704 vs stock 719 and disabled 744; no BigCherry causal hardware result. The previous 16 KiB floor sketch was internally contradictory.

## Authoritative owners and existing work

PRBE65 owns RD82's terminal research disposition; upstream Vulkan scalar FA owns code/shader policy. Existing QFP07/PRBE62/PRBE68 own attention placement/queue/submission independently. No subsequent BigCherry implementation or patch is known to act on RD82; PRBE65's last independent plan commit was 2026-09-24. No duplicate tuning table, runtime flag, shader path or new plan is warranted.

## Terminal disposition / reopening

**Close generic 32 KiB scaling without patch.** Reopen only after actual fleet 32 KiB + scalar prefill activation and >=5% E2E FA wall-share, then a disposable SPIR-V shared-memory/correctness discriminator and paired exact-driver qualification (CI95-low >=3% E2E, <=1% controls). See PRBE65 for exact gates. Do not queue a hardware campaign without Gate 0.

## Dependencies / validation

Sources: https://github.com/ggml-org/llama.cpp/discussions/21043 and https://github.com/ggml-org/llama.cpp/issues/26163 . Seven pinned-source assertions and five host arithmetic fixtures passed; no build, shader compile, test-backend-ops or hardware benchmark ran. This is an external-evidence-based rejection, not a BigCherry performance claim.
