---
id: BCOP328
order: 328
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# PRBE21: Vulkan GDN transposed-CONCAT shader admission

Discovery: pinned b11552 uses flattened concat.comp for GDN conv_input; transposed-B loads are strided. Upstream llama.cpp #30591 adds a 32x32 shared-memory transpose tile (256 threads, 1 barrier) and reports R9700 Qwen3.6-35B pp2048 4940.6 to 5723.6 t/s (+15.8%), external not BigCherry. Competing #30149 reuses two copy/transpose launches.

Owner: PRBE21 conv_input, not its separate HIP fusion or retired MoE rpb2. Preserve SSM_CONV and recurrent conv_state_update CPY. RRVP02 Vulkan implementation pause applies. Do not create duplicate shader/dispatch/allocator/telemetry.

Eligibility: independently active feat/rad13-fp8-worker at Audumla 45c6edc01e14 (2026-10-19 14:08 UTC), newer than main. No independent BigCherry matching work in 12 hours; PRBE21 last changed Oct 11. Protected RAD13/PR330, RCS09/RCS05, RAD11/12/10, WSL and release work untouched; off-repository queues unverified. Novel Vulkan CONCAT mechanism, not recent Q4 KV transpose.

Host oracle: 1140 contiguous and 120 strided byte-exact cases, 8137 modeled tile barriers, six selector cases, Python compilation passed. No GPU build or benchmark.

Next: hold until Vulkan resumed. Check upstream absorption. Prove eligible gfx1100/gfx1201 single-sequence route, complete output/CPY work, >=2.9126% removable exposed E2E ceiling. Compare baseline, #30591, #30149; test direct CONCAT, strides/tails/limit, SSM state, logits/KLD/MTP, wave/LDS/VGPR/spills and physical traffic. Require CI95-low >=3% E2E and <=1% control regression. Terminal: PAUSED, PIN_ABSORBED, NO_VK_CONCAT_ROUTE, BELOW_CEILING, NUMERIC_FAIL, REGRESSION, NO_GAIN.
