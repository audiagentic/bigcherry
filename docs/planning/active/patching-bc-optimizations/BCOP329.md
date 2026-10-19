---
id: BCOP329
order: 329
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# PRVP06: Vulkan wide-FWHT shader admission (1024–8192)

**Discovery:** Pinned llama.cpp b11552 Vulkan FWHT supports F32 widths 64–512 only; wide Hadamard-hinted MUL_MAT uses dense fallback. Open upstream [#29772](https://github.com/ggml-org/llama.cpp/pull/29772) adds one-row/256-thread LDS FWHT for 1024–8192 and F16 variants stacked on [#29101](https://github.com/ggml-org/llama.cpp/pull/29101). N8192 requires 32 KiB LDS and 32 F32 registers/lane before temporaries. External gfx115x Bonsai 2 +11.5–12.5% E2E is **not** a BigCherry gfx1100/gfx1201 result; no actual BigCherry wide route is established.

**Authoritative owner:** New PRVP06 (only Vulkan FWHT owner). RNX07 WHT collective, PRVP05 RMSNorm, RNX08 norm/RoPE, RNX02 attention and PRBE21 CONCAT are distinct. RRVP02's user-directed Vulkan implementation pause controls admission. No duplicate dispatcher, allocator, telemetry, kernel package or queue.

**12-hour and novelty:** Independently active feat/rad13-fp8-worker at Audumla 8aa65ee (2026-10-19 15:08:56 UTC), newer than main. No repository-visible independent wide-FWHT plan, patch, commit, PR, benchmark or recorded queue within preceding 12 hours; no recent BCOP owns this shader. Excluded active RAD13/PR #330, RCS09/RCS05, RAD11/12/10, WSL/release; off-repository queues unverified. Recheck on any implementation run.

**Completed cheap discriminator:** Host-only 208 eligible geometry, 80 fallback, 1456 row-tail, 66 exact integer FWHT comparisons, seven negative gates; Python compilation passed. No GLSL/HIP build, repository tests, GPU execution or BigCherry benchmark.

**Unresolved action:** HOLD while RRVP02 paused. Then prove an actual 1024–8192 Vulkan Hadamard node, exact pipeline identity and >=2.9126% removable exposed E2E wall; check upstream absorption and #29101 composition. Only if admitted, run source/GLSL, correctness, ISA/LDS/VGPR/spill and paired gfx1100/gfx1201 qualification described in PRVP06. **Terminal:** BLOCKED_VULKAN_PAUSE, PIN_ABSORBED, NO_WIDE_FWHT_ROUTE, BELOW_CEILING, UNSUPPORTED_ROUTE, NUMERIC_FAIL, REGRESSION, NO_GAIN, PROMOTED. No work on protected capability.
