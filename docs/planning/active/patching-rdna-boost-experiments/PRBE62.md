---
id: PRBE62
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:49.294113+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# VK-DRV-002 / RD79: Qualify AMD Vulkan graphics queue without adding a second scheduler

## 2026-10-08 audit decision

**Disposition: no new Vulkan queue implementation or patch yet.** First qualify the *already implemented* upstream, process-scoped `GGML_VK_ALLOW_GRAPHICS_QUEUE` switch on actual BigCherry drivers/hardware. The previous dual-queue, per-MoE-graph patch design below is superseded and is deliberately removed from this authoritative plan. If process-scoped routing is sufficient, terminate this item with a deployment/qualification note rather than modifying ggml.

This is an eligible dormant slice: last independent PRBE62 change was 2026-09-24 04:52 UTC; no PRBE62/RD79/graphics-queue BigCherry patch, plan, PR, queued hardware run or commit was found in the 12 hours preceding the audit. Recent MTP, QFP37/MMQ, MoE-cache, MET05/1328, patch-system and pin/release work is protected and is not modified here.

## Current implementation and causal boundary (verified at pinned llama.cpp b11474 / b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea)

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_get_device` (approximately lines 4278-4296) reads `GGML_VK_ALLOW_GRAPHICS_QUEUE` **at device initialization**, passes the resulting `graphics_flag` to `ggml_vk_find_queue_family_index` for both compute and transfer families. It is not a per-graph or per-kernel selector. A flag set does not itself prove a graphics family was selected; verify the returned family and queue flags.
- The same function (approximately 4477-4493, 4777-4790) creates device queues and one `device->compute_queue`; `ggml_vk_init` (approximately 5440-5462) binds the compute command pool to that queue. Graph execution obtains its context through `ggml_vk_get_compute_ctx`. No independently selectable MoE graphics compute lane is exposed.
- **Confound:** `prefers_transfer_queue` (approximately 4873-4894) includes `!allow_graphics_queue`; toggling the flag can also change asynchronous transfer-queue policy and queue-family choice. A speed difference cannot be attributed to graphics-queue compute execution without recording transfer-path state and family selection.
- `device->driver_id = driver_props.driverID` (approximately 4170-4180). `eAmdOpenSource` (AMDVLK), `eAmdProprietary` (proprietary driver), and `eMesaRadv` are **distinct** Vulkan driver IDs. The old proposed `eAmdProprietary`-only gate would exclude the AMDVLK Linux lane that supplied the motivating evidence. Do not silently equate them.
- The prior sample package ID `1263_prbe62_amdvlk_moe_graphics_queue` was **invalid**: order 1263 already belongs to `patches/1263_prbe41_ssm_conv_channels_major`. No PRBE62 package exists. Never reuse that ID.
- Existing queue/pool/transfer synchronization is shared by all graphs. A per-graph queue-family switch would require second device queue and pool, buffer ownership or concurrent sharing, semaphore/fence ordering across compute and transfer, command-buffer lifetime and multi-request safety. It is not a small environment-variable selector change.

## Evidence and source decisions

1. Upstream [llama.cpp #20599](https://github.com/ggml-org/llama.cpp/pull/20599), merged 2026-03-17 as `740a447f`, deliberately restored opt-in graphics-family selection after driver/desktop regressions; its diff also makes async-transfer preference conditional on the flag. **Adopt existing flag for A/B; do not fork queue-family discovery.**
2. External [R9700 Vulkan experiment log](https://github.com/JohnTDI-cpu/RDNA4-Llama-Experiments-R9700-Vulkan-Optimization-Log) (llama.cpp `dc8d14c58`, Ubuntu, R9700 PCIe 5.0 x16, `rm_kq=1`, three benchmark repetitions): AMDVLK MoE tg128 **156.3 -> 163.7 tok/s (+4.7%)** with graphics queue; AMDVLK dense tg128 **32.73 -> 30.07 (-8.1%)**. RADV MoE **149.5 -> 149.2 (-0.2%)**. **External measurements only**; not BigCherry, not b11474, and not transferable to gfx1100, R9700 PCIe x4 or multi-GPU/no-P2P. The log's AMDVLK reports *AMD open-source driver*, not `eAmdProprietary`.
3. [llama.cpp #25195](https://github.com/ggml-org/llama.cpp/issues/25195) reports a streamed MoE partial-offload transfer-queue write-after-write synchronization hazard. [#25196](https://github.com/ggml-org/llama.cpp/pull/25196) proposes an AMDVLK async-queue mitigation but is **closed/unmerged** as inspected 2026-10-08. Treat partial-offload/long-context async transfer as a **correctness gate**, not a free performance control. Do not force async transfer on AMDVLK merely to match A/B policy until sync validation passes.
4. [AMDVLK](https://github.com/GPUOpen-Drivers/AMDVLK) is discontinued; its installed driver version/ICD must be recorded and reproducible. ROCm vLLM/SGLang HIP streams and collectives are not Vulkan queue-family substitutes. No transferable dynamic Vulkan per-graph queue implementation was established by this audit.

## Bounded qualification (authoritative next action)

**Gate 0 — zero-code discriminator; terminate if unavailable.** On the current pinned BigCherry binary, enumerate `vulkaninfo` physical device, `driverID`, `driverVersion`, `deviceUUID`, queue-family `queueFlags/queueCount`, selected compute/transfer family, `single_queue`, `async_use_transfer_queue`, ICD path, PCIe width/speed, kernel/Mesa/AMDVLK versions, `rm_kq`, and actual model/quant. Check whether a separate compute-only and graphics+compute family exist and that the flag changes the *selected* family. If no qualifying AMDVLK/proprietary driver or no actual family switch exists, close as **no applicable lane**; do not create a patch. Never mix RADV and AMDVLK duplicate physical-device aliases in a tensor split.

**Gate 1 — process-scoped A/B, no source edits.** For the same driver/architecture and otherwise identical binary, run independent processes with `GGML_VK_ALLOW_GRAPHICS_QUEUE` unset versus set, explicit ICD/device selection. Include one VRAM-resident Qwen MoE decode (tg128/tg512) and one dense decode control, plus pp512/pp2048 and a representative long-context repeat; use identical `-b/-ub`, FA, cache, model, quant, `rm_kq`, clocks/power and no other tuning changes. On R9700 gfx1201 first, then gfx1100 dual-XTX device-by-device only if available; gfx1030 is not a primary lane. Record actual queue-family selection, async-transfer status, Vulkan timestamp/dispatch counts and E2E latency/TPS, VRAM, memory and transfer bytes. Use >=4 interleaved A/B pairs and CI95 intervals; warm both variants. Do **not** add a new benchmark definition when the existing BigCherry campaign harness can express these lanes.

**Gate 1a — transfer confound / safety.** If the flag also toggles async transfer, report the measured result as a *combined queue policy*, not isolated compute-queue gain. Only attempt a same-async control if both families support it and Vulkan synchronization validation proves it safe; never force a known unsafe streamed-expert upload path. Capture `VK_LAYER_KHRONOS_validation` synchronization findings where supported. Partial CPU-MoE/offload, >100K context and multiple requests must be correctness-qualified before speed claims.

**Correctness:** reference logits (or established KLD tolerance), greedy token identity, exact node/dispatch and expert-work accounting, repeated same-process requests, multi-ubatch, long-context, buffer ownership/sync-validation and allocation stability. MTP lanes require separate verification/acceptance and cannot be borrowed from active FMTP/QFP owners. A faster result with skipped work is FAIL.

**Gate 2 — disposition:**
- **No change:** absent family switch, unavailable driver, sync failure, MoE CI95-low <3% E2E, or insufficient evidence. Close PRBE62; retain upstream defaults.
- **Process-scoped opt-in only:** MoE CI95-low >=3% E2E and correctness pass, but dense or prefill regresses >1% or mixed workloads differ. Record a **model/driver-specific launch policy** using the existing upstream flag. Do not enable globally, and do not create an in-backend selector.
- **Potential code path (separate approval only):** only if a single long-lived mixed-model process cannot use process-scoped routing, the *measured mixed-workload* E2E opportunity is >=5%, and a trace proves the queue policy rather than transfer changes drives the gain. Before a patch, require a current-pin prototype of dual queues/pools and concurrent-sharing or explicit queue-family ownership barriers, timeline semaphore ordering, allocation lifetime and all correctness gates. Reuse `ggml_vk_get_device`, `ggml_vk_init`, `ggml_vk_get_compute_ctx` and existing submission paths; no second scheduler, generic placement table or extra runtime flag without a justified design review. Assign a **unique** patch ID only after approval.

For promotion of any future implementation: CI95-low >=3% production E2E and <=1% unaffected-control regression, exact route/transfer accounting, no correctness or sync-validation errors. Revert to current upstream queue policy on any failure.

## Ownership and dependencies

- PRBE62/RD79: only the Vulkan queue-policy **qualification and disposition**; no owned patch today.
- Upstream `ggml-vulkan.cpp`: sole queue-family selection, command-pool, buffer-sharing and submit implementation owner.
- PRBE61/RD78: separate `rm_kq` kernel geometry; hold `rm_kq` constant during PRBE62 comparisons, do not merge the selectors.
- PRBE55: Vulkan MMVQ/DMMV routing; independent kernel policy, hold fixed.
- MET05/1328 and MTP/QFP/F MTP: protected active work; no edits or overlapping experiments. If a later experiment needs those lanes, coordinate after their active window.
- BCOP53: thin action ledger; technical design stays here.

## 2026-10-11 b11552 recheck: environment admission and post-pin MoE prepass

**Scope/novelty:** b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`, 2026-10-10 19:25 UTC) is now the independent development branch's pinned llama.cpp baseline. The Vulkan queue policy is unchanged from the b11474 audit. A **new, merged upstream MoE mechanism** landed *after* that pin: [#29998](https://github.com/ggml-org/llama.cpp/pull/29998), commit `966baae76b578d5d3dd4c179a75a35c80ff75577` (2026-10-11 12:02 UTC). This changes Vulkan `MUL_MAT_ID` expert-row prepass, row-map allocation, tile dispatch and shader indexing. It is **not in b11552** and cannot be credited to, or mixed into, a b11552 graphics-queue A/B. No PRBE62 implementation or independent PRBE62 hardware lane was found in the prior 12 hours; BCOP53 remains the earlier disposition ledger. This recheck does not take ownership of PA49 pin work or active QFP/Meta/MMQ/collective lanes.

### Source-proven A/B admission pitfall

In b11552 `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_get_device`:

```cpp
const bool allow_graphics_queue = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != nullptr);
device->async_use_transfer_queue = prefers_transfer_queue ||
    (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != nullptr);
```

Both are **presence** switches, not parsed booleans: setting either variable to `0`, `false` or the empty string **enables** its path. `GGML_VK_ALLOW_GRAPHICS_QUEUE` affects *both* compute and transfer family search and removes the AMD dGPU default `prefers_transfer_queue`; a single-queue device still forces async transfer off. An A/B arm with `GGML_VK_ALLOW_GRAPHICS_QUEUE=0` is **not a control**. The old plan's unset/set wording is correct, but future scripts must fail closed on this easy-to-miss confound.

**Implementation-ready Gate 0 (no patch):** launch **fresh independent processes**, control with both environment variables **absent** (not assigned empty/zero), subject with only `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`; record an exact environment presence/values receipt, driver ID/ICD, physical-device UUID, compute/transfer queue-family indices and flags, `single_queue`, `async_use_transfer_queue`, graph/cmd-pool binding, actual `MUL_MAT_ID` route and `rm_kq`. Reject a result if the compute family did not change, the driver/device identity differs, or the transfer policy changed without being explicitly classified as a *combined policy* result. Do not add a new queue selector or force the async override to equalize arms without synchronization proof.

**Post-pin upstream mechanism and independent correctness fence:** #29998 replaces optional expert-ID hoisting with a compact row-tile map and a one-workgroup expert prepass. Its host upper bound is `max_tiles=min(n_rows, ceil(n_rows/BN)+min(n_experts,n_rows)-1)`; shaders consume `(expert,valid_rows)` tile descriptors and packed 16-bit token/slot IDs. The new device-support gate rejects token/slot dimensions above 65536 for this route. This is a distinct dispatch/work-accounting change, **not** a queue-family optimization. Never compare b11552 control against post-#29998 subject. At a later pin, first prove exact expert multiplicities (including repeated IDs), row-tile coverage, full logits and prefill/decode/MTP parity with the new prepass, and rebaseline both graphics-queue arms on the **same** commit/driver before interpreting performance. If the prepass changes workgroup counts or bytes, report them independently; faster due to missing expert work is FAIL.

**Cheapest discriminator actually run:** nine b11552 source-static queue/transfer assertions (one initial line-range omission corrected) and a deterministic host upper-bound fixture: 42,810 exhaustive small expert-count partitions plus 10,000 seeded random partitions; all satisfied `actual_tiles <= max_tiles`. A separate presence-value fixture confirmed that absent is false and `"", "0", "1", "false"` are all true. These validate only source policy and a mathematical bound; they do **not** verify Vulkan shader correctness, queue selection on an installed driver, performance, GPU execution or actual transfer traffic.

**Promotion/rejection remains BCOP53's gate:** first prove an actual AMDVLK/proprietary graphics-family switch and safe transfers on the deployed R9700 gfx1201, then isolate the single-device Qwen MoE and dense controls; do not infer gains from the dual-XTX HIP or gfx1030 lanes. Keep context, quant, MTP, `rm_kq`, device and pin fixed. Require same-process repeated requests, multi-ubatch, long context, logits/KLD and exact expert-work accounting. No available driver/family switch or MoE CI95-low <3% E2E => terminate with upstream default; positive MoE but >1% dense/prefill regression => only model/driver-scoped **process opt-in**, not a backend scheduler. External AMDVLK +4.7% MoE / -8.1% dense remains external evidence, not a BigCherry measurement.

## Historical provenance

Created 2026-09-09 by capability-rebaseline-v3; re-reviewed 2026-09-24 against b11126. The original design proposed a new `BIGCHERRY_VK_MOE_GRAPHICS_QUEUE` flag, per-graph dual queues, a new buffer-sharing policy and a hypothetical 1263 patch. Those sketches are superseded by the 2026-10-08 code/evidence audit; preserved in Git history rather than maintained as contradictory active steps. No hardware/build/benchmark ran during this audit; static pinned-source assertions verified the queue/transfer coupling.

## Change Log

- 2026-10-08 (triage): Preserved concurrent main audit of upstream GGML_VK_ALLOW_GRAPHICS_QUEUE, transfer confound and driver-scoped qualification; no new Vulkan queue patch or device-lane evidence.
