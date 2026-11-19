# LTX-2.5 IQ8R native TP3 + optional gfx1030 text worker

**Status:** engineering design and detailed implementation backlog; **not an authoritative Planning MCP item**.
**Source baseline:** BigCherry origin/main 358ddd197, frozen upstream candidate 3deabdaeca17b58bad90d332f4b204cf7bb8a10e.
**Model:** https://huggingface.co/StillDeadcode/radiance-ltx-2.5-iq8r (license acceptance required).
**Non-negotiable:** the installed inference engine and any conditioning worker are native C++/HIP. Python remains optional build/test tooling, never an engine runtime dependency.

## Target configuration

- TP ranks 0,1: RX 7900 XTX gfx1100, libr3.
- TP rank 2: Radeon PRO R9700 gfx1201, libr4d.
- Optional fourth, **separate** worker: RX 6900 XT gfx1030, libr2 + new native Gemma4 kernels.
- TP1 remains production default/recovery until TP3 device parity/performance acceptance.
- The existing Flash-Next four-card remote-MoE layout is a separate, mutually exclusive deployment. No GPU may be owned by both sessions.
- This work must use the RUM01–RUM10 frozen native-YAML/ABI migration as its target, not rebase new functionality onto obsolete v1.17.0 ABI.

**2026-11-20 model availability:** the local IQ8R container is present and was checked with a bounded format-v2 header read (ltx2, 6,114 directory entries, 8,726 planes); see docs/design/ltx25-iq8r-model-preflight.md. This unlocks real-container CPU inspection but does not qualify any GPU path.

**Actual model plane proof (2026-11-20):** a permitted, read-only development test verified shape/size/256-byte alignment of codes, U8 group scales and BF16 row scales for all 1,306 IQ8R entries in the local RAD2 container. See docs/design/ltx25-iq8r-model-plane-evidence.md. This does not establish decoded numerical or GPU execution correctness.

## Source-grounded findings

- Upstream arch/ltx2/ltx2.cpp:3153–3166 explicitly refuses world_size > 1. At 3183–3199 it constructs the text geometry with world=1.
- At 1020–1115, Attn.qkv is [all Q, all K, all V]; RMS of Q and K spans **every head**, before RoPE. At 1490–1536 and 3590–3710, the block runs video/audio self-attention, text cross-attention and both A2V/V2A cross-modal directions. Block 28 self-attention is bypassed under STG.
- At 3199–3234, model metadata defines 32 video heads of 128 dims, 32 audio heads of 64 dims, 48 blocks, video 4096 and audio 2048 widths. Runtime validation must use metadata, not assume all future models share these shapes.
- At 873, Gemma stack is bf16 [T, C*(L+1)] for every hidden state. With at most two 1024-token prompts, C=3840, L=48, a full stack transfer is 770,703,360 bytes (~735 MiB). If connectors also ran on the text card, their final 4096+2048-wide two-prompt contexts would be only 25,165,824 bytes (24 MiB), **but** the eight video and eight audio connector blocks and their weights may not fit the 6900's remaining memory. The initial option should keep connectors on a fast card and move only Gemma; the fast card receives its stack.
- The model card declares int8 signed -127..127 with u8 scale per 256 K, bf16 second scale per row and an unnormalised Hadamard transform in 128-K blocks; this is not a generic unrotated W8A8 weight.
- engines/radiance/components/libr2/units.txt currently builds only r4d_moe, r4d_fused_quant_fp8 and r4d_quant_act_fp8. It does **not** expose the dense GEMMs, causal/sliding attention, norms and other operations needed by Gemma4. engines/radiance/archs.toml allows gfx1030 as helper only.
- BigCherry already has real mixed-arch TP3: rad0016 (per-rank kernel selection), rad0017 (shared collective rendezvous), rad0018/rad0019 (uneven ownership), rad0021/rad0022/rad0024/rad0032/rad0033 (exact host-staged collective optimizations), with manifest settings tested in PRs #863, #935 and #959. LTX needs its own graph and quantized weight ownership.
- This host has no observed GPU P2P for the XTX/R9700 TP3 group, and the R9700 uses a narrower PCIe path. Its reduction cost must be measured at video token counts; LLM decode benchmarks cannot establish LTX speedup.

## Dependency order

RUM frozen source + ABI/plugin SDK and native YAML
-> LT3-01 to LT3-05 sharding/normalization correctness
-> LT3-06 to LT3-09 full DiT + guidance + leader-only decode
-> TP3 device numerical and latency gates
-> LT3-10 to LT3-13 optional 6900 text worker
-> four-card numerical/latency gates
-> LT3-14/15 release.

TP3 must be independently deliverable **without** the 6900 text worker.

## Gap register: concrete code work and evidence

### LT3-01 — Upstream model, version and build identity (P0)

**Gap:** current engines/radiance/recipes.toml production pin remains v1.17.0; published LTX-2.5 IQ8R needs newer Radiance. Active RUM01–RUM10 ports may change ABI/native YAML.

**Implementation:** wait for RUM owner to freeze a qualifying candidate, model hash and SDK; add LTX patches to a separate opt-in managed patch-set once upstream source applicability is proven. Verify full plugin build identity; disallow stale libr2/libr3/libr4d or mixed ABI fingerprints. Preserve production recipes and old configs.

**No-card:** patch applicability, idempotence, forbidden terms, build graph and plugin ABI validation. **Hardware:** complete mixed-card load and native generation.

### LT3-02 — Rank, device and pass ownership (P0)

**Gap:** arch/ltx2/ltx2.cpp:3161 TP1 refusal; text and DiT currently declared in one world. Single-process VCOND/VSTEP/VDECODE semantics are not rank-safe.

**Implementation:** extend the LTX-owned arch builder, opt-in, to accept exactly three named engine BDF ranks; per-rank receipt and kernel availability from rad0016/17. Declare all ranks for DiT, a designated leader for VAE/output, and explicit producer/consumer boundaries for conditioning. Unsupported combinations continue to fail before allocation. Do not delete the TP1 guard until a functional path is ready.

**No-card:** duplicate/missing/rank-order/device-role rejection. **Hardware:** mixed-rank startup/shutdown, clean failure without hanging other ranks.

### LT3-03 — Uneven attention-head ownership (P0)

**Gap:** 32 video and 32 audio heads cannot be evenly partitioned by three.

**Implementation:** evaluate 8/8/16 and 12/12/8 as starting splits, plus measured alternatives. Derive whole-head counts and offsets per modality from metadata (video head_dim 128; audio head_dim 64). Use one consistent slice table across self/text/cross-modal variants, with special handling for cross-modal attention using the audio head width. No silent dropped/duplicated heads.

**No-card:** coverage/uniqueness/offset tests; zero or nonwhole-head cuts refused. **Hardware:** actual shape probes and mixed arch kernel support.

### LT3-04 — IQ8R-aware linear/QKV sharding (P0)

**Gap:** Attn.qkv is fused by Q bank, K bank, V bank; splitting its contiguous full output into 3 parts is wrong. Row-parallel output and FFN-down must reduce rank partials.

**Implementation:** for a rank's local inner offset and width, map source rows at offset, D+offset, 2D+offset into local [Q,K,V] banks. Text K/V uses two banks; gates and gains require their own head slices. Use RAD_MAP_ROWS and native model builder ownership; row-parallel output/down uses K spans and one explicit all-reduce after local GEMM, with biases applied **once**.

**No-card:** fused bank source reconstruction, output index checks and exactly-once bias fixture. **Hardware:** decoded actual container rows/scales and real plugin dispatch for each linear type.

### LT3-05 — All-head Q/K normalization (P0)

**Gap:** LTX uses RMS over all Q or all K channels. Rank-local RMS changes Q/K and every attention result.

**Implementation:** for each row, compute partial f32 sum of squares per rank, use a deterministic all-reduce over three partials, divide by global H*hd, then scale each shard with one shared denominator before its local RoPE and attention. Q and K may have **different row counts** in cross attention. Keep the existing gain applied to exactly the local channels. Prefer a fused per-row stats pair only when both row shapes match. Explicitly test NaN/zero and epsilon behavior.

**No-card:** dense vs uneven-sharded RMS oracle, numerical stress and wrong-local-RMS negative control. **Hardware:** f32 reduction order, bf16 rounding, native TP1 Q/K comparison and collective latency by op kind.

### LT3-06 — Complete dual-stream DiT tensor parallelism (P0)

**Gap:** 48 blocks each contain video/audio self, text cross and cross-modal attentions, head gating, modulated norms, and separate FFNs. Existing Qwen sharding does not cover them.

**Implementation:** column-shard Q/K/V/gate/FFN-up, row-shard outputs/FFN-down; collective reduced result re-enters replicated residuals and adaptive norms. Keep video 4096, audio 2048 and A2V/V2A dimensions explicit. Tune rank ownership of FFN separately from head counts when justified. Verify every rank runs collectives in identical order for all branches.

**No-card:** tiny CPU linear reference graph for every op type; collective schedule equality. **Hardware:** block-level output parity, full 48-block BF16/IQ8R parity, deadlock and cancellation stress.

### LT3-07 — IQ8R encoding planes and quantization semantics (P0)

**Gap:** rot-int8 scale/u8/bf16 planes and FWHT128 over K may change under naive input column splitting.

**Implementation:** preserve full-input activation quantization and original per-256-K scales; only slice K on boundaries compatible with FWHT128 and scale group 256. Output-row splits should keep full encoded rows. Audit native LinearFP8 span, container row planes, RAD_MAP_ROWS/COLS, signed round/clamp and transformed weights before production mapping.

**No-card:** source row reconstruction and K boundary refusal; later add real CPU quant/dequant fixtures for code/scale/scale2. **Hardware:** gfx1100/gfx1201 GEMM/GEMV comparison across decode/prefill-like row counts.

### LT3-08 — Guidance and per-pass collective order (P0)

**Gap:** LTX may perform four DiT evaluations per step; conditional/unconditional, block 28 STG bypass and isolated cross-modal branch issue different paths.

**Implementation:** a shared native pass ticket/epoch chooses the same branch and collective schedule for every TP rank. Where a pass bypasses an attention, ensure all ranks bypass its Q/K and reduction stages consistently; don't leave a rank waiting on a skipped collective. Preserve guidance rescale and f32 sampler.

**No-card:** schedule fixtures for every guidance combination, positive/negative prompt, STG/ISO; refuse mismatched tickets. **Hardware:** all guided/unguided modes, exact step counts, video/audio parity.

### LT3-09 — VAE, image conditioning, vocoders, result ownership (P0)

**Gap:** VCOND includes optional first-frame VAE encoder; ADECODE/VDECODE/VUPSCALE and diffusion decoder are currently single-rank and write shared output.

**Implementation:** initially keep non-DiT passes on a single chosen fast GPU; other ranks receive only required latent/conditioning data and synchronize but never execute duplicate output writers. Preserve tiling, frame/audio timing, diffusion decoder and duration head. Define explicit request cleanup and worker cancellation.

**No-card:** rank/pass ownership graph, duplicate-writer prevention and simulated cancellation. **Hardware:** T2VA, I2VA, auto-duration, VAE/diffusion decoder, audio and upscale modes.

### LT3-10 — Native Gemma 4 on gfx1030 (P1, separate from TP3)

**Gap:** libr2/units.txt has only three helper kernels; no complete Gemma4 stack, dense IQ8R W8A8, rotary/casual/sliding attention, norms, FFN or embeddings.

**Implementation:** inventory every op declared by arch/ltx2/ltx2.cpp declare_te. Add genuine gfx1030 C++/HIP kernels to the existing libr2 component using its RDNA2 16x16 emulation where correct, extending unit coverage only after nonstub kernel schemas and tests. Add no generic GPU backend and no Python inferencer.

**No-card:** registry coverage/shape/quant fixture; fail closed on missing units. **Hardware:** rad-kbench all Gemma operators, 6900 embeddings vs fast-card TP1 at identical prompts.

### LT3-11 — 6900 memory fit and native conditioning role (P1)

**Gap:** nominal 16 GiB gfx1030 VRAM minus configured 1 GiB reserve and approximately 13 GiB text weights leaves little room for activations, a 49-level bf16 stack and temporary buffers. archs.toml admits gfx1030 only as helper. Current vision-tower-placement supports RAM or local VRAM, not a distinct GPU owner.

**Implementation:** separate typed conditioning role, not MoE helper or fourth TP rank. Start with Gemma-only on 6900 and connectors on a fast rank; measure real memory workspace and allocation fragmentation. Refuse if weights+scratch+live reserve exceed the actual card. Preserve RAM fallback if optional worker fails qualification, never silently substitute a new behavior in a measured run.

**No-card:** disjoint BDF, architecture, budget and unknown-scratch refusal. **Hardware:** complete Gemma prompt at peak resident VRAM, both prompts, sustained repetition.

### LT3-12 — Cross-device text handoff and VCOND split (P1)

**Gap:** the current VCOND executes text encoder, connector and duration head locally and has no cross-process handoff. Transferring the final 24 MiB contexts requires relocating both large connectors; **first** transfer the ~735 MiB full stack.

**Implementation:** native producer on gfx1030 computes Gemma stack for positive/negative prompts, publishes immutable schema/shape/epoch/request identifier and completion fence to a host-staged transfer; a fast-card native consumer runs connectors and duration head, then copies/broadcasts resulting contexts as needed for TP3. Bound pinned buffers, validate receipt and lifetime, provide cancel/worker-death behavior. No per-denoising-step text transfer.

**No-card:** producer/consumer mock with large logical buffer, offset/order, stale epoch, aborted request and checksum negative cases. **Hardware:** actual PCIe3 x4 bandwidth/overlap, lifetime safety, prompt TTFT change.

### LT3-13 — Config, measurement and safe resource admission (P1)

**Gap:** current configs support engine ranks and MoE helpers, not standalone conditioning card; RUM04 will migrate native YAML. A fake helper role or argv projection would bypass the measured admission contract.

**Implementation:** add one narrow native conditioning-device field/section and bind its exact YAML bytes, model/build digest, BDF and resource estimate through the existing BC measured manifest path. Preserve all engine-rank order, live inventory/reservation checks and sealed replay. Do not add another YAML serializer or independent allocator; retain Python-free release.

**No-card:** strict config schema, S8 replay drift, model/SDK mismatch, duplicate/overlapping devices, memory rejection and copy-lifetime tests. **Hardware:** bc-serve starts/stops/replays exact four-card contract and refuses stale or stolen GPU reservations.

### LT3-14 — Performance and parity evidence (P0 and P1)

**Gap:** Qwen 27B TP3 throughput is not evidence for LTX video. Host-staged per-block reductions across large Q/K and FFN activations might outweigh sharding gains.

**Implementation:** measure TP1 vs 3-rank DiT, then TP3 vs TP3+text separately with exact frozen build/model, prompt/negative prompt, size/frame count, seed, guidance modes and decoder. Record Gemma time, transfer time, per-layer QK norm collective, GEMMs, all-reduces, video+audio decode, peak VRAM, whole clip wall time, and numerical intermediate hashes/reference comparisons. Keep raw evidence under the plan's run/evidence owner, never invent results.

**No-card:** fixture/schema validation and missing-evidence refusal. **Hardware:** ABBA runs on same host, stable power/session conditions, reference image/audio/latent quality gates approved by owner, no deadlock. A proposed release target is at least 20% faster end-to-end for the default 121-frame video (not a prediction or claimed result); optimize or withhold TP3 if it regresses.

### LT3-15 — Release/rollback and documentation (P0 and P1)

**Gap:** HF container is gated, upstream pin/ABI is changing, release packaging must remain Python-free.

**Implementation:** opt-in named LTX TP3 config only after P0 hardware qualification; optional -text6900 variant only after P1. Include native binaries, matching plugin ABI and verified config in release allowlist, no Python interpreter/packages. Document license access, GPU role topology, fallback to TP1, 6900 unavailable/failure, no-P2P constraints and quality thresholds.

**No-card:** release inventory/ABI/reproducibility and forbidden terms, install/rollback check. **Hardware:** installed-bundle generation, full start/cancel/restart/shutdown.

## Numerical and resource invariants

- **Global RMS:** for each Q or K row rank r computes sr = sum(xri^2). The collective yields S = sum(sr). Each rank computes yri = bf16((xri / sqrt(S / (H*hd) + epsilon)) * gainri). The mock uses double accumulation to isolate algebra; real HIP uses a specific f32 reduction/bf16 order to validate against TP1. Local per-rank RMS is wrong.
- **Fused QKV:** each rank's compact Q,K,V receives source spans at offset, D+offset, 2D+offset, respectively; naïve contiguous [3D]/3 is wrong.
- **IQ8R:** K cuts must align to 256 (also divisible by FWHT128); encoded row slicing by output rows is different from recomputing per-shard activation scales. Alignment alone is not numerical proof.
- **Transfer sizes:** 770,703,360 bytes is the **full Gemma intermediate stack** for two padded prompts, whereas 25,165,824 bytes is the **final connector context** for two prompts. The latter is possible only with connector placement and memory proof.
- **Resource uncertainty:** a mock 6900 16 GiB minus 1 GiB reserve and 13 GiB weights is not evidence of fit. Require measured workspace and margin. No real GPU access without a lease.

## Narrow next code implementation seam (verified against frozen candidate)

Existing ABI already exports rad_weight_shard_span_parts (abi/rad_builder.h:553-572) and LinearFP8::span_parts (arch/common/rad_fp8.h:924-945), which slices the code and scale planes together. arch/gemma4/gemma4.cpp:866-890 demonstrates fused Q/K/V row-part spans. Do not add a second sharding ABI or manually slice IQ8R planes without proving the helper covers the rotated scale encoding. The LTX-specific owner must adapt arch/ltx2/ltx2.cpp Lin::declare (around 229-285) and attn_declare (around 1020-1115), retain the full metadata width for global Q/K norm, and use an opt-in guarded builder while TP1 remains unchanged. Source edits belong in a managed Radiance patch package, not a directly maintained upstream mirror.

## PR sequence and acceptance boundaries

1. **Foundation:** this design plus isolated native card-free oracle and CI test. The oracle is not imported into production; no new LTX runtime flags, no weakening of TP1 refusal.
2. **Managed LTX patch package:** following RUM completion, introduce metadata-driven head/FFN ownership and encoded QKV mapping, source applicability and tests on frozen candidate. Keep launch refused until functional.
3. **Distributed Q/K norm:** new HIP partial sum and existing exact collective; device comparison for every self/text/cross attention.
4. **Complete DiT TP3:** output/FFN reductions, gates, guidance and leader-only generator passes, with full 48-block numerical comparisons.
5. **TP3 only qualification:** optimize collective placement after profiling, publish native YAML/manifest and a named opt-in config, restoreable TP1 fallback.
6. **6900 Gemma kernels:** separate nonstub kernel coverage through libr2 with actual IQ8R and attention tests.
7. **Text worker and measured four-card contract:** split VCOND, publish Gemma stack with request/epoch/shape checks, fast-card connector consumption, native admission and cleanup.
8. **Four-card qualification and release:** real 6900 peak VRAM and PCIe link timing, numerical quality, ABBA end-to-end benefit, Python-free installed bundle.

Each slice must undergo its CTests/patch tests, patch-lint/forbidden-terms, review, GitHub required checks and confirmed merge. Mark applicability, compilation, GPU activation, numerical quality and performance **separately**.

## What is validated in this foundation slice

- tools/lab/radiance/ltx-tp3/ltx_tp3_reference.h and its C++ CTest cover: nonuniform 32-head maps, fused QKV bank source offsets, IQ8R 256-K boundaries, all-head normalization vs dense reference and a deliberately wrong local-RMS negative control, uniqueness/memory refusal for four mock cards, and both conditioning shapes.
- tools/tests/lab/test_ltx_tp3_reference.py runs the native CMake test through the existing tooling tests; **Python is never in the engine**.
- The second native CTest exercises guidance-specific per-rank collective schedule equality, STG block-28 and isolated-branch omissions, leader-only execution ownership, bias-once projection reduction, and a conditioning mailbox that refuses stale epochs, requests, mismatched shapes, replay, pending overwrite and cancellation. This is a test protocol, not a production GPU IPC mechanism.
- The third native CTest simulates multirow self-attention and text cross-attention with all-head RMS, per-head softmax and gating, QKV bank ownership and row-parallel output across three asymmetric splits. Both alternatives agree within a strict f64 fixture tolerance. No RoPE, BF16 rounding, IQ8R codec or device kernel is exercised. Mock transfer/memory extents refuse overflowing arithmetic.
- This reference neither changes production execution nor establishes actual IQ8R encoding-plane parity, GPU kernels, TP3 execution, 6900 memory fit, video/audio quality or end-to-end speed.

## Required Planning and Ledger handoff

The session's available tools did **not** expose the authoritative Planning MCP or Ledger MCP, and the remote shell policy denies the project worktree executable. Per docs/reference/agents/WORKFLOW.md, **do not hand-edit** docs/planning/active items or docs/releases/events files. The owning Planning MCP group must be found/created and LT3-01–15 registered there when access is available. Any substantive PR requires a Ledger MCP unreleased event as part of the normal workflow. This design is a concrete implementation handoff, not a counterfeit authoritative plan or release event.
