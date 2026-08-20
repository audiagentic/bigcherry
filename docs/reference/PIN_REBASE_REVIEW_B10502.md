# llama.cpp pin rebase — comprehensive review: b10362 → b10502

Generated 2026-08-20. Every claim below was mechanically verified in this session
(dry-run of all 30 patch modules against a clean `b10502` checkout with the overlay
installed, `sources check`, and per-commit diff inspection) — not inferred from
release notes. Supersedes the earlier pre-repin P0/P1 summary, which targeted a
different tip (`d59d455f`, 21 commits *past* b10502) and contains one retracted
finding (see §2.1).

---

## 0. Repin state — what is actually true right now

| Fact | State |
| --- | --- |
| `recipes.toml` | `pinned = "b10502"` — **valid tag** = `0adcc3bb5710` (2026-08-19 10:23 +0300, "ci: add attestation for signed release artifacts"). |
| Pin window | `4801e3c5` (b10362) → `0adcc3bb` (b10502) = **140 commits**. |
| Vendor tree | **Still at `4801e3c` (b10362)** with the old selection applied. No `pull` to b10502 has run yet. |
| Release record `b10502` | Stage `audited`, audit **33/33 PASS** (verified this session on a clean checkout + overlay). |
| Stray record `d59d455fd8ea.json` | Stage `broken`, `release_tag: ""`. `d59d455f` is a **master tip 21 commits past b10502 with no tag** — a pre-overlay audit artifact from an earlier probe, **not the pin**. Delete or ignore. |
| Stale records | `680a9ae63d60.json` (stage pulled, 2026-08-12), `unknown.json` (2026-08-19) — pre-repin probes. |
| Newer releases now exist | `b10505` (`ee4c505a`), `b10506` (`8ef78e64`, published 2026-08-20 02:57 UTC — one minute *before* the repin record was written; `latest_release()` raced a release publish). Pinning b10502 is a legitimate snapshot; §6.8 is the move-or-stay decision. |
| Own-ref recipes | `[compat.recipe.b10257]` pins `b10257` explicitly (linux-multi lane) — unaffected by `pinned`. |

**Conclusion: the repin is recorded but not executed.** The vendor tree, all
campaign lanes, and all builds are still on b10362.

---

## 1. Mechanical patch verification (the core result)

Method: clean detached worktree at `0adcc3bb`, overlay (`src/`, 28 files) copied
exactly as `apply` does, then the in-memory two-pass dry run of all 62
`FilePatch`es (30 modules), for every selection that matters:

| Selection | Modules | Result |
| --- | --- | --- |
| **PRODUCTION** (`core`+`upstream-fixes`, state `validated` = `patch-set.framework` + 1000) | 27 FilePatches | **PASS — 0 failed edits** |
| **Each of the 11 experiment lanes** (production + exactly one 12xx) | 28–33 FilePatches each | **PASS — 0 failed edits each** |
| **Last-vendor selection** (`core`+`upstream-fixes`, all states — what was used for the current vendor apply) | 34 FilePatches | **FAIL — 1 edit: `1002 hip-unsafe-math-opt-in`** |
| **Double experiment** (rd12 + rd17 together) | 34 FilePatches | **FAIL — 2 edits: `1207 rd17-struct-host/device` — PRE-EXISTING, fails identically on the old pin** |

Every other edit lands with a unique anchor on b10502. The 20 "already-applied"
edits in the full run were all classified: 7 are self-chains inside 0100/0200/0830
(guard matches the module's own earlier edit — correct), 17 are genuine
mainline dedup (1003 ×11, 1004 ×5, `rd08-op-timing-flag` dedup between 1203/1204)
— see §2.2.

---

## 2. Patch-by-patch impact (all 30)

Upstream churn = commits in the 140-commit window touching the file(s) the patch
anchors. "Applied" = dry-run verified at b10502.

### 2.1 The retracted P0

Earlier review claimed `25ae3a9b` (#26843, DGX Spark GB10 nwarps) inserted
`MMVQ_PARAMETERS_GB10` after `MMVQ_PARAMETERS_RDNA3_0` and broke 1208's enum
anchor. **Verified false.** At b10502 the enum is:

```
MMVQ_PARAMETERS_RDNA3_0,
MMVQ_PARAMETERS_RDNA4,
MMVQ_PARAMETERS_GB10      <- appended last, no trailing comma
```

1208's anchor (`RDNA3_0,` → `RDNA4,`) is untouched; 1208 applies cleanly
(verified). What 25ae3a9b *did* do: extend `calc_nwarps`/`calc_launch_params`
with `small_k`/`halve_iters` params — which our 0600 was already re-anchored
against in the previous session. **Consequence worth knowing: 0600 now only
applies on bases ≥ 25ae3a9b. It fails on the old pin (verified: 5 failed edits
on b10362). The pin change is one-way forward until 0600 is dual-anchored or
accepted as new-base-only.**

### 2.2 The two real blockers / pre-existing bugs

**1002 `hip_unsafe_math_opt_in` — the only rebase-caused failure.**
Upstream `e79e4bf6` (#26696) **deleted** the line 1002 anchors on:
`set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -funsafe-math-optimizations")`.
The PR 1002 was backported from was OPEN at port time; it merged *differently*
— upstream removed the flag entirely instead of adding our
`GGML_HIP_UNSAFE_MATH` opt-in gate.

- Our current vendor build already runs **OFF** (1002 applied, default OFF), so
  the *effective* build behavior after rebase is **unchanged** — the
  "all numbers were measured with unsafe math" concern from the earlier summary
  does not apply to current builds.
- What is lost: the `-DGGML_HIP_UNSAFE_MATH=ON` escape hatch.
- Decision needed (action A1): **adapt** 1002 to insert the option +
  conditional flag against a surviving anchor (e.g. the `GGML_HIP_EXPORT_METRICS`
  block directly above the removed lines), or **deprecate** it (upstream default
  now equals our default). The gfx1100 MTP-temperature-0 divergence evidence
  (AIESW-40114 class) stays valid either way; keeping the opt-in preserves the
  ability to buy back fast-math speed on non-MTP workloads.

**1207 `rd17_moe_topk_down_fold` — pre-existing latent bug (not rebase-caused).**
Its struct anchor is the 2-line span `gate_scale` → `glu_op` in
`ggml_cuda_mm_fusion_args_{host,device}`. 1205 (rd12) inserts `dst_gate` between
exactly those two lines, so any selection containing **both** 1205 and 1207
fails. Verified: fails identically on old pin and b10502; single-lane
experiments (the only production path, per `[experiment.*]` one-patch-per-entry
design) pass. Fix options (action A2): make 1207's struct edits
1205-aware (two alternative anchors via `applies_if`), or record the
rd12×rd17 double-lane incompatibility as a documented constraint.

### 2.3 Redundant patches — superseded by mainline at *both* pins

**1003** (quantized cpy thread-block fix): its upstream source commit
`69bf6437` (#26731, merged 2026-08-08) **is an ancestor of our old pin
`4801e3c`** (pin is 2026-08-11). Verified: all 11 edits are "already-applied"
against pristine pin *and* pristine b10502 — the patch has been a silent no-op
since it was ported. **1004** (rms_norm_mul_rope fusion): the full fusion
(kernel, `should_fuse`, dispatch wiring, incl. the VIEW/SET_ROWS op-list)
already exists in the old pin — all 5 edits no-op on both pins.

Why this matters beyond tidiness: a no-op patch whose *guard* matches upstream
text becomes a **loud rebase failure the day upstream evolves that text**
(1004's rope files churn at the very next bump: `fe8156f7` #27120
`ggml_rope_set_offset`). The RD01-style mainline-dedup audit covered the
rdna-boosts fork but not the upstream-fixes group ported from already-merged
upstream commits — that's the process gap. Recommendation (action A3):
**deprecate both**, with provenance notes ("superseded by mainline
69bf6437 / <commit>; no-op at pin b10362 and b10502").

### 2.4 All other patches (verified applied at b10502)

| Patch | File(s) | Window churn on its files | Notes |
| --- | --- | --- | --- |
| 0100 cmake_options | ggml/CMakeLists ×4, ggml-hip/CMakeLists ×1 | version bumps ×3, `13fd0bb5` config-version, `e79e4bf6` | Anchors intact. GGML_VERSION 0.19.0→0.20.2 flows into build metadata. Install-only file rename `ggml-version.cmake`→`ggml-config-version.cmake` (no in-tree impact). |
| 0200 dispatch_hook | ggml-cuda.cu | 153d324b, 60eeeb60, 1692f9e5, ebb546b7 (see §3) | All 22 edits applied. **Compile check mandatory**: ebb546b7 adds a *new call site* to `ggml_cuda_should_use_mmf(..., /*mul_mat_id=*/true)` and `ggml_cuda_should_use_mmq(...)` — functions this patch wraps. Dry run can't catch C++-level signature mismatches. |
| 0300 mmq_forced_j | mmq.cu/cuh | none | Safe as-is. |
| 0400 mmvf_forced_block | mmvf.cu/cuh | none | Same new-call-site compile note as 0200. |
| 0500 mmf_forced_nwarps | mmf.cu/cuh | none | Safe as-is. |
| 0600 mmvq_geometry | mmvq.cu | 25ae3a9b | **Now new-base-only** (fails on old pin). Verified applied at b10502. |
| 0650 mmvq_native_variant | mmvq.cu | 25ae3a9b | Verified applied. |
| 0700 coverage_counters | mmvq/mmvf/mmf | 25ae3a9b (mmvq) | Verified applied. |
| 0800 server_shutdown_endpoint | server.cpp | 0021a77d (ui naming, cosmetic) | Verified applied. |
| 0810 replay_hit_diagnostics | ggml/CMakeLists, ggml-hip/CMakeLists, ggml-cuda.cu | as 0100/0200 | Verified applied. |
| 0820 measurement_signature_shapes | test-backend-ops.cpp | 6 commits | Verified applied. No GGML_VERSION embedding found in overlay/tools (grepped) — signature keys unchanged by the version bump. |
| 0830 split_reduce_telemetry | ggml-cuda.cu | as 0200 | Verified applied (2 self-chained already-applied edits, correct). |
| 0900 pool_workspace_metrics | common.cuh | none | Safe as-is. |
| 1000 rdna4_mmq_q2k_q6k_fix | mmq-vec-dot.cuh, mmq.cu/cuh | none | Safe as-is. |
| 1002 | ggml-hip/CMakeLists | e79e4bf6 | **BLOCKER — see §2.2.** |
| 1003 | cpy.cu | none | **No-op at both pins — deprecate (§2.3).** |
| 1004 | rope.cu/cuh, ggml-cuda.cu | none (next bump: fe8156f7) | **No-op at both pins — deprecate (§2.3).** |
| 1005 prompt_cache_checkpoint_selection | server-context.cpp, server-task.cpp | 533b1825, 22b8e310, 77918caf, decaf508, 5d9e5ac3 (large server reworks) | Verified applied. Behavioral smoke needed after build (slot save/restore semantics changed under it). |
| 1100 hi70_direct_op_evidence | test-backend-ops.cpp | 6 commits | Verified applied. |
| 1200 rd19_single_gpu_meta_bypass | ggml-backend-meta.cpp, src/llama.cpp | 153d324b (mmap_support caps) | Verified applied. **Behavioral**: default load-mode is now `auto` → mmap avoided on iGPUs by default — this is RD19's exact lane; iGPU load-path baseline shifts. |
| 1201 rd20_attn_gate_tp_split | src/llama-model.cpp | 5 model commits | Verified applied. |
| 1202 rd04_bf16_flash_attn_tile | fattn-tile.cu/cuh, fattn.cu, common.cuh | none | Safe as-is. RD04's pending cross-arch repeat is unaffected by this rebase anchor-wise. |
| 1203 rd050607 | mmq-vec-dot.cuh, ggml-cuda.cu, test-backend-ops.cpp, fattn.cu, mmq.cuh | 1692f9e5, ebb546b7, test churn | Verified applied. Fork source commit drifted (whitespace only, §4). |
| 1204 rd08_q6k_mmvq_vdr2 | vecdotq.cuh, ggml-cuda.cu, mmvq.cu, test-backend-ops.cpp | 25ae3a9b (mmvq) | Verified applied. `rd08-op-timing-flag` self-dedups against 1203's flag (correct). |
| 1205 rd12_paired_mmvq_dual_output | common.cuh, ggml-cuda.cu, mmvq.cu | 25ae3a9b (mmvq) | Verified applied (single lane). See 1207 conflict. |
| 1206 rd13_mul_mat_add_view_fusion | ggml-cuda.cu | as 0200 | Verified applied. |
| 1207 rd17_moe_topk_down_fold | common.cuh, mmvq.cu, ggml-cuda.cu | 25ae3a9b (mmvq) | Single lane verified applied. **Pre-existing 1205 conflict — §2.2.** |
| 1208 rd21_gfx1151_mmvq_nwarps_table | mmvq.cu | 25ae3a9b | Verified applied. Provenance doc must be updated for fork snapshot v3 (§4). |
| 1209 rd22_integrated_gpu_host_buffer_backout | ggml-cuda.cu | as 0200 | Verified applied. |
| 1210 rd26_bitidentical_decode_verify_standalone | ggml-cpu/llamafile/sgemm.cpp, ggml-cuda.cu | as 0200 | Verified applied. |

---

## 3. Commit-by-commit impact (the 140-commit window)

### 3.1 Blocks the rebase (apply-level)

- **`e79e4bf6` #26696** — `ggml-hip: remove -funsafe-math-optimizations`.
  Verified: deletes the flag + its comment (3 lines) from
  `ggml/src/ggml-hip/CMakeLists.txt`; nothing else in the file changed in the
  window. → 1002 anchor dies. (Only apply-level blocker.)

### 3.2 Behavioral / baseline-affecting (apply cleanly, but change numbers or behavior)

| Commit (PR) | Change (diff-verified) | Impact on us |
| --- | --- | --- |
| `153d324b` #26081 | Default **load-mode `auto`**; new `mmap_support` device cap (ggml-backend.h + meta + cuda props: `false` for IGPU type) | iGPU lanes (RD19/RD22 targets): default load path changes → **iGPU bench baselines shift**. llama.h gains load-mode API. Our bench scripts don't pass `--mmap` (grepped) so no flag breakage. |
| `ebb546b7` #26802 | CUDA graphs now kept for `mul_mat_id` unless the fallback path needs a stream sync (new `ggml_cuda_mul_mat_id_needs_sync`; `GGML_CUDA_CC_IS_AMD` small-batch path graphable) | We build `GGML_HIP_GRAPHS=ON`. **MoE workloads (Qwen) get more graph capture** → MoE decode baselines shift; interacts with the HI70 `GGML_CUDA_DISABLE_GRAPHS=1` workaround lineage. |
| `60eeeb60` #27083 | UMA memory override now skipped for HIP (`!defined(GGML_USE_HIP)`) | Memory reporting (`--fit`, free-memory display) changes on UMA hosts (Strix Halo). Discrete (gfx1100) functionally unaffected. |
| `1692f9e5` #26623 | `ggml_ssm_scan` gains `K` param (ggml.h signature 4→5 args); Mamba `K>1` now `can_execute=false` on the CUDA/HIP backend | Any Mamba/SSM workload: K>1 falls back to another backend → dispatch path changes. **Compile risk**: signature change — overlay doesn't call it (grepped), but 0200's wrapped region sits nearby; build check covers it. |
| `25ae3a9b` #26843 | MMVQ `small_k`/`halve_iters` params + GB10 table (nwarps=8 bs=1 dense, DGX Spark) | 0600/0650 re-anchored (done). Upstream "shape-aware warp count" convergence continues — third signal (#20831, #27233, #26843) that our nwarps-table work is converging with upstream; relevant to RD21's long-term dedup question. |
| `04b56914` #27138 | Common: share thread pools when `n_threads` differ | In-window and active at b10502 (**reverted after b10502** by `3e734467`, in the 21 later commits) → concurrent llama-bench numbers at b10502 may differ from both b10362 and future pins. Note for bench comparability. |
| `59886331` #26111 | CUDA warp-per-row wkv7 kernel (single-token decode) | New decode path for wkv7 models; not our bench models, but test-backend-ops churn (1100 file) — anchor verified. |
| `decaf508` #26920 | Server metrics refactor + correctness fixes (cached-vs-processed tokens, free-first-token, slot fast-release) | **Methodology upgrade**: server-bench endpoint numbers were contaminated pre-fix; post-rebase numbers are comparable in a new regime. 0810/1005 anchor regions moved — verified applied. |
| `22b8e310` #27133 | Server: redesign `yield_to_queue` thread model | Server concurrency behavior changes under 0800/1005. Smoke-test server lanes post-build. |
| `77918caf` #27041 | `/metrics` and `/slots` accessible during `llama_decode()` | Our monitoring scripts can poll during decode without 404s. |
| `5d9e5ac3` #26640 | Slot save/restore with media inputs (+ include/llama.h) | 1005 region; behavior smoke needed. |
| `533b1825` #27278 | Server: save processed mtmd chunks as placeholder | 1005 region; low risk. |
| `4c1a0af4` #26953 | Allow virtual iGPU devices (llama.cpp) | iGPU enumeration behavior — relevant to RD19's virtual-device story. |
| `0177dcc7` #26934 | `--mmap`/`--no-mmap` deprecated → `--load-mode` (common + llama-bench + scripts) | **No action** — we don't use `--mmap` (verified by grep across tools/tmp/recipes). Future scripts must use `--load-mode`. |
| `13fd0bb5` (ggml/1582) + `680a9ae6` #26839 | CMake config-version support; semantic versioning | Build metadata/LLAMA_VERSION semantics change; no anchor impact (verified). |
| `06ae2326`/`cea66f4c`/`da786dc2` | ggml 0.20.0/0.20.1/0.20.2 bumps | Version strings in artifacts. No GGML_VERSION usage in overlay/tools (grepped). |
| `d8a8beac` #25596 | GGUF loader hardening (malformed tensor dims/metadata) | Robustness only. |
| Model additions: `37333667` BailingMoE3, `ad1de39e` Kimi-K3, `16d222fc` MiniMax-01/M1, `6e62ba53` pocket-tts, `55f453b9` wavtokenizer bound, `7221e24f` is AFTER b10502 | New families in llama-model.cpp/llama.cpp | Loader growth only; 1201 anchors verified. |
| Spec/MTP engine: `5f754ea0` #24431 (`--models-dir` MTP assistant models), `1d2869c6` #27005 (auto-detect MTP draft type), `f65e568f` #26814 (auto-detect spec type from draft GGUF), `0d0bfcd4` #26958 (backend sampling for dflash & dspark), `9cd719af` #26275 (DSpark checkpoints), `a4a4c51f` #26925, `f785fc9e` #26904, `5d16e81d` #26903 | Speculative-decoding engine maturation | **MTP track**: when we build the E2E MTP bench lane, draft-model specification and spec-type handling have changed — EX02/HI71 reproduction and the future `--spec-type` lane should be re-validated on the new pin, and the auto-detect paths tested (they may change how our Qwen3.5-9B MTP setup is specified). |
| Other backends: TQ2_0 metal/vulkan, SYCL fusions, OpenCL ssm, wkv7, hexagon, vulkan dequant | `65091386`, `4a84b0ad`, `a7cd2f0e`, `1ee1cd9b`, `dc72703f`, `98d1e92c` (post-pin), `b062ba73` (post-pin) … | No HIP impact; shared-file churn only in test-backend-ops.cpp (anchors verified). |
| CI-only: attestation commits (`0adcc3bb`, `01ac3ad7`, `645ca283`, `7c35571e`…), rocm cache, ubuntu-rocm disable, windows-rocm, vgpr ignore lists | No code | None. |

### 3.3 The 21 commits AFTER b10502 (out of scope for this rebase — the NEXT-bump watch list)

Already diff-known from this session's earlier (wider) analysis:
`fe8156f7` #27120 rope_set_offset (**rope files → 1004's files if not retired**),
`d59d455f` #26502 tensor-split meta backend (**meta → 1200's file**),
`947fd9bb` #27376 server sleep refactor + metrics-during-sleep,
`ee0ea03a` #26347 models endpoints private under auth,
`7221e24f` #25505 Granite-SWA (llama-model.cpp → 1201's file),
`3e734467` #27337 thread-pool-sharing revert, `8ef78e64`/`ee4c505a`/`98d1e92c`/`5112b973` (metal/server-preset/vulkan/webgpu — no anchor impact), rest CI/other-backends.

---

## 4. Fork source state (stew675 rdna-boosts) — `sources check` findings

Run this session (online). **5 findings:**

1. **`FINDING rebased`**: active snapshot v2 head `9e46e1fdc` is **not an
   ancestor of the current fork tip `7a845b709`** — the branch was rebased a
   third time. All 33 tracked commits re-committed. Per the registry's own
   rules, a **snapshot v3 + re-audit is required** before any further porting
   decisions. New base (verified): **`fe8156f7`** — i.e. mainline *past* our
   b10502 pin (includes the rope_set_offset commit).
2. **Content-identical re-commits: 29/33** — including the entire RD25
   bit-identical cluster (`93510434f`, `b2655d381`, `d152888fc`, `10b83d6b2`,
   `6cdf5aff9`) and the RD21 fix `8cdf1ab08` → `17931495c`. RD25/RD26 port
   plans remain valid; re-derive anchors against the new base.
3. **`FINDING drifted` RD21 `1818c3b37` → `56c7bb14a` (ported as 1208)**:
   diffed — **content-equivalent**. The new commit is rebased onto a base that
   already contains GB10 (so the enum hunk shows GB10 in context) and keeps the
   pre-fix `ncols_dst == 1` table (the fix lives in the separate
   `8cdf1ab08`/`17931495c` commit, which we baked in per the RD25 rule). Our
   1208 is correct; only the provenance block needs the v3 SHAs.
4. **`FINDING drifted` RD05/06/07 `1d525bd45` → `5711c243c` (ported as 1203)**:
   diffed — **whitespace-only** (trailing blank line dropped in the
   test-backend-ops perf block; 25→24 lines added). No semantic change.
5. **`FINDING drifted` RD18 `8473c0da7` → `b13da2b4c` and RD27 `0510d7cfa` →
   `1a3212dd0` (NOT yet ported)**: genuinely re-adapted to the newer base —
   RD18's `rope_multi` hunk now targets the `inplace` param (was `is_imrope`
   on v2's base) and RD27's mmvq template line now includes `halve_iters`.
   **Future ports of RD18/RD27 must use the new commits** — and RD18 in
   particular: b10502's `rope_multi` may still have the old parameter shape
   (the `inplace` rename comes with the post-b10502 rope work), so RD18's port
   may need base-conditional handling. Flag for the RD18 plan item.

Mainline dedup: **0/33 fork commits merged into ggml master** (still, at tip
`d59d455f`). No rdna-boosts patch is redundant.

---

## 5. Non-patch consequences

1. **Tuning inventory is invalidated.** Tune winners/journal entries are keyed
   to base + build descriptor; a new pin means the binary identity changes.
   Full re-inventory + re-tune on b10502 (HI35 flow) before trusting any
   tuning result against the new base.
2. **All RD bench results are b10362-era.** They remain valid as
   b10362-era evidence, but every ported patch's bench verdict must be
   re-established on b10502 before promotion — and the §3.2 behavioral changes
   (graphs, load-mode, UMA, thread pools) mean native baselines shift
   independently of the patches. Re-run at least: RD04 pair, an MoE model
   (graphs), one iGPU load (1200/1219).
3. **Campaign acceptance records** tied to b10362 (RE-item acceptances,
   release-probe logs in `artifacts/`) stay historical; the b10502 acceptance
   pass is a fresh event per the RE process.
4. **The compile gate is real, not bureaucratic.** Dry-run proves anchors, not
   C++. Known compile risks: (a) ebb546b7's new call sites into functions
   wrapped by 0200/0400; (b) ggml.h `ssm_scan` signature; (c)
   ggml-backend.h caps struct growth. Only a full ROCm build (local gfx1100 +
   Brutus gfx1201/gfx1030) closes this.
5. **Windows/ROCm toolchain.** b10502 is ~2 weeks newer than b10362; upstream
   CI compiles newer clang. Our ROCm 7.1/Windows build is the risk bearer —
   expect the possibility of toolchain-side fixes that are *not* patch anchors
   (would become new framework patches if needed — the "create a new patch"
   case; none anticipated from the diffs seen).
6. **MTP/speculative track.** The server reworks (metrics, yield_to_queue,
   sleep post-pin) and spec auto-detect changes land underneath EX02 (MMQ Q6_K
   illegal-memory-access under MTP) and the HI71 fix. Re-reproduce EX02 on the
   new pin before drawing further conclusions, and treat
   `--spec-type`/`--models-dir` behavior as changed.
7. **Release/records hygiene.** Delete/annotate the stray `d59d455fd8ea`
   (broken, untagged master tip) and stale probe records; the `b10502` record
   should advance from `audited` → `patched` once the real apply + build
   passes.

---

## 6. Action list (ordered)

| # | Action | Why | Effort | Status |
| --- | --- | --- | --- | --- |
| A1 | **Decide 1002**: adapt (insert `GGML_HIP_UNSAFE_MATH` option + conditional flag against the surviving `GGML_HIP_EXPORT_METRICS` anchor) **or** deprecate (upstream default now equals ours). Record decision + ledger event. | Only rebase-caused apply blocker | S | **DONE** — adapted, re-anchored to `GGML_HIP_EXPORT_METRICS` (insert_after). Escape hatch preserved. |
| A2 | **Fix 1207 struct anchors** to be 1205-aware (alternative anchors via `applies_if`), or document the rd12×rd17 double-lane conflict in the plan item. | Pre-existing latent bug; blocks any future combined lane | S | **DONE (documented)** — every `[experiment.*]` entry is one patch, so no production selection ever hits this; noted in 1207's own docstring for whoever plans a combined lane later. |
| A3 | **Deprecate 1003 + 1004** with provenance notes (already mainline at both pins; anchor liability next bump). | Process gap: upstream-fixes dedup was never checked against the pin | S | **DONE** — `STATE = "rejected"` (this project's existing convention for "never selected by default"), provenance notes added to both docstrings. |
| A4 | **Record snapshot v3** in `external-sources.toml` (head `7a845b709`, base `fe8156f7`), re-audit note (29 identical / 4 drifted, content verdicts in §4), update 1208 provenance SHAs; note RD18/RD27 must port from the new commits. | Registry rule: rebase of source = finding, snapshot change is explicit | M | Not done here — `external-sources.toml` is being actively edited by the other concurrent session tracking the fork; left for them. |
| A5 | **Execute the rebase**: `pull` recipes to b10502 → audit (33/33 verified) → apply production selection (verified clean) → **full ROCm builds** (local + Brutus) → smoke (server incl. 0800/1005 endpoints, llama-bench dense + MoE, iGPU load path, MTP crash repro per HI71/EX02). | The actual rebase; compile gate is the real risk | L | Partial — the full `bigcherry:control` production build across all three platform-declared architectures (gfx1100, gfx1201, gfx1030) compiled successfully on Brutus against b10502 with all A1-A3 patch content applied. Closes the compile-gate risk (ebb546b7's new call sites, ssm_scan signature growth, ggml-backend.h caps struct) -- none of it broke the build. Windows/local ROCm build and the smoke suite (server endpoints, llama-bench dense+MoE, iGPU load path, MTP crash repro) still open. |
| A6 | **Re-baseline**: re-inventory + re-tune on b10502; re-bench ported RD patches (min: RD04 pair, MoE, iGPU load); tag all pre-rebase numbers as b10362-era in their bench records. | §5.1–5.2 | L | Not started — depends on A5 completing. |
| A7 | **Records hygiene**: remove/annotate `d59d455fd8ea.json`, `680a9ae63d60.json`, `unknown.json`; advance `b10502` record to `patched` after A5. | Stray "broken" record at the pin's index head is misleading | S | **DONE (partial)** — all three stray/stale records removed, `releases/index.json` rebuilt via `releases._rebuild_index()`. `b10502` record stays at `audited`, not advanced to `patched`, until A5 actually completes. |
| A8 | **Decision: stay at b10502 or move to b10506 now** (+5 commits: metal dequant, server dedup preset, vulkan/webgpu, ci — no anchor impact on our files; verified). Cheap now, an extra rebase later. | The repin raced a release publish; b10506 exists | S (decision only) | **DECIDED: stay at b10502.** Don't re-target mid-rebase for a 5-commit, no-anchor-impact difference; b10506 (or whatever is current) is a cheap next bump once this cycle actually closes (A5/A6 done). |
| A9 | **Next-bump watch list** (post-b10502 commits, §3.3): `fe8156f7` rope (1004 if not retired), `d59d455f` meta (1200), `947fd9bb` server sleep, `ee0ea03a` server models, `7221e24f` Granite-SWA (1201), `3e734467` thread-pool revert. Pre-verify before the next repin. | Don't discover rebase commit-by-commit next time | S (doc) | Already recorded above (§3.3) — no further action needed until the next repin. |
