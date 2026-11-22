# LTX02 C06 — native IQ8R weight-share and AQL readiness slice

**2026-11-22. Result: PASS for native host weight-share contract and all host regressions; GPU AQL binary built but not run.** This is a narrow continuation of the merged C06 instance test (PR #1301), not full checkpoint-backed C06 execution, TP1 parity, or TP3 serving admission.

## Frozen source and intent

- Source: Radiance 2.2 / ABI44 `arch/ltx2/ltx2.cpp`, SHA-256 `543490eb70829693f945233d1b66e2d6d87b5e6a2817a72233fa0649218e6b53`.
- Unselected diagnostic patch: `rad0060_ltx2_tp3_linear_declare/patch.py`, SHA-256 `964d6dea9001b052092aca3124f188dc4766f89cfbed8c9ae4ac6e5aadf87ba2`. The source-managed patch is unchanged; TP1 and the production world>1 refusal are unchanged.
- Actual read-only LTX-2.5 IQ8R RAD2 container, block-0 self-attention, audio and video, both 8/8/16 and 12/12/8 head ownership maps. Each case uses all three rank-specific native Builders.
- Reuse the existing `ltx_c02_lin_declare_test.cpp` and **the upstream** `rad::weight_share` from `core/format/share.h`. This checks the actual `WeightInfo` selection from the same graph that C06 would execute. No new load algorithm or alternate model reader is introduced.

## Native weight-share finding

For each of **12 rank / modality / split cases**, the native graph declares **10 model weights per rank**; `weight_share` accepts every real selected plane and the test verifies that each rectangle is nonempty, within the source plane and densely covers its declared share. Identity layouts must agree byte-for-byte with their native stored extent.

Every rank has exactly **two non-identity layouts**:

- `dit.blk.0.{attn1,audio_attn1}.qkv.scale` from the corresponding `qkv.weight` source, `rad.planes` layout.
- `dit.blk.0.{attn1,audio_attn1}.out.scale` from the corresponding `out.weight` source, `rad.planes` layout.

Each needs **256 more stored bytes than the selected source plane**, i.e. **512 bytes extra per rank** across the two scale weights. The layout is native kernel-selected, not arbitrary padding. Directly binding the raw checkpoint plane for these weights would bypass the engine's relayout requirement and would not qualify as a valid C06 inference test.

| Modality | Head split | Ranks 0 / 1: selected → stored bytes, each | Rank 2: selected → stored bytes |
| --- | --- | ---: | ---: |
| Video | 8/8/16 | 17,150,016 → 17,150,528 | 34,005,056 → 34,005,568 |
| Video | 12/12/8 | 25,577,536 → 25,578,048 | 17,150,016 → 17,150,528 |
| Audio | 8/8/16 | 4,364,352 → 4,364,864 | 8,581,184 → 8,581,696 |
| Audio | 12/12/8 | 6,472,768 → 6,473,280 | 4,364,352 → 4,364,864 |

Source of these numbers: `ltx_c06_native_weight_share_contract` native output. They are test-case bytes, **not VRAM admission or runtime memory estimates**.

## AQL readiness without another launcher

- Added optional `ltx_c06_instancing_aql_test` build target for the **same** C06 model-backed graph declaration, using the existing C05 HIP-enabled Radiance core linkage and isolated `libr3/libr4d/libref` kernel directory. No separate graph, dispatch or collective is created.
- Test-only `ltx_native_aql_roster.h` consolidates previously duplicated three-device, architecture, peer fact and AQL-backend checks from C05 and C06. Production device admission remains RUM05-owned.
- `--c06-instances-aql` requires the Radiance AQL backend and three visible gfx1100/gfx1201 devices. It refuses missing device kernel bands and uses concurrent rank Builder finishing, as the native exact collective requires. Instances are retired **before** the three Builders' frozen geometry is destroyed, including exception unwinds. This does **not** allocate model weights or call `RadCtx::run_step`.
- A host-only `ltx_c06_aql_refuses_host_backend` negative CTest ensures AQL-only mode cannot be run against `libref`.
- The AQL C05 and C06 executables were compiled and linked with the frozen Radiance **2.2.0, RAD_WITH_HIP=ON** core. **Neither AQL executable was run in this slice**; exclusive GPU access was held / queued for RUM06. No device availability, throughput, or numerical claims are made for this new build.

## Verification

- Native real-model host suite: **9/9 passed**, including C01, C02, C04, C05, C06 graph, C06 36-instance rendezvous, new C06 weight-share contract, fail-closed AQL-on-host negative and C05 issued host Q/K oracle.
- Focused Python tests with actual frozen source and IQ8R model: **12/12 passed**, no skips.
- Both HIP-enabled AQL binaries: **built**, not executed.
- `git diff --check`: pass. No production patch edits, GPU jobs or lease use by this work.

## Exact next integration gate

The model-aware `RadCtx` test must bind weights via Radiance's **production** container share, staging and layout pipeline. The share arithmetic is exported as `rad::weight_share`, but the actual source read/relayout pipeline is owned privately by `core/engine_bringup.cpp`: `plan_container_weight`, `RelayoutPool` and `sweep_container`. Do not copy those routines into an LTX test or bind `rad.planes` scale weights directly. Prefer running a constrained pre-admitted native Engine or a narrowly extracted shared loader seam that replaces, rather than forks, production code and belongs to the generic Radiance owner. Keep the serving world>1 refusal intact.

Once that path is available, issue the **actual** native `Act::step`, `Lin::step` and `attn_core` sequence with checkpoint-backed weights in `RadCtx`, compare captured TP1 intermediates for both modalities and both legal splits, then rerun on leased gfx1100/gfx1100/gfx1201 cards, including rank failure and timeout. Treat RUM06's generic group lifecycle and RUM13's AQL qualification as dependencies, not duplicated LTX implementations.

The next diagnostic GPU gate is the opt-in C06 AQL **instance** target; it must be run under a short exclusive GPU lease and an outer subprocess timeout. Successful instancing would **still not** close LTX02.
