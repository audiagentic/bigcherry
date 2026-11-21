# LTX-2.5 IQ8R C07 — native exact TP3 collective on GPU

**Date:** 2026-11-22. **Branch:** `agent/ltx25-tp3-main22`.
**Source/build:** Radiance 2.2.0, ABI44, frozen source and matching native core, installed `libr3.so`/ `libr4d.so`.

## Result: two controls GREEN, heterogeneous GPU RED

| Visible physical cards | Rank placements | Native plugin(s) | Result |
|---|---|---|---|
| R9700 only | 3 independent streams/ranks on one `gfx1201` | `libr4d` | **PASS**: 21/21 exact BF16 rank-by-case comparisons; no mismatches |
| Both RX 7900 XTX | 2 ranks on XTX0, 1 on XTX1 (`gfx1100`) | `libr3` | **PASS**: 21/21 exact BF16 rank-by-case comparisons; no mismatches |
| XTX0 + R9700 | 2 ranks on XTX0, 1 on R9700 | `libr3` + `libr4d` | **FAIL**: SIGSEGV (exit -11) at first native GPU collective launch |
| XTX0 + XTX1 + R9700 | one rank per card, `gfx1100/gfx1100/gfx1201` | `libr3` + `libr4d` | **FAIL**: SIGSEGV (exit -11) at first native GPU collective launch |

**No successful heterogeneous/mixed 3-card TP3 collective was demonstrated.**

The stand-alone native host-mailbox test uses `ar_oneshot_3rank_exact_host`, native RadKernelInfo `init/launch/fini`, the actual `rad_bc_shared_slot` from Radiance, and actual card-specific HIP streams. The positive controls complete seven C06-related patterns on **all three ranks** with bit-exact BF16 outputs:
video/audio Q/K zero-packed rows at **8/8/16** and **12/12/8** head splits, overlapping video output partial sums in place, overlapping audio output partial sums out of place, and a one-row video Q/K transfer.

This proof is **not C06 model-weight graph execution**: operands are deterministic synthetic BF16 rows at the real head geometry. The earlier C06 model-backed kernel tests independently verified IQ8R GEMM on each card; this C07 test verifies only the collective boundary.

## Failure isolation and exact observation

- On mixed cards, `hipDeviceCanAccessPeer` reports **0** for all directed card pairs; the production-relevant path is host-pinned coherent mailbox, not GPU P2P.
- The native cross-rank initialization succeeds on **all ranks**: `init rc=0`, and the native three-rank weighted setup derives **w4/w4/w1** from the two XTX links and R9700 x4 link. The source emits duplex settings `64,64,0`.
- The two-XTX control successfully traverses two *physical* GPUs with no direct P2P, so having distinct devices or no P2P **alone** does not explain the failure.
- The R9700-only control successfully executes three independent streams/ranks under the same host-mailbox logic, so the BF16 arithmetic and basic rendezvous work independently.
- When an R9700 is combined with an XTX, the first `video q/k 8,8,16` collective, **16,384 BF16 elements per rank**, hits SIGSEGV before any rank's output comparison. The runner returns `-11`, not a silent skip.
- GDB on the earlier mixed-card reproducer identified the faulting stack as `libamdhip64.so.7` called from `libr4d.so:r4d_ar_oneshot_3rank_exact` -> `r4d_rad_ar_oneshot_3rank_exact`. A separate direct `ar_twoshot_3rank_exact_host` attempt also returned `-11`. There is **no proof of the underlying root cause**. The failure could involve the standalone direct-plugin execution context; it has not been reproduced through Radiance's actual session/engine owner.
- The same real model/device libraries already ran separately with zero IQ8R/GEMM failures on all three cards (C06 GPU evidence), so this does **not** establish an IQ8R projection failure.

Raw receipts/logs are in the project work root under `ltx25-tp3-c07-proof/`:
- `ar_oneshot_3rank_exact_host-devices-0-0-0-visible-2.{json,log}` — R9700-only GREEN
- `ar_oneshot_3rank_exact_host-devices-0-0-1-visible-0-1.{json,log}` — two-XTX GREEN
- `ar_oneshot_3rank_exact_host-devices-0-0-1-visible-0-2.{json,log}` — XTX+R9700 RED
- `ar_oneshot_3rank_exact_host-devices-0-1-2-visible-0-1-2.{json,log}` — intended three-card RED

There is also an earlier GDB trace in the work-root diagnostic run history; the backtrace summary above is sufficient to reproduce with a fresh isolated debugger.

## Next, smallest correct fix gate

**Do not enable LTX TP3 serving.** First reproduce this exact mixed `all_reduce` in the **existing Radiance engine runtime/dispatch owner**, with correct selected kernel initialization, shared-slot lifetime and device contexts. Compare direct native-row launch to engine-issued launch on the same cards, retaining fixed per-rank ABI44 build/graph/owner receipt. If real engine dispatch passes, correct the standalone C07 harness to use the existing device dispatch wrapper; if it fails too, isolate the native kernel/ROCm interaction under the real owner. Do **not** add an alternative all-reduce implementation, fork the scheduler or relax exactness.

Only after a **mixed three-card exact BF16 collective passes** should the C06 Q/K zero-pack + full-width RMS path be issued as one real self-attention component and compared against TP1. Full LTX DiT, guidance and TP3 admission remain gated. The existing 2.2 LTX production `world_size > 1` refusal is unchanged.

## Host proof / safeguards

The mixed-ar gate compiles against a version/source-verified native core and the existing upstream shared-slot implementation. The runner is an explicitly selected, timeout-bounded script with per-card/row JSON receipts. It does **not** run automatically in normal CTest and makes no GPU measurements without an agent lease. All test code is restricted to `tools/lab/radiance/ltx-tp3/mixed-ar-gate/`.


## C08 follow-up: production AQL dispatch clears this limitation

The next gated test linked a HIP-enabled Radiance 2.2 core whole, used
rad_dev_set, rad_stream_create, rad_dev_peer_launch and rad_stream_sync,
and preserved the same native exact all-reduce rows, heterogeneous GPU
libraries and real BigCherry shared-slot rendezvous. Mixed gfx1100/gfx1201
one-shot **and** two-shot host-mailbox runs each passed 33 of 33 exact BF16
comparisons, including 64-row batches. The direct-HIP SIGSEGV above is
specific to the original standalone launch path; it does not prove a
production native transport defect. See
docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md.
