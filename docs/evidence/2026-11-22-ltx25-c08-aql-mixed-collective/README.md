# LTX-2.5 IQ8R C08: real Radiance AQL three-card exact collective

**Date:** 2026-11-22.
**Scope:** Native exact BF16 all-reduce on two gfx1100 XTX cards plus one gfx1201 R9700, using Radiance 2.2/ABI44 and the *existing* libr3/libr4d kernel implementations.
**Result: PASS** for both one-shot and weighted two-shot host-mailbox variants. **The production LTX graph and TP3 serving are not activated.**

## Control and differential diagnosis

C07 had reported a SIGSEGV on a direct HIP launch in the mixed-card process. All three rank initializations had succeeded but the first reduction never returned; controls using only gfx1100 or only gfx1201 passed. That C07 result was **not evidence against Radiance's normal GPU dispatch**.

Radiance 2.2 does not use ordinary HIP kernel dispatch in production. core/device/aql.cpp implements a native AQL backend and exports hipLaunchKernel and the HIP fat-binary registration hooks. Its __hipRegisterFatBinary handles a mixed-system case in which HIP's registration decision for device 0 omits a different architecture on a later device (see the BC_VTP14_N1_AQL_REGISTER_FAT comment). A direct HIP launcher bypasses that backend and its per-card code object resolution.

C08 adds an **optional second executable** to the existing C07 lab (no second collective implementation): it links the **exact pinned HIP-enabled 2.2 SDK core archive whole**, uses rad::device_backend_name() to require "aql", selects each rank's real card with rad_dev_set, creates a Radiance AQL stream with rad_stream_create, brackets the native collective launch with rad_dev_peer_launch(1/0), and synchronizes with rad_stream_sync. It uses the same native RadKernelInfo init/launch/fini as C07 and the unchanged bc_shared_slot service. The simple test still calls native kernel entries directly rather than building/running a production Ctx program; HIP buffer allocations and transfers remain part of this isolated test.

The executable exports the real engine's hipLaunchKernel, __hipRegisterFatBinary, __hipRegisterFunction, rad_dev_set, and rad_bc_shared_slot. The source and version of the HIP-enabled core are validated at CMake configure time; a HIP-off archive is refused. **No Radiance upstream source, existing transport, or production patch has been changed.**

## Live GPU results

All rank-to-rank HIP peer-access queries reported 0, so this exercised the pinned coherent host mailbox rather than direct VRAM peer access. At startup the three-rank weighted path derived w4/w4/w1 from the two XTX PCIe paths and the R9700 link. The native rendezvous recorded both libr3.so and libr4d.so using the **same shared slot**.

| GPU mapping | Device backend | Native entry | Result |
|---|---|---|---|
| Three ranks on one R9700 | AQL | exact one-shot host | PASS, 21/21 BF16 outputs |
| Two ranks XTX0 + one R9700 | AQL | exact one-shot host | PASS, 21/21 BF16 outputs |
| XTX0 + XTX1 + R9700 | AQL | exact one-shot host | PASS, 33/33 BF16 outputs |
| XTX0 + XTX1 + R9700 | AQL | exact two-shot host | PASS, 33/33 BF16 outputs |
| XTX0 + XTX1 + R9700 | Direct HIP C07 (historical negative control) | exact one-shot host | SIGSEGV on first launch, no comparison |

The full three-card final runs each checked **11 scenarios per rank** without any mismatches (66 comparisons across the two collective entry variants). The seven original C07 cases included video/audio Q/K zero-packing at head splits 8/8/16 and 12/12/8, replicated output partial sums in place and out of place, and a one-row case. C08 adds video Q/K 16 rows, audio Q/K 16 rows, video Q/K **64 rows x 4096 elements**, and video output 64 rows. That last message is **262,144 BF16 elements, 512 KiB per rank**. The native init scratch upper bound is derived from these declared test cases, not a hand-picked model limit.

Both variants returned native init=0 and launch=0 for all three ranks and wrote zero-mismatch numerical receipts. The one-shot entry may internally hand off above its configured threshold to the existing exact two-shot implementation. The explicit two-shot entry also ran successfully.

### Exact evidence paths

Receipts are written outside the checkout, at BC_WORK_ROOT/ltx25-tp3-c08-proof/:
- ar_oneshot_3rank_exact_host-devices-0-0-0-visible-2.json and .log (gfx1201-only control)
- ar_oneshot_3rank_exact_host-devices-0-0-1-visible-0-2.json and .log (mixed two-physical-card control)
- ar_oneshot_3rank_exact_host-devices-0-1-2-visible-0-1-2.json and .log (**final 33/33**)
- ar_twoshot_3rank_exact_host-devices-0-1-2-visible-0-1-2.json and .log (**final 33/33**)

The previous negative-control receipts remain in BC_WORK_ROOT/ltx25-tp3-c07-proof/ (and the C07 evidence document). The runner is opt-in and timeout-bounded, records per-card/entry JSON and raw stdout/stderr, and returns nonzero for native/ROCm failure.

## Gate status and exact next slice

**MET:** three heterogeneous cards in one process; mixed gfx1100/gfx1201 libraries; native exact BF16 host-mailbox one-shot and weighted two-shot; shared-slot ownership across both libraries; 8/8/16 and 12/12/8 legal head ownership; in-/out-of-place outputs; 1, 4, 16 and 64 row probes; bit-exact independent numerical reference; native AQL launch through the real HIP-enabled 2.2 core.

**OPEN:** test that C05's *actual* fill + cast/insert + exact Q/K all-reduce + full-width rmsnorm + cast/extract graph executes in the native runtime on three cards, with model-backed full-width gains and TP1 reference; then test C06's *complete* sharded self-attention QKV, global Q/K RMS, local RoPE/attention/gate and exact output reduce+bias. Separate production startup/admission, DiT blocks, guidance and video decoding must remain disabled until parity and safety gates are met.

The direct-HIP SIGSEGV is a **harness-specific red result**, not a production regression. Its precise ROCm-level cause is not independently demonstrated, and should not be "fixed" by changing production collectives now that the actual AQL path passes.

**Worktree state:** branch agent/ltx25-tp3-main22; native lab modifications only. GPU lease released and no resident GPU jobs at close.
