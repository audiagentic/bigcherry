# Radiance 1.3.0 remote-MoE prototype: recovered local source (13 November 2026)

This is an **immutable historical source archive**, not a managed engine patch, an active worktree, a performance result, or a request to apply these edits. It preserves the local-only work found at `${BC_WORK_ROOT}/engines/radiance-patched` before that checkout is considered for retirement.

## Provenance and contents

- Source repository: `https://codeberg.org/StillDeadcode/radiance.git`.
- Base: `89cee7ce414486c717aa2085c99e1f522e4c0ea6` (Radiance `v1.3.0`, detached `HEAD`).
- Captured on 2026-11-13 from a **dirty local checkout**, without changing it. Source edit dates and original author are not established by this snapshot.
- `tracked-changes.patch.gz` contains a byte-preserving gzip of the exact `git diff --binary` of eight modified tracked files, relative to that base.
- `abi/rad_moe_remote.h` and `core/place/moe_remote.cpp` are copies of two *untracked* prototype source files not included in `git diff`.
- `historical-build-output.txt` is a copied local build log: it reaches the `radiance` link target, including `moe_remote.cpp.o`. The separate local configure log reports HIP `gfx1201`, `RelWithDebInfo` and FFmpeg 6; it has not been packaged because it embeds a machine-local build path.
- The original dirty checkout also has untracked `build.log` and `configure.log`, treated as run artifacts rather than source edits. No benchmarks, GPU acceptance, correctness, or current-pin build were performed for this archive.

## What the prototype explored

The core change offers a callback-based offload for host-resident routed experts: annotate residency slots, determine eligible routes after top-k selection, send route/activation data to a helper, mask offloaded routes from local scatter, and join a BF16 partial before the collective. Its prototype uses `R3_MOE_REMOTE_SO`, `R3_MOE_REMOTE_MIN_ROWS`, dynamic hook registration, host route copies and explicit synchronization. The remaining edits include a mover-stream teardown drain and `RAD_MTP_FULL_HEAD`.

**Do not cherry-pick or reintroduce this code** into current Radiance. The relevant current packages are:

| Original change | Managed successor | Disposition |
| --- | --- | --- |
| Mover drain before arena destruction | [rad0001](../../../engines/radiance/patches/rad0001_drain_mover_before_arena_free/SUMMARY.md) | Superseded by upstream teardown handling from v1.7.0 |
| Remote whole-expert MoE handoff and routing | [rad0002](../../../engines/radiance/patches/rad0002_moe_remote_handoff/SUMMARY.md) | Redesigned, managed and profile-qualified for v1.14.0; part of serving-core |
| MTP trunk-head opt-in | [rad0003](../../../engines/radiance/patches/rad0003_mtp_full_head_switch/SUMMARY.md) | Preserved as an untested opt-in experiment |

The production remote-MoE design uses a distinct `bc_moe_remote_*` extension point and mailbox lifecycle, and now includes newer placement/fused-gather handling. The old prototype's synchronous route transfer, external hook loading and `host_pool` flags are **not evidence** that current code needs the same design. Its build log is not a qualification record.

## Reproducibility and retention

To inspect the old tracked edits, decompress `tracked-changes.patch.gz` and apply its contents only in a disposable checkout of the exact Radiance base above, then add the two untracked files to their recorded relative paths. The `patch` file intentionally omits those untracked files because Git does not include them in `git diff`.

`SHA256SUMS` binds the captured source, log and this explanation. After the archive is reviewed and merged, the original dirty checkout may be reconciled and removed separately. This document does **not** authorize deletion.
