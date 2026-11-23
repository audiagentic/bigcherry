# Unmerged candidates kept from retired worktrees (2026-11)

Two performance candidates were found on retired worktree branches that never reached main. Neither was ever measured, so
neither was promoted; both are kept here as plain patches so the work is not lost when the worktree archives are pruned.
Everything else on those branches was either merged (by squash, so the commits differ) or was a deletion or move.

| Patch | Written | Idea | Why not merged | To revive |
| --- | --- | --- | --- | --- |
| `rad0038-b16i8-decode-scale-broadcast.patch` | 2026-11-19 | In the `rad0038_bf16_i8_kv` decode path, at tile 16 and a 32-wide scale group, four lanes of a wave read the same token's scale: one lane loads it and `__shfl` distributes the exact 16-bit value. | Marked "hardware A/B pending"; no speed or bit-identity result was produced. The patch targets the pre-2.2 engine source. | Re-anchor on the current pin, run the kbench A/B on a decode row, prove bit-identity, then promote through the patch lifecycle. |
| `moe-remote-gemv-capability-cache.patch` | 2026-11-20 | Cache the decode GEMV kernel's pre-launch capabilities per shape instead of recomputing them per launch (`gemv_geometry.h`, `moe_remote_gpu.h`). | No measurement of the per-launch cost it removes; main's `gemv_geometry.h` has since changed. | Measure the per-launch capability cost first; re-apply onto the current component only if it is a visible share of a decode step. |

Apply with `git apply --3way` onto a worktree, expecting conflicts.
