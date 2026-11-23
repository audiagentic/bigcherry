# Unmerged candidates kept from retired worktrees (2026-11)

Performance candidates that never reached main are kept here as plain patches so the work is not lost when the worktree archives are pruned.
Everything else on those branches was either merged (by squash, so the commits differ) or was a deletion or move.

| Patch | Written | Idea | Why not merged | To revive |
| --- | --- | --- | --- | --- |
| `rad0038-b16i8-decode-scale-broadcast.patch` | 2026-11-19 | In the `rad0038_bf16_i8_kv` decode path, at tile 16 and a 32-wide scale group, four lanes of a wave read the same token's scale: one lane loads it and `__shfl` distributes the exact 16-bit value. | Marked "hardware A/B pending"; no speed or bit-identity result was produced. The patch targets the pre-2.2 engine source. | Re-anchor on the current pin, run the kbench A/B on a decode row, prove bit-identity, then promote through the patch lifecycle. |
| `moe-remote-gemv-capability-cache.patch` | 2026-11-20 | Cache the decode GEMV kernel's pre-launch capabilities per shape instead of recomputing them per launch (`gemv_geometry.h`, `moe_remote_gpu.h`). | No measurement of the per-launch cost it removes; main's `gemv_geometry.h` has since changed. | Measure the per-launch capability cost first; re-apply onto the current component only if it is a visible share of a decode step. |
| `vtp37-kpar-wave-per-token-tile.patch` | 2026-11-23 | Give each token tile of the gfx1100 K-parallel int8 tile (NT >= 2) its own wave (a block of KW x NT waves) so the tiles' math overlaps the weight stream; outputs bit-identical, registers drop to the one-tile level with zero spills | Measured and rejected: the cold effective bandwidth at 32 rows was 63-70% of the one-tile row (gate: 80%), and the 13312x5120 row got 5% slower (143-144 us against 137 us) with NT 3 slower too; small shapes mixed (-14% to +5%). Evidence: `docs/evidence/2026-11-23-vtp37-kpar-wave-per-tile/` | Only with a new idea for why the duplicate weight reads cost time (needs counters this stack does not give), or a variant that shares the weight fragment between the token-tile waves through LDS |

Apply with `git apply --3way` onto a worktree, expecting conflicts.
