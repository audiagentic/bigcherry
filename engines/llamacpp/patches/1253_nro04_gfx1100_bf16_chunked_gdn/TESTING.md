# Testing — 1253 PNRO04 BF16 chunked GDN (2026-10-15 reconciliation)

## Already evidenced, not rerun

`patch.toml` is validated. `evidence/validation.json` contains older b11126 campaigns: gfx1100 pp512 +9.31884% (CI95 low 8.4784%) and gfx1201 +10.99672% (CI95 low 10.8858%), near-neutral tg128 controls; subject 46/46 and control 36/36 GATED_DELTA_NET backend-reference cases. Do not claim current-pin or tensor-split gain. 1254 MTP prefix/tail is a separate untested package.

## Cheap offline gate (not yet run as a repository test)

`PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1253_nro04_gfx1100_bf16_chunked_gdn`; patch-lint and package tests with the engine-explicit MEN10 CLI as appropriate to the current branch. Compose 1253 **without** conflicting 1221; verify `1254` requires 1253. Host selector fixture: supported ratios 1/2/3/4/6/8, unsupported 5/7/16 must choose sequential **before** the BF16 launcher, invalid H/H_k fail closed; gfx1100/gfx1201 positive, gfx1030 negative, S_v!=128, KDA, K>1, n_tokens<=1 and env-off negative. A currently unguarded unsupported ratio reaches `GGML_ABORT` in `launch_gdn_bf16_kkt` switch. Do not call this a reproduced GPU failure.

Host-only discriminator actually executed outside the repository: 20/20 selector tests; source-shape checks 13/13. No HIP build, backend test suite or new GPU benchmark executed.

## Required next hardware gate

On the exact pinned composition and one physical gfx1100 then gfx1201: test `GATED_DELTA_NET` K=1 with n_tokens=63/64/65/128/512/1024/2048, n_seqs=1/2, GQA ratios, gate extremes, beta extremes, padding and noncontiguous-negative controls. Compare F32 output **and recurrent state** with stock CPU/backend reference, full logits/KLD, greedy, multi-request same-process, graph capture/replay and scratch lifetime. Use `BIGCHERRY_PATCH_TRACE=1` positive/negative markers **and** kernel dispatch traces. No-P2P cross-device borrowing. A KKT or scan failure must not leave a falsely successful result.

For performance, existing contract: 4 independent sessions/architecture, >=10 ABBA pairs/session, pp512 positive CI95-low >=3% E2E and tg128 <=1% regression. Repeat on b11552 only if composition passes; compare post-pin upstream #29353 without assuming its RDNA3.5 HIP gate selects this fleet. MTP K>1 snapshot/rollback testing belongs to 1254/PNRO05, not 1253.
