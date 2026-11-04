# RAD46: rows-kernel accuracy on the gfx1030 helper (2026-11-03)

Layout: MEN26 production layout, R9700 engine, two RX 7900 XTX helpers plus an RX 6900 XT (gfx1030) tail helper.
Plan item: RAD46. Source: the Codex session's notes on branch `codex/rad46` (unmerged), summarised here so the
outcome survives branch deletion. Numbers are the session's; nothing was re-measured for this note.

## Problem

With the 6900 XT tail helper serving about 2.3% of routes, the helper-KLD tail p99.9 fails its gate. The worst
positions are the same hard tokens in every arm (corpus document kld-docs-02, positions about 5000-5500); the tail
amplifies existing error. Control values: helper-off p99.9 2.79229, two XTX 3.19805. Gate: tail no more than +0.5
over control.

## Attempts

| Attempt | Result | Why it passed or failed |
|---|---|---|
| Rows kernel with float scales and unscaled half operands (candidate `383c17c1b`) | FAIL: tail p99.9 3.71049 (+0.918 over control; old tail 4.36970) | Lowers the tail but not enough. Stage replay on a 122-route capture: 3 gate and 5 up BF16 mismatches on rank 0, 5 and 6 on rank 1 against grouped GEMM. |
| Same run with `R3_MOE_WORKER_ROWS=0` (grouped GEMM) | PASS: tail p99.9 1.9001, all six limits | Localises the error to the rows path. A diagnostic, not a fix: throughput not compared. |
| `groupfold`: fold the raw 64-K groups and scales in GEMM order | Output hashes match grouped GEMM on both ranks; FAIL on speed: gate/up 4445 us vs 740 us, down 2721 vs 519 us | Matching the arithmetic costs far too many cross-lane operations. |
| `stageprobe`: canonical per-stage snapshots | Diagnostic only | Showed quantisation groups changed on each rank (6), none with identical gate/up inputs. |
| Route attribution (`rad46-attribution`): scheduler step, ticket and helper request records | 5318 captures joined through 131 tickets, none missing | Position 5458 used tail experts (layer 15 expert 280, layer 40 expert 491); 5459 used five tail routes. Correlation, not causation. |
| `outputlane`: output-owned batched rows kernel | PASS quality twice, byte-identical .rows files: tail p99.9 1.90005 (all six limits). FAIL throughput: pp1024 1928.32 to 1320.18 tok/s (-31.54%), tg512 75.92 to 83.20 (+9.59%) over four interleaved pairs | Exact like grouped GEMM, but large-lane tail gate/up rose 2282 to 6482 us per request. Violates the no-prefill-regression criterion. |
| `outputlane-rb16`: larger weight-reuse batch | Hashes match; layer-13 gate/up about 12 ms vs 2-3 ms for RB8 | Slower: reuse did not pay at this width. Abandoned. |
| `quaddpp` / `quadasm`: four-lane quad-DPP, ordered assembly fold (VGPRs 216 down to 120, no scratch) | Hashes match accurate output-lane on outlier and three 2048-row captures; gate/up 1084 us, down 494 us vs original rows 710/488 and output-lane 1567/502 | Best so far, still slower than the original rows kernel on gate/up. Exploratory timings only. |
| `pairdecode`, `pairdpp`, `quadadd` | No recorded result | Preserved by the session as unvalidated work in progress. |

## Conclusions

- Accuracy and speed trade against each other: every variant that matched grouped GEMM bit for bit was slower in
  prefill. The remaining target is matching the arithmetic with far fewer cross-lane operations.
- Production resolution (lead, 2026-11-04, branch `feat/rad46-accurate-rows`, commit `90ecda12f`): the kernel is chosen per stage by request rows. Output-lane RB8 at or below derived crossovers (gate/up 309 rows, down 922 rows), quad-DPP RB16 above. KLD on the 6900 XT tail passes; decode is at parity; 4k prefill costs about 5-10%; large gate/up is still about 1.4x slower than the old inaccurate kernel in isolation.
- Not yet measured: the row-owned `rowlane16`/`rowlane8` gate/up candidate (Codex report 3). Resume there only if large-prefill speed matters.
- The PR also needs `r3_rad46_rows_policy_test` declared in `component.toml` and `tools/tests/release/test_bump_gates.py` (CI requires both).
- Quality evidence: `artifacts/RAD46-diagnostic/capture-{attested-1,gemm}` and `artifacts/RAD46-outputlane/{capture-1,capture-2,throughput-ab}` inside the `codex/rad46*` worktrees. Copied to `runs/rad46-evidence/<variant>/` in the work root.
