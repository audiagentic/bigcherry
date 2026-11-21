# RUM11 — plain E4M3 page-16 fused KV store, Radiance 2.2

**Evidence captured:** 2026-11-22. **Installed release:** `bc-radiance-1170.37.0` (Radiance 2.2.0, ROCm 7.2.4). This closes the *technical validation* of the previously implemented RUM11 page-16 path. It does not modify the kernel, add dispatch fallbacks, or duplicate the managed patch. The authoritative plan/ledger transition must be performed through Planning/Ledger MCP.

## Existing implementation and dispatch contract

- Owning package: `engines/radiance/patches/rad0049_qwen4exp_e4m3_kv/patch.py`, already in `main` and marked `validated`; related shared KV geometry packages remain unchanged.
- The row `qk_norm_rope_gate_kv_store_h256_page16_fp8` uses the existing `r4d_rad_qk_norm_rope_gate_kv_store` launcher; its constraints are BF16 Q, head dimension 256, page size 16, E4M3 KV. It does **not** pin local query/KV head counts or rotary width.
- The managed patch already inserts two cases in the upstream `r4d_selftest.cpp`: 6 query / 1 KV head and 12 query / 2 KV heads. These exercise the rank-local TP3 shapes without adding a second test harness or editing `vendor/`.

## GPU selftest acceptance

| GPU architecture | Plugin | Local heads (Q/KV) | Result |
| --- | --- | --- | --- |
| gfx1100 | Released `libr3.so`, with the existing `r3_selftest` rewrite | 6/1 and 12/2 | PASS — byte-identical |
| gfx1201 | Released `libr4d.so`, existing `r4d_selftest` | 6/1 and 12/2 | PASS — byte-identical |

Each case compares the fused store with **`qk_norm_rope_gate` followed by `kv_store_fp8`**, checks the Q and rotated-K outputs and the cache bytes, and reports byte equality. Both architecture runs returned exit code 0 under an exclusive GPU lease. The gfx1100 selftest was compiled with the existing `engines/radiance/components/libr3` CMake target (a selftest-only build; the installed plugin was not modified). An initial gfx1201-only runner attempted on gfx1100 skipped the architecture; it was replaced by the proper gfx1100 runner and is **not** counted as a pass.

Raw results: `selftest-gfx1100.txt`, `selftest-gfx1201.txt`.

## TP3 release smoke and missing-band check

`smoke-summary.json` checks the installed release's six Qwen 27B configurations, including all four E4M3/FP8 KV variants: `qwen27b-2xtx`, `qwen27b-tp3`, `qwen27b-tp3-capacity`, `qwen27b-tp3-long`. The remaining two are `-long-bf16` and `-long-bf16i8`.

For **all six**: the captured serve-status snapshot shows healthy/running, the canary was passed and returned nonempty text, stop returned `stopped`, and the engine log had **zero** matches for `no kernel`, `unresolved`, `resolving to no kernel`, or `unsupported`. This reuses the existing full 16/16 release-roster acceptance evidence in `docs/evidence/radiance-latest-bump/RUM09-GATE-E.md`, cross-checked against its per-config records.

Separately, the earlier Radiance 2.2 `qwen27b-tp3-int8-556` W8A8 candidate (not shipped under that name in this release) exited 0, produced the canary, stopped cleanly and had zero missing-kernel markers. That preserves the specific reproducer named in RUM11 without claiming it is a shipped configuration.

## Quality evidence reused from RUM06

`quality-summary.json` records **existing RUM06 measurements from 2026-11-21**, not a fresh KLD sweep. Each used the four-document, 8,240-position reference corpus, the BF16 reference model, FP8 KV, and speculative-token depth **0** to isolate quality.

| Model and TP local layer split | KLD mean | Top-1 agreement | Perplexity ratio |
| --- | ---: | ---: | ---: |
| IQ8R 4/4/8 | 0.00122887 | 97.512% | 1.000539 |
| IQ8R 6/6/4 | 0.00128228 | 97.269% | 1.000491 |
| Earlier W8A8 5/5/6 | 0.00139157 | 96.772% | 0.999773 |

The installed `qwen27b-tp3` uses the **same IQ8R container, 6/6/4 ownership and FP8 KV** as the second measurement. Its normal serving configuration enables speculative-token depth 3, rather than the measurement's 0. The 4/4/8 IQ8R variant was measured for comparison, but is not the installed default split. The W8A8 run is a historical quality baseline, not the installed release model. These measurements are beneath the existing 0.0027 mean numerical bar cited in VTP43; the maximum per-document p99.9 is below 1. Their provenance and exact precision are retained in `quality-summary.json`. No new KLD claim is made for the installed release binary.

## Patch workflow verification

`patch-validation.txt` records repository `patch-lint` **exit 0** (warnings for already-present patch metadata) and the focused `test_radiance_rad0049.py` plus `test_radiance_kv_row_helper.py` results: **5 passed, 1 skipped, 2 subtests passed**, exit 0, with the pinned Radiance checkout made available. The passing exact-source test covers managed patch application, idempotent reapplication, generated row/constraint declarations, and host registry compilation. The skipped optional test requires an explicitly supplied frozen pre-RUM11 composed source, which was not supplied; the released composed source and binaries are independently exercised by the device selftests above.

No additional kernel implementation was required; the owning patch, plugin ABI and shipped runtime remain unchanged. The missing Planning/Ledger MCP interfaces in this execution session prevent recording the prescribed ledger event and changing RUM11's authoritative state; do not interpret this evidence file as an MCP state transition.
