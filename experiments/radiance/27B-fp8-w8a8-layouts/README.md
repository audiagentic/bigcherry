# 27B FP8/W8A8 and RDNA2-tail layout experiments

Archived from `exp/27b-fp8` (tip `da541e20b`, base `619590728`), three commits dated 2026-11-04. No runtime default is changed.

## What and why

Six opt-in Qwen3.8-27B two-XTX layouts: FP8 with DFlash2; W8A8 without a drafter; W8A8+DFlash2 at 8,192 × 6, 131,072 × 2, 262,144 × 1 and 262,144 × 5. Two more opt-in Flash-Next layouts reserve 4 or 8 GiB for an RDNA2 tail helper. These explore quantisation, drafter use, context and tail VRAM budgets.

## Outcome and why stopped

**(c): three branch-only layout commits.** Their 77 added lines have been copied verbatim into `engines/radiance/layouts.toml` rather than preserved as a duplicate patch. All eight names are new and opt-in; existing layouts and launcher defaults are unchanged. There is no quality or throughput measurement attached to these *exact* variants. The reason work stopped is **unknown**. Retention does not qualify these layouts for production.

## Evidence and resumption

- Commits `0bf6073a7` (initial variants), `5b1aab194` (262,144 × 5), `da541e20b` (4/8 GiB tails).
- `docs/reference/engines/radiance/models/qwen3.8-27b.md` and `docs/planning/active/patching-radiance-rdna3/RAD47.md` cover other measured 27B configurations, **not** these entries.
- To resume, verify model availability and budget fit, explicitly choose a named experimental layout and run correctness, KLD and throughput gates.
