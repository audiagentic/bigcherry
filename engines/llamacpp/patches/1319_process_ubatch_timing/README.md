# 1319_process_ubatch_timing

## Promotion record

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11474 / b9acf138, 2026-10-08). A diagnostic, promoted as a
neutral enabler: it prints only when its flag is set and changes nothing otherwise, so it can be used on the production
binary without a special build.

- Build and run: experiment build `b-metamem-mig-diag` (production + 1319 + 1320 + 1325 + 1346) compiled clean for
  gfx1100 / gfx1201 / gfx1030. Flash-Next production topology, ctx 245760, 38,673-token prompt (run `metamem-mig-diag`).
- Activation with `BIGCHERRY_SUBMIT_TIMING=1`: 297 `BIGCHERRY_SUBMIT_TIMING` lines; target per 512-token chunk: graph
  build 17.2 ms, set_inputs 4.3 ms, graph_compute 50.5 ms; draft context 1.9 / 3.2 / 17.5 ms
  (`tools/lab/flash-next/submit-timing-table.py`).
- Neutral: greedy text with the flag on equals the flag-off arm of the same binary (md5 fe307bdfb7e1), prefill 1167.5
  t/s with the timing on against 1106.4 on the cold flag-off load and ~1170 on production.
- It is one of the three diagnostics that located the MTP prompt cost (QFP31: 354 against 437 ms per chunk).

## Native llama.cpp comparison

Native llama.cpp has no counterpart: the patch only adds reporting behind its flag. With the flag unset the build behaves as without the patch.
