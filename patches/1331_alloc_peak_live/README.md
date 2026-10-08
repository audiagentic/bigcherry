# 1331_alloc_peak_live

## Promotion record

Promotion record (BPB01 lightweight promotion tier, pin b11474, Brutus 2026-10-08). Diagnostic-only neutral enabler; it remains off unless its environment variable is set.

build deploy-v6-alloc-peak compiled clean; with BIGCHERRY_ALLOC_PEAK=1 BIGCHERRY_ALLOC_TOP=1 the server log has 1298 ALLOC_PEAK and 1106 ALLOC_TOP lines; greedy text identical to the arm without them (md5 a34a34c1f48d both), 0 error lines.

## Native llama.cpp comparison

Native llama.cpp has no counterpart for this BigCherry diagnostic. With its flag unset the patch does not emit the diagnostic path.
