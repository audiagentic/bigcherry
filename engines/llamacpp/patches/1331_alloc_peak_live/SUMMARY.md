# 1331_alloc_peak_live

**Status:** validated
**Plan item:** QFP17

## What it does

With `BIGCHERRY_ALLOC_PEAK=N`, the graph allocator tracks live bytes per compute buffer and, at each buffer's peak,
logs the node being allocated, the N largest tensors live at that moment and live bytes per op
(`BIGCHERRY_ALLOC_PEAK ...` lines at sched_reserve). This is what sets the compute buffer size; 1329 only ranks
the largest tensors overall, which misled 1330 (removing a large short-lived mask left the ub1024 buffer unchanged).
Fragmentation is not modelled; compare with the `compute buffer size` line. No behaviour change when unset.

## PA44-B ownership review

No definition is moved. The 121-line allocator diagnostics block is a `.c`-file detail: it depends directly
on private `ggml-alloc.c` types/state (`hash_node`, gallocr hash tables, bufts and dynamic allocator internals).
Moving it into a header would manufacture a C include seam solely to hide file-private implementation detail, which
is the explicit exception for this slice. It remains immediately before `ggml_gallocr_allocate_node`; all hooks,
state and env docs are unchanged.

## Identity proof

Refactor delta is zero: 0 definition lines moved and 0 include lines added. The patch output is byte-for-byte
unchanged by this commit; the retained helper block is 121 physical lines / 115 non-blank
whitespace-stripped lines (115/115 preserved).
