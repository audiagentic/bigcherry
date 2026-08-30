# 1231_hi14_graph_capture_lifecycle_evidence: Real graph capture-lifecycle activation evidence (HI14)

**Status:** untested
**Group:** core
**Plan item:** HI14

## What it does

Instruments the four real HIP graph API call sites (cudaStreamBeginCapture/EndCapture/GraphInstantiate/GraphLaunch) with a once-per-process, opt-in GGML_LOG_WARN marker (BIGCHERRY_GRAPH_LIFECYCLE_TRACE), and adds a parser (graph_lifecycle_evidence.py) that turns captured markers into the validator's capture_lifecycle schema.

## Why

HI14's offline validator has always required explicit per-stage lifecycle evidence for an 'enabled graph' claim, but nothing in the runtime produced it; real hardware runs that plausibly exercise the graph path are not the same as observing the four actual API events happen. Uses GGML_LOG_WARN rather than INFO because INFO is filtered out of llama-server's default log verbosity (confirmed on real hardware).

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI14), following the same opt-in marker convention established by HI85 (patch 1225).
