"""HI14: real graph capture-lifecycle activation evidence.

HI14's offline validator (tools/bigcherry/multi_gpu_validate.py) has always
required explicit per-stage evidence for an "enabled graph" claim --
graph.capture_lifecycle = {capture_begin, capture_end, instantiate, replay},
each boolean True -- but nothing in the runtime has ever produced that
evidence. The 2026-08-20 PH01 completion scan and RV60 (2026-08-21/22, dual-
XTX corrected run) both independently found the same gap: real hardware runs
exist that plausibly exercise the graph path (server logs show graph reuse,
dispatch records show observations), but none of them carry the four actual
HIP graph API lifecycle events the validator's own schema demands. A
plausible-looking run is not evidence; "graphs enabled" and repeated server
traffic are not a substitute for observing capture_begin/capture_end/
instantiate/replay actually happen.

This patch instruments the four real HIP graph API call sites in
ggml_backend_cuda_graph_compute()/ggml_cuda_graph_evaluate_and_capture()
(confirmed against the real vendored ggml-cuda.cu, 2026-08-22):

  - cudaStreamBeginCapture (capture begins)
  - cudaStreamEndCapture   (capture ends)
  - cudaGraphInstantiate   (executable graph created from the captured graph)
  - cudaGraphLaunch        (graph replayed -- happens on EVERY graph-mode
                             forward pass once warmup completes, so this is
                             the one site that fires at high frequency and
                             is deliberately made emit-once, matching the
                             other three)

Each site is instrumented immediately after its CUDA_CHECK(...) call
succeeds (CUDA_CHECK aborts on failure, so reaching the marker means the
call actually returned success) with a once-per-process, opt-in-only
GGML_LOG_WARN marker: "BIGCHERRY_GRAPH_LIFECYCLE stage=<name>". Gated behind
BIGCHERRY_GRAPH_LIFECYCLE_TRACE so ordinary graph-mode inference (including
production replay) pays nothing beyond one getenv() per call site per
process -- the same pattern as patches/1205_rd12_paired_mmvq_dual_output.py's
and 1204_rd08_q6k_mmvq_vdr2.py's activation markers (once-per-process
std::atomic_flag + a log call, same BIGCHERRY_PATCH_TRACE-style opt-in
convention, here scoped to its own env var since this is a cross-cutting
runtime lifecycle signal, not one patch's own activation evidence).

HI90 (2026-08-23, real-hardware finding): a raw ggml GGML_LOG_INFO call is
NOT visible in llama-server's default logs. vendor/llama.cpp/common/log.cpp's
common_get_verbosity() maps GGML_LOG_LEVEL_INFO to LOG_LEVEL_TRACE (4), and
common_log_default_callback() drops anything above the server's verbosity
threshold (default 3) before it ever reaches a sink. Confirmed on real
hardware: a genuine graph-capturing dual-XTX run produced zero
BIGCHERRY_GRAPH_LIFECYCLE lines at default verbosity despite the server's
own "graphs reused = N" counters proving graphs were captured and replayed;
the identical run at `-lv 4` showed all four markers cleanly. GGML_LOG_WARN
is NOT filtered by this verbosity mechanism (only GGML_LOG_LEVEL_DEBUG is
gated in log.cpp's print()), so it is visible at any normal server
verbosity -- these markers aren't warnings semantically, but the existing
NCCL heterogeneous-arch guard (HI85, patches/1225) already established the
same "use WARN for something an operator must be able to see without special
flags" precedent for opt-in evidence markers in this codebase.

tools/bigcherry/graph_lifecycle_evidence.py (new, this same change) parses
these markers from a captured server log/stderr into the exact
multi_gpu_validate.py capture_lifecycle dict shape.
"""

import re

GROUP = "core"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

_CAPTURE_BEGIN_OLD = """        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));"""

_CAPTURE_BEGIN_NEW = """        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
        // bigcherry (HI14): capture-lifecycle activation evidence, not part
        // of the upstream change. See multi_gpu_validate.py's
        // capture_lifecycle contract -- real begin/end/instantiate/replay
        // evidence, not an inference from "graphs enabled".
        if (getenv("BIGCHERRY_GRAPH_LIFECYCLE_TRACE") != nullptr) {
            static std::atomic_flag bigcherry_capture_begin_logged = ATOMIC_FLAG_INIT;
            if (!bigcherry_capture_begin_logged.test_and_set(std::memory_order_relaxed)) {
                GGML_LOG_WARN("BIGCHERRY_GRAPH_LIFECYCLE stage=capture_begin\\n");
            }
        }"""

_CAPTURE_END_OLD = """            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));"""

_CAPTURE_END_NEW = """            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            // bigcherry (HI14): capture-lifecycle activation evidence -- see
            // the capture_begin site above for the full rationale.
            if (getenv("BIGCHERRY_GRAPH_LIFECYCLE_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_capture_end_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_capture_end_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_GRAPH_LIFECYCLE stage=capture_end\\n");
                }
            }"""

_INSTANTIATE_OLD = """            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));"""

_INSTANTIATE_NEW = """            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
            // bigcherry (HI14): capture-lifecycle activation evidence -- see
            // the capture_begin site above for the full rationale.
            if (getenv("BIGCHERRY_GRAPH_LIFECYCLE_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_instantiate_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_instantiate_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_GRAPH_LIFECYCLE stage=instantiate\\n");
                }
            }"""

_LAUNCH_OLD = """        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));"""

_LAUNCH_NEW = """        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        // bigcherry (HI14): capture-lifecycle activation evidence -- see the
        // capture_begin site above for the full rationale. Deliberately
        // emit-once: unlike the other three stages, this call happens on
        // EVERY graph-mode forward pass once warmup completes, so this is
        // the one site that fires at high frequency in normal operation.
        if (getenv("BIGCHERRY_GRAPH_LIFECYCLE_TRACE") != nullptr) {
            static std::atomic_flag bigcherry_replay_logged = ATOMIC_FLAG_INIT;
            if (!bigcherry_replay_logged.test_and_set(std::memory_order_relaxed)) {
                GGML_LOG_WARN("BIGCHERRY_GRAPH_LIFECYCLE stage=replay\\n");
            }
        }"""

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="real graph capture-lifecycle activation evidence for HI14's "
                "capture_begin/capture_end/instantiate/replay contract",
    edits=(
        Edit(
            id="hi14-capture-begin-marker",
            anchor=re.escape(_CAPTURE_BEGIN_OLD),
            mode="replace",
            rationale="cudaStreamBeginCapture succeeding is the real "
                      "capture_begin lifecycle event",
            text=_CAPTURE_BEGIN_NEW,
            guard=r"BIGCHERRY_GRAPH_LIFECYCLE stage=capture_begin",
        ),
        Edit(
            id="hi14-capture-end-marker",
            anchor=re.escape(_CAPTURE_END_OLD),
            mode="replace",
            rationale="cudaStreamEndCapture succeeding is the real "
                      "capture_end lifecycle event",
            text=_CAPTURE_END_NEW,
            guard=r"BIGCHERRY_GRAPH_LIFECYCLE stage=capture_end",
        ),
        Edit(
            id="hi14-instantiate-marker",
            anchor=re.escape(_INSTANTIATE_OLD),
            mode="replace",
            rationale="cudaGraphInstantiate succeeding is the real "
                      "instantiate lifecycle event",
            text=_INSTANTIATE_NEW,
            guard=r"BIGCHERRY_GRAPH_LIFECYCLE stage=instantiate",
        ),
        Edit(
            id="hi14-replay-marker",
            anchor=re.escape(_LAUNCH_OLD),
            mode="replace",
            rationale="cudaGraphLaunch succeeding is the real replay "
                      "lifecycle event",
            text=_LAUNCH_NEW,
            guard=r"BIGCHERRY_GRAPH_LIFECYCLE stage=replay",
        ),
    ),
)

PATCHES = [PATCH]
