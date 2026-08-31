"""HI155: size-adaptive internal/RCCL AllReduce provider dispatch.

1001_hip_internal_allreduce (validated) enables a low-latency internal
AllReduce on HIP that real dual-XTX hardware evidence shows is a large win
for decode (+17.33% TPS, MTP completion-bench; +6.88%, plain tg128) but a
severe regression for prefill (-32% to -34%, pp512/pp2048/pp4096
llama-bench) -- see patches/1001_hip_internal_allreduce/SUMMARY.md.
``GGML_CUDA_ALLREDUCE`` today selects exactly one provider once, for the
whole communicator's lifetime (a single stored ``try_allreduce`` function
pointer) -- there is no per-call, size-aware choice, so shipping either
provider alone regresses one of the two regimes.

Real HI155-1 telemetry (0830's ``reduction_bytes`` field, added for this
item) captured on real traffic shows a clean, 10x separation with zero
overlap: MTP decode reduction sizes range 20,480-1,044,480 bytes; pp2048
prefill is a flat 10,485,760 bytes. Notably, decode's max sits right at
allreduce.cu's own ``GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT`` (1,048,576 bytes,
the point where the internal implementation's own small-message chunked
kernel hands off to its large-message copy-engine strategy) -- decode never
reaches that copy-engine path in practice, so the entire prefill regression
is attributable to that one strategy, not to internal's small-message path
scaling up.

This patch adds a fourth ``GGML_CUDA_ALLREDUCE=hybrid`` provider:

* Init brings up BOTH RCCL and the internal pipeline independently (not the
  existing greedy nccl->internal->none chain, where each step's failure
  falls through to the next -- hybrid needs both alive simultaneously so a
  per-call choice is possible). Either can fail to initialize without being
  fatal to the other; if both fail, this degrades to the meta backend's
  generic butterfly exactly like the existing chains do.
* Per call, ``ggml_backend_cuda_comm_try_allreduce_hybrid`` computes
  ``ggml_nbytes(tensors[0])`` and compares it against the internal
  pipeline's OWN real copy-engine threshold (via the new
  ``ggml_cuda_ar_pipeline_copy_threshold`` accessor -- not a second,
  independently-configurable constant that could drift out of sync with
  ``GGML_CUDA_AR_COPY_THRESHOLD``). Below that threshold, it tries internal
  first; on failure (or if unavailable, or at/above the threshold), it
  falls through to RCCL; if neither is available, it returns false and lets
  the existing meta-backend fallback handle the call, same as every other
  provider chain here.
* ``comm_ctx->provider_name`` is set to whichever sub-provider actually
  serviced each call, right before invoking it -- 0830's existing telemetry
  seam reads this same field immediately after ``try_allreduce`` returns,
  so per-call attribution (``effective_provider``: "internal" or "rccl")
  falls out of the existing telemetry pipeline for free, no separate
  labeling edit needed.
* Init forces the internal pipeline to exact F32 (a new
  ``ggml_cuda_ar_pipeline_force_exact_f32`` setter, called once right after
  ``ggml_cuda_ar_pipeline_init`` succeeds) rather than trusting whatever
  ``GGML_CUDA_AR_BF16_THRESHOLD`` happens to be set to. That env var
  defaults to 1 (BF16 wire round-trip for every nonzero-size reduction);
  hybrid mode must never inherit that silently -- 1001's own validated
  performance result requires exact F32, and real hardware evidence in
  1001's SUMMARY.md shows BF16 wire compression is a net loss on this
  workload, not a neutral tradeoff. This was caught by HI155-4's own
  correctness gate: the first run (no override) failed the analytical F32
  bound on both sides of the threshold -- exactly the BF16-approximation
  signature, not a real defect -- confirming gpt-dev-agent's warning that
  hybrid must not depend on an external env combination the operator has
  to remember.

Deliberately NOT done in this slice (per gpt-dev-agent's explicit guidance,
dev-gpt-agent gateway session ses_5307d9c58ec645cb): raising the threshold
above the internal pipeline's own copy-engine boundary, or introducing a
second independent threshold knob -- both would risk routing hybrid mode
into exactly the code path the real prefill regression evidence implicates.
Also not yet done: splitting the internal pipeline's init so hybrid mode
skips paying for its large-message (32MB-class host/device staging) buffers
it will never route through below the copy threshold -- accepted for this
validation slice per gpt's guidance, with the resource cost (VRAM, pinned
host allocation, clean teardown) to be measured, not assumed, before wider
adoption.
"""

GROUP = "core"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

ALLREDUCE_CUH = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cuh",
    description="expose the internal pipeline's real copy-engine threshold "
                "so a caller can avoid ever selecting it for a size it "
                "would itself route through the slow large-message path",
    edits=(
        Edit(
            id="declare-copy-threshold-accessor",
            anchor=r"^bool ggml_cuda_ar_allreduce\($",
            rationale="alongside the other pipeline accessor declarations, "
                      "before the per-call allreduce entry point",
            mode="insert_before",
            text=(
                "// Real per-pipeline copy-engine threshold (bytes) -- the point where\n"
                "// the internal AllReduce switches from its small-message chunked kernel\n"
                "// to its large-message copy-engine strategy. A caller choosing whether\n"
                "// to route a given reduction through this pipeline (HI155's hybrid\n"
                "// dispatcher) needs this exact value, not a second, independently\n"
                "// configurable threshold that could drift out of sync with the pipeline's\n"
                "// own GGML_CUDA_AR_COPY_THRESHOLD.\n"
                "size_t ggml_cuda_ar_pipeline_copy_threshold(\n"
                "    const ggml_cuda_ar_pipeline * pipeline);\n\n"
                "// HI155: hybrid mode must never silently inherit whatever\n"
                "// GGML_CUDA_AR_BF16_THRESHOLD happens to be set to (default: 1, i.e. BF16\n"
                "// wire round-trip for every nonzero-size reduction) -- 1001's own validated\n"
                "// result requires exact F32, and real hardware evidence in\n"
                "// patches/1001_hip_internal_allreduce/SUMMARY.md shows BF16 wire\n"
                "// compression is a net loss on this workload, not a neutral tradeoff. This\n"
                "// forces an already-constructed pipeline to exact F32 regardless of the env\n"
                "// var, so hybrid's policy does not depend on an external env combination\n"
                "// the operator has to remember.\n"
                "void ggml_cuda_ar_pipeline_force_exact_f32(\n"
                "    ggml_cuda_ar_pipeline * pipeline);\n\n"
            ),
            guard=r"ggml_cuda_ar_pipeline_force_exact_f32\(\n    ggml_cuda_ar_pipeline \* pipeline\);",
        ),
    ),
)

ALLREDUCE_CU = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="implement the copy-engine-threshold accessor, both the "
                "real (HIP/CUDA) and MUSA-stub branches",
    edits=(
        Edit(
            id="implement-copy-threshold-accessor",
            anchor=r"    return ok;\n\}\n\n#else",
            rationale="right after ggml_cuda_ar_allreduce's closing brace "
                      "(its final statement, 'return ok;', makes the anchor "
                      "unique), before the MUSA-only stub branch",
            mode="replace",
            text=(
                "    return ok;\n}\n\n"
                "size_t ggml_cuda_ar_pipeline_copy_threshold(\n"
                "        const ggml_cuda_ar_pipeline * pipeline) {\n"
                "    return pipeline == nullptr ? 0 : pipeline->copy_threshold;\n"
                "}\n\n"
                "void ggml_cuda_ar_pipeline_force_exact_f32(\n"
                "        ggml_cuda_ar_pipeline * pipeline) {\n"
                "    if (pipeline != nullptr) {\n"
                "        pipeline->bf16_threshold = 0;\n"
                "    }\n"
                "}\n\n"
                "#else"
            ),
            guard=r"size_t ggml_cuda_ar_pipeline_copy_threshold\(\n        const ggml_cuda_ar_pipeline \* pipeline\) \{",
        ),
        Edit(
            id="implement-copy-threshold-accessor-musa-stub",
            anchor=r"^bool ggml_cuda_ar_allreduce\(ggml_cuda_ar_pipeline \*, ggml_backend_t \*, ggml_tensor \*\*\) \{\n    return false;\n\}$",
            rationale="MUSA never builds a real pipeline (pipeline_init "
                      "always returns nullptr there), so both accessor "
                      "stubs are no-ops/zero",
            mode="insert_after",
            text=(
                "\nsize_t ggml_cuda_ar_pipeline_copy_threshold(const ggml_cuda_ar_pipeline *) {\n"
                "    return 0;\n"
                "}\n"
                "void ggml_cuda_ar_pipeline_force_exact_f32(ggml_cuda_ar_pipeline *) {\n"
                "}"
            ),
            guard=r"void ggml_cuda_ar_pipeline_force_exact_f32\(ggml_cuda_ar_pipeline \*\) \{\n\}",
        ),
    ),
)

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="add GGML_CUDA_ALLREDUCE=hybrid: both providers alive, "
                "per-call byte-threshold dispatch with internal->rccl->meta "
                "fallback",
    edits=(
        Edit(
            id="hybrid-try-allreduce",
            anchor=(
                r"static bool ggml_backend_cuda_comm_try_allreduce_internal\(\n"
                r"        ggml_backend_cuda_comm_context \* comm_ctx, struct ggml_tensor \*\* tensors\) \{\n"
                r"    return ggml_backend_cuda_comm_allreduce_internal\(comm_ctx, tensors\);\n"
                r"\}"
            ),
            rationale="right after the plain internal try_allreduce "
                      "wrapper, before the butterfly stub",
            mode="insert_after",
            text=(
                "\n\n"
                "// HI155: both providers are alive simultaneously in hybrid mode (see\n"
                "// ggml_backend_cuda_comm_init_hybrid below). Per call, route reductions\n"
                "// below the internal pipeline's OWN copy-engine threshold through internal\n"
                "// (the regime real hardware evidence shows it wins -- decode); everything\n"
                "// else through RCCL (prefill, where internal's large-message path measured\n"
                "// -32% to -34% against RCCL on real hardware). Internal failure at small\n"
                "// sizes falls through to RCCL rather than straight to the meta fallback,\n"
                "// same as the ordinary single-provider chains do on their own init failure.\n"
                "static bool ggml_backend_cuda_comm_try_allreduce_hybrid(\n"
                "        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {\n"
                "    const size_t reduction_bytes = tensors != nullptr && tensors[0] != nullptr\n"
                "        ? ggml_nbytes(tensors[0]) : 0;\n"
                "    const size_t internal_threshold = comm_ctx->ar_pipeline != nullptr\n"
                "        ? ggml_cuda_ar_pipeline_copy_threshold(comm_ctx->ar_pipeline) : 0;\n"
                "    if (comm_ctx->ar_pipeline != nullptr && reduction_bytes < internal_threshold) {\n"
                "        comm_ctx->provider_name = \"internal\";\n"
                "        if (ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors)) {\n"
                "            return true;\n"
                "        }\n"
                "    }\n"
                "#ifdef GGML_USE_NCCL\n"
                "    if (!comm_ctx->comms.empty()) {\n"
                "        comm_ctx->provider_name = \"rccl\";\n"
                "        return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);\n"
                "    }\n"
                "#endif // GGML_USE_NCCL\n"
                "    return false;\n"
                "}"
            ),
            guard=r"ggml_backend_cuda_comm_try_allreduce_hybrid\(",
        ),
        Edit(
            id="hybrid-init",
            anchor=(
                r"static void ggml_backend_cuda_comm_init_nccl\(ggml_backend_cuda_comm_context \* ret\) \{"
            ),
            rationale="right before the existing (greedy, chained) NCCL "
                      "init function -- hybrid's init is independent of "
                      "that chain",
            mode="insert_before",
            text=(
                "// HI155: unlike comm_init_{nccl,internal}, this does not chain-fallback\n"
                "// into the other provider on failure -- hybrid mode needs both alive at\n"
                "// once so ggml_backend_cuda_comm_try_allreduce_hybrid can choose per call.\n"
                "// A failure in either is not fatal to the other; if both fail this\n"
                "// degrades to the meta backend's generic butterfly, exactly like the\n"
                "// existing chains do when their one provider fails.\n"
                "static void ggml_backend_cuda_comm_init_hybrid(ggml_backend_cuda_comm_context * ret) {\n"
                "    bool have_nccl = false;\n"
                "#ifdef GGML_USE_NCCL\n"
                "    const ggml_cuda_device_info & info = ggml_cuda_info();\n"
                "    if (info.device_count <= info.physical_device_count) {\n"
                "        const size_t n = ret->dev_ids.size();\n"
                "        ret->comms.resize(n);\n"
                "        ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());\n"
                "        if (rc == ncclSuccess) {\n"
                "            have_nccl = true;\n"
                "        } else {\n"
                "            ret->comms.clear();\n"
                "            GGML_LOG_WARN(\"hybrid: NCCL init failed (%s); hybrid dispatch will use \"\n"
                "                          \"internal only\\n\", ncclGetErrorString(rc));\n"
                "        }\n"
                "    } else {\n"
                "        GGML_LOG_WARN(\"hybrid: NCCL disabled (virtual devices in use); hybrid \"\n"
                "                      \"dispatch will use internal only\\n\");\n"
                "    }\n"
                "#endif // GGML_USE_NCCL\n"
                "    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());\n"
                "    const bool have_internal = ret->ar_pipeline != nullptr;\n"
                "    if (have_internal) {\n"
                "        // Never inherit GGML_CUDA_AR_BF16_THRESHOLD's default (1, BF16 for\n"
                "        // every nonzero reduction) -- hybrid's internal side must stay exact\n"
                "        // F32, matching 1001's own validated result.\n"
                "        ggml_cuda_ar_pipeline_force_exact_f32(ret->ar_pipeline);\n"
                "    }\n"
                "    if (!have_internal) {\n"
                "        (void) cudaGetLastError();\n"
                "        GGML_LOG_WARN(\"hybrid: internal AllReduce init failed (n_devices != 2?); \"\n"
                "                      \"hybrid dispatch will use %s only\\n\", have_nccl ? \"rccl\" : \"meta\");\n"
                "    }\n"
                "    if (have_internal || have_nccl) {\n"
                "        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_hybrid;\n"
                "        ret->provider_name = have_internal ? \"internal\" : \"rccl\";\n"
                "        return;\n"
                "    }\n"
                "    ggml_backend_cuda_comm_init_none(ret);\n"
                "}\n\n"
            ),
            guard=r"ggml_backend_cuda_comm_init_hybrid\(ggml_backend_cuda_comm_context \* ret\) \{",
        ),
        Edit(
            id="hybrid-env-selector",
            # csource.strip_noise blanks string-literal contents (not just
            # comments) to same-length whitespace before anchor matching, so
            # the quoted "none" is invisible to the anchor regex here -- \s+
            # matches the blanked span instead of the literal text.
            anchor=(
                r'        \} else if \(env_str ==\s+\) \{\n'
                r'            ggml_backend_cuda_comm_init_none\(ret\);\n'
                r'        \} else \{'
            ),
            rationale="add hybrid as a fourth GGML_CUDA_ALLREDUCE value, "
                      "beside the existing nccl/internal/none branches",
            mode="replace",
            text=(
                '        } else if (env_str == "none") {\n'
                '            ggml_backend_cuda_comm_init_none(ret);\n'
                '        } else if (env_str == "hybrid") {\n'
                '            ggml_backend_cuda_comm_init_hybrid(ret);\n'
                '        } else {'
            ),
            guard=r'else if \(env_str == "hybrid"\) \{',
        ),
    ),
)

PATCHES = [ALLREDUCE_CUH, ALLREDUCE_CU, CUDA]
