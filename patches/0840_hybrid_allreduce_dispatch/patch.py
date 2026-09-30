"""0840: size-adaptive internal/RCCL AllReduce dispatch.

Adaptive initializes RCCL and the stock internal host pipeline together. Calls
below the internal pipeline's copy threshold prefer host; calls at/above it
prefer RCCL, with the surviving provider used as fallback if the other is
unavailable. The adaptive host path deliberately leaves the internal
pipeline's wire policy untouched, so GGML_CUDA_AR_WIRE/1272 and pristine
wire selection behave exactly as they do for the plain host provider.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

ALLREDUCE_CUH = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cuh",
    description="expose the internal pipeline copy threshold to adaptive dispatch",
    edits=(
        Edit(
            id="declare-copy-threshold-accessor",
            anchor=r"^bool ggml_cuda_ar_allreduce\($",
            rationale="declare the threshold accessor beside the pipeline allreduce API",
            mode="insert_before",
            text=(
                "size_t ggml_cuda_ar_pipeline_copy_threshold(\n"
                "    const ggml_cuda_ar_pipeline * pipeline);\n\n"
            ),
            guard=r"ggml_cuda_ar_pipeline_copy_threshold\(\n    const ggml_cuda_ar_pipeline \* pipeline\);",
            expect_matches=1,
        ),
    ),
)

ALLREDUCE_CU = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="implement the adaptive copy-threshold accessor and MUSA stub",
    edits=(
        Edit(
            id="implement-copy-threshold-accessor",
            anchor=r"    return ok;\n\}\n\n#else",
            rationale="insert after the real ggml_cuda_ar_allreduce implementation",
            mode="replace",
            text=(
                "    return ok;\n}\n\n"
                "size_t ggml_cuda_ar_pipeline_copy_threshold(\n"
                "        const ggml_cuda_ar_pipeline * pipeline) {\n"
                "    return pipeline == nullptr ? 0 : pipeline->copy_threshold;\n"
                "}\n\n"
                "#else"
            ),
            guard=r"size_t ggml_cuda_ar_pipeline_copy_threshold\(\n        const ggml_cuda_ar_pipeline \* pipeline\) \{",
            expect_matches=1,
        ),
        Edit(
            id="implement-copy-threshold-accessor-musa-stub",
            anchor=r"^bool ggml_cuda_ar_allreduce\(ggml_cuda_ar_pipeline \*, ggml_backend_t \*, ggml_tensor \*\*\) \{\n    return false;\n\}$",
            rationale="MUSA never constructs a real internal pipeline",
            mode="insert_after",
            text=(
                "\nsize_t ggml_cuda_ar_pipeline_copy_threshold(const ggml_cuda_ar_pipeline *) {\n"
                "    return 0;\n"
                "}"
            ),
            guard=r"size_t ggml_cuda_ar_pipeline_copy_threshold\(const ggml_cuda_ar_pipeline \*\) \{\n    return 0;\n\}",
            expect_matches=1,
        ),
    ),
)

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="add adaptive provider with per-call host/RCCL dispatch",
    edits=(
        Edit(
            id="hybrid-try-allreduce",
            anchor=(
                r"static bool ggml_backend_cuda_comm_try_allreduce_internal\(\n"
                r"        ggml_backend_cuda_comm_context \* comm_ctx, struct ggml_tensor \*\* tensors\) \{\n"
                r"    return ggml_backend_cuda_comm_allreduce_internal\(comm_ctx, tensors\);\n"
                r"\}"
            ),
            rationale="insert the adaptive dispatcher beside the plain internal wrapper",
            mode="insert_after",
            text=(
                "\n\n"
                "static bool ggml_backend_cuda_comm_try_allreduce_hybrid(\n"
                "        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {\n"
                "    const size_t reduction_bytes = tensors != nullptr && tensors[0] != nullptr\n"
                "        ? ggml_nbytes(tensors[0]) : 0;\n"
                "    const bool have_internal = comm_ctx->ar_pipeline != nullptr;\n"
                "    const size_t internal_threshold = have_internal\n"
                "        ? ggml_cuda_ar_pipeline_copy_threshold(comm_ctx->ar_pipeline) : 0;\n"
                "    const bool below_copy_threshold = internal_threshold == 0 || reduction_bytes < internal_threshold;\n"
                "#ifdef GGML_USE_NCCL\n"
                "    const bool have_rccl = !comm_ctx->comms.empty();\n"
                "#else\n"
                "    const bool have_rccl = false;\n"
                "#endif\n"
                "    const bool prefer_internal = have_internal && (below_copy_threshold || !have_rccl);\n"
                "    if (prefer_internal) {\n"
                "        comm_ctx->provider_name = \"internal\";\n"
                "        if (ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors)) {\n"
                "            return true;\n"
                "        }\n"
                "    }\n"
                "#ifdef GGML_USE_NCCL\n"
                "    if (have_rccl) {\n"
                "        comm_ctx->provider_name = \"rccl\";\n"
                "        return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);\n"
                "    }\n"
                "#endif // GGML_USE_NCCL\n"
                "    if (have_internal && !prefer_internal) {\n"
                "        comm_ctx->provider_name = \"internal\";\n"
                "        return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);\n"
                "    }\n"
                "    return false;\n"
                "}"
            ),
            guard=r"ggml_backend_cuda_comm_try_allreduce_hybrid\(",
            expect_matches=1,
        ),
        Edit(
            id="hybrid-init",
            anchor=r"static void ggml_backend_cuda_comm_init_nccl\(ggml_backend_cuda_comm_context \* ret\) \{",
            rationale="adaptive initialization is independent of the stock greedy provider chain",
            mode="insert_before",
            text=(
                "static void ggml_backend_cuda_comm_init_hybrid(ggml_backend_cuda_comm_context * ret) {\n"
                "    bool have_nccl = false;\n"
                "#ifdef GGML_USE_NCCL\n"
                "    const ggml_cuda_device_info & info = ggml_cuda_info();\n"
                "    if (info.device_count <= info.physical_device_count &&\n"
                "            ggml_backend_cuda_comm_rccl_admission_ok(ret->dev_ids.data(), ret->dev_ids.size())) {\n"
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
                "    } else if (info.device_count > info.physical_device_count) {\n"
                "        GGML_LOG_WARN(\"hybrid: NCCL disabled (virtual devices in use); hybrid \"\n"
                "                      \"dispatch will use internal only\\n\");\n"
                "    } else {\n"
                "        GGML_LOG_WARN(\"hybrid: NCCL disabled (RCCL admission check failed); hybrid \"\n"
                "                      \"dispatch will use internal only\\n\");\n"
                "    }\n"
                "#endif // GGML_USE_NCCL\n"
                "    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());\n"
                "    const bool have_internal = ret->ar_pipeline != nullptr;\n"
                "    // Do not mutate the pipeline's BF16 threshold or wire state here. The\n"
                "    // adaptive host side must behave exactly like the plain internal provider,\n"
                "    // including 1272's GGML_CUDA_AR_WIRE codec selection when composed.\n"
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
            expect_matches=1,
        ),
        Edit(
            id="gp03-fix-explicit-rccl-plan-telemetry",
            anchor=(
                r"    if \(strcmp\(plan, [^\n]*\) == 0\) \{\n"
                r"#ifdef GGML_USE_NCCL\n"
                r"        if \(comm_ctx->comms\.size\(\) == comm_ctx->backends\.size\(\)\) \{\n"
                r"            return ggml_backend_cuda_comm_allreduce_nccl\(comm_ctx, tensors\);\n"
                r"        \}\n"
                r"#endif\n"
                r"    \}\n"
                r"    return false;"
            ),
            rationale="record the provider actually used by the explicit RCCL plan",
            mode="replace",
            text=(
                "    if (strcmp(plan, \"rccl\") == 0) {\n"
                "#ifdef GGML_USE_NCCL\n"
                "        if (comm_ctx->comms.size() == comm_ctx->backends.size()) {\n"
                "            comm_ctx->provider_name = \"rccl\";\n"
                "            return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);\n"
                "        }\n"
                "#endif\n"
                "    }\n"
                "    return false;"
            ),
            guard=r"comm_ctx->provider_name = \"rccl\";\n            return ggml_backend_cuda_comm_allreduce_nccl",
            expect_matches=1,
        ),
    ),
)

CUDA_PROVIDER = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    language="none",
    description="register adaptive on 0860's explicit provider seam",
    edits=(
        Edit(
            id="adaptive-provider-available",
            anchor=r'    if \(p == \"adaptive\" \|\| p == \"p2p\" \|\| p == \"root3\"\) \{\n',
            rationale="0840 provides adaptive, so remove only adaptive from 0860's unavailable list",
            mode="replace",
            text='    if (p == "p2p" || p == "root3") {\n',
            guard=r'    if \(p == \"p2p\" \|\| p == \"root3\"\) \{\n',
            expect_matches=1,
        ),
        Edit(
            id="adaptive-provider-init",
            anchor=(
                r'    \} else if \(provider == \"butterfly\"\) \{\n'
                r'        ggml_backend_cuda_comm_init_none\(ret\);\n'
                r'    \} else \{\n'
            ),
            rationale="select adaptive beside 0860's implemented providers",
            mode="replace",
            text=(
                '    } else if (provider == "butterfly") {\n'
                '        ggml_backend_cuda_comm_init_none(ret);\n'
                '    } else if (provider == "adaptive") {\n'
                '        ggml_backend_cuda_comm_init_hybrid(ret);\n'
                '    } else {\n'
            ),
            guard=r'provider == \"adaptive\"\) \{\n        ggml_backend_cuda_comm_init_hybrid',
            expect_matches=1,
        ),
    ),
)

PATCHES = [ALLREDUCE_CUH, ALLREDUCE_CU, CUDA, CUDA_PROVIDER]
