"""0860: explicit AllReduce provider/wire configuration seam.

Source ranges still required before the CLI edits can be authored without
inventing anchors/APIs:
- common/arg.cpp: exact backend-registry/proc lookup helper or the model-load
  boundary where a common_params-stored value can be applied.
- examples/llama-bench/llama-bench.cpp: exact -sm parser block (~750), cmd_params
  definition carrying split-mode (~parser state), and post-ggml_backend_load_all
  block (~2263).

The CUDA seam below is fully anchored from the supplied b11233 excerpts.
"""

GROUP = "core"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch


CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="replace GGML_CUDA_ALLREDUCE env selection with an explicit registered provider/wire configuration seam",
    edits=(
        Edit(
            id="allreduce-provider-config",
            anchor=r"^static void ggml_backend_cuda_comm_init_none\(ggml_backend_cuda_comm_context \* ret\) \{$",
            rationale="define the process-wide validated CLI configuration immediately before the stock provider init functions",
            mode="insert_before",
            text=(
                "struct ggml_backend_cuda_comm_config {\n"
                "    std::string provider = \"auto\";\n"
                "    std::string wire = \"native\";\n"
                "};\n\n"
                "static ggml_backend_cuda_comm_config g_ggml_backend_cuda_comm_config;\n\n"
                "static bool ggml_backend_comm_set_config(\n"
                "        const char * provider, const char * wire, char * err, size_t err_len) {\n"
                "    const std::string p = provider != nullptr ? provider : \"\";\n"
                "    const std::string w = wire != nullptr ? wire : \"\";\n"
                "    const bool known_provider = p == \"auto\" || p == \"ccl\" || p == \"host\" ||\n"
                "        p == \"adaptive\" || p == \"p2p\" || p == \"root3\" || p == \"butterfly\";\n"
                "    const bool known_wire = w == \"native\" || w == \"q8\";\n"
                "    if (!known_provider || !known_wire) {\n"
                "        if (err != nullptr && err_len != 0) {\n"
                "            snprintf(err, err_len, \"invalid allreduce provider/wire: %s/%s\", p.c_str(), w.c_str());\n"
                "        }\n"
                "        return false;\n"
                "    }\n"
                "    if (w == \"q8\" && p != \"p2p\") {\n"
                "        if (err != nullptr && err_len != 0) {\n"
                "            snprintf(err, err_len, \"--allreduce-wire q8 requires --allreduce p2p\");\n"
                "        }\n"
                "        return false;\n"
                "    }\n"
                "    // Extension providers are registered by patches that require 0860. Until\n"
                "    // those patches extend this dispatch, fail explicitly rather than silently\n"
                "    // selecting another provider.\n"
                "    if (p == \"adaptive\" || p == \"p2p\" || p == \"root3\") {\n"
                "        if (err != nullptr && err_len != 0) {\n"
                "            snprintf(err, err_len, \"allreduce provider not available in this build: %s\", p.c_str());\n"
                "        }\n"
                "        return false;\n"
                "    }\n"
                "    g_ggml_backend_cuda_comm_config.provider = p;\n"
                "    g_ggml_backend_cuda_comm_config.wire = w;\n"
                "    return true;\n"
                "}\n\n"
            ),
            guard=r"static bool ggml_backend_comm_set_config\(",
            expect_matches=1,
        ),
        Edit(
            id="allreduce-provider-init",
            anchor=(
                r"    const char \* env = getenv\(\"GGML_CUDA_ALLREDUCE\"\);\n"
                r"    if \(!env\) \{\n"
                r"        // Platform default: Linux uses NCCL, otherwise \(generally Windows\) internal\n"
                r"#if defined\(__linux__\)\n"
                r"        ggml_backend_cuda_comm_init_nccl\(ret\);\n"
                r"#else\n"
                r"        ggml_backend_cuda_comm_init_internal\(ret\);\n"
                r"#endif // defined\(__linux__\)\n"
                r"    \} else \{\n"
                r"        std::string env_str\(env\);\n"
                r"        if \(env_str == \"nccl\"\) \{ ggml_backend_cuda_comm_init_nccl\(ret\); \}\n"
                r"        else if \(env_str == \"internal\"\) \{ ggml_backend_cuda_comm_init_internal\(ret\); \}\n"
                r"        else if \(env_str == \"none\"\) \{ ggml_backend_cuda_comm_init_none\(ret\); \}\n"
                r"        else \{ GGML_LOG_WARN\(\"unknown GGML_CUDA_ALLREDUCE value: %s\\n\", env\); ggml_backend_cuda_comm_init_none\(ret\); \}\n"
                r"    \}"
            ),
            rationale="remove GGML_CUDA_ALLREDUCE and select only from the validated explicit configuration; auto preserves the stock platform default",
            mode="replace",
            text=(
                "    std::string provider = g_ggml_backend_cuda_comm_config.provider;\n"
                "    const std::string & wire = g_ggml_backend_cuda_comm_config.wire;\n"
                "    if (provider == \"auto\") {\n"
                "#if defined(__linux__)\n"
                "        provider = \"ccl\";\n"
                "#else\n"
                "        provider = \"host\";\n"
                "#endif\n"
                "    }\n"
                "    if (provider == \"ccl\") {\n"
                "        ggml_backend_cuda_comm_init_nccl(ret);\n"
                "    } else if (provider == \"host\") {\n"
                "        ggml_backend_cuda_comm_init_internal(ret);\n"
                "    } else if (provider == \"butterfly\") {\n"
                "        ggml_backend_cuda_comm_init_none(ret);\n"
                "    } else {\n"
                "        GGML_ABORT(\"allreduce provider reached init without implementation: %s\", provider.c_str());\n"
                "    }\n"
                "    if (getenv(\"BIGCHERRY_PATCH_TRACE\") != nullptr) {\n"
                "        GGML_LOG_INFO(\"BIGCHERRY_PATCH_HIT patch=0860_allreduce_provider_cli provider=%s wire=%s\\n\",\n"
                "            provider.c_str(), wire.c_str());\n"
                "    }"
            ),
            guard=r"BIGCHERRY_PATCH_HIT patch=0860_allreduce_provider_cli",
            expect_matches=1,
        ),
        Edit(
            id="allreduce-provider-proc",
            anchor=(
                r"    if \(strcmp\(name, \"ggml_backend_comm_init\"\) == 0\) \{\n"
                r"        return \(void \*\) ggml_backend_cuda_comm_init;\n"
                r"    \}"
            ),
            rationale="publish the configuration setter beside the existing communication proc-address seam",
            mode="insert_before",
            text=(
                "    if (strcmp(name, \"ggml_backend_comm_set_config\") == 0) {\n"
                "        return (void *) ggml_backend_comm_set_config;\n"
                "    }\n"
            ),
            guard=r'\"ggml_backend_comm_set_config\"',
            expect_matches=1,
        ),
    ),
)


PATCHES = (CUDA,)
