"""0860: explicit AllReduce provider/wire configuration seam."""

GROUP = "core"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="replace GGML_CUDA_ALLREDUCE env selection with an explicit registered provider/wire configuration seam",
    language="none",
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
                r"        if \(env_str == \"nccl\"\) \{\n"
                r"            ggml_backend_cuda_comm_init_nccl\(ret\);\n"
                r"        \} else if \(env_str == \"internal\"\) \{\n"
                r"            ggml_backend_cuda_comm_init_internal\(ret\);\n"
                r"        \} else if \(env_str == \"none\"\) \{\n"
                r"            ggml_backend_cuda_comm_init_none\(ret\);\n"
                r"        \} else \{\n"
                r"            GGML_LOG_WARN\(\"unknown GGML_CUDA_ALLREDUCE value: %s\\n\", env\);\n"
                r"            ggml_backend_cuda_comm_init_none\(ret\);\n"
                r"        \}\n"
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
                r"        return \(void \*\)ggml_backend_cuda_comm_init;\n"
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

ARG_CPP = FilePatch(
    path="common/arg.cpp",
    description="add explicit AllReduce provider/wire CLI options and forward them through the backend registry",
    language="none",
    edits=(
        Edit(
            id="allreduce-config-helper",
            anchor=r"^static void add_rpc_devices\(const std::string & servers\) \{$",
            rationale="place the process-wide CLI-to-backend configuration bridge beside the existing backend-registry helper",
            mode="insert_before",
            text=(
                "static std::string common_allreduce_provider;\n"
                "static std::string common_allreduce_wire;\n\n"
                "// Option callbacks only record values; applied once after all args/env are parsed, so option order is irrelevant.\n"
                "static void common_apply_allreduce_config() {\n"
                "    if (common_allreduce_provider.empty() && common_allreduce_wire.empty()) {\n        return;\n    }\n\n"
                "    const std::string effective_provider = common_allreduce_provider.empty() ? \"auto\" : common_allreduce_provider;\n"
                "    const std::string effective_wire = common_allreduce_wire.empty() ? \"native\" : common_allreduce_wire;\n\n"
                "    ggml_backend_load_all();\n"
                "    bool found = false;\n"
                "    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {\n"
                "        ggml_backend_reg_t reg = ggml_backend_reg_get(i);\n"
                "        auto set_config = (bool (*)(const char *, const char *, char *, size_t))\n"
                "            ggml_backend_reg_get_proc_address(reg, \"ggml_backend_comm_set_config\");\n"
                "        if (set_config == nullptr) {\n            continue;\n        }\n"
                "        found = true;\n        char err[256] = {};\n"
                "        if (!set_config(effective_provider.c_str(), effective_wire.c_str(), err, sizeof(err))) {\n"
                "            throw std::invalid_argument(err);\n        }\n    }\n\n"
                "    if (!found && (effective_provider != \"auto\" || effective_wire != \"native\")) {\n"
                "        throw std::invalid_argument(\"--allreduce requires a CUDA/HIP build\");\n    }\n}\n\n"
            ),
            guard=r"common_apply_allreduce_config",
            expect_matches=1,
        ),
        Edit(
            id="allreduce-cli-options",
            anchor=r'add_opt\(common_arg\(\n        \{"-ts", "--tensor-split"\}',
            rationale="keep AllReduce selection adjacent to the existing multi-GPU split-mode controls",
            mode="insert_before",
            text=(
                "add_opt(common_arg(\n        {\"--allreduce\"}, \"PROVIDER\",\n"
                "        \"multi-GPU AllReduce provider: auto|ccl|host|adaptive|p2p|root3|butterfly (default: auto)\",\n"
                "        [](common_params &, const std::string & value) {\n            common_allreduce_provider = value;\n        }\n"
                "    ).set_env(\"LLAMA_ARG_ALLREDUCE\"));\n"
                "    add_opt(common_arg(\n        {\"--allreduce-wire\"}, \"WIRE\",\n"
                "        \"AllReduce wire format: native|q8 (default: native; q8 requires p2p)\",\n"
                "        [](common_params &, const std::string & value) {\n            common_allreduce_wire = value;\n        }\n"
                "    ).set_env(\"LLAMA_ARG_ALLREDUCE_WIRE\"));\n    "
            ),
            guard=r'"--allreduce-wire"',
            expect_matches=1,
        ),
        Edit(
            id="allreduce-config-apply",
            anchor=r"^    parse_cli_args\(\);$",
            rationale="apply the recorded AllReduce provider/wire pair once after every option and env var is parsed, making option order irrelevant",
            mode="insert_after",
            text="\n    common_apply_allreduce_config();",
            guard=r"^    common_apply_allreduce_config\(\);$",
            expect_matches=1,
        ),
    ),
)

LLAMA_BENCH = FilePatch(
    path="tools/llama-bench/llama-bench.cpp",
    description="add llama-bench AllReduce provider/wire CLI controls and apply them after parsing",
    language="none",
    edits=(
        Edit(
            id="bench-allreduce-config-helper",
            anchor=r"^static void print_usage\(int /\* argc \*/, char \*\* argv\) \{$",
            rationale="keep benchmark-only scalar CLI state and registry bridge immediately before usage/parser code",
            mode="insert_before",
            text=(
                "static std::string bench_allreduce_provider = \"auto\";\n"
                "static std::string bench_allreduce_wire = \"native\";\n\n"
                "static void bench_apply_allreduce_config(const std::string & provider, const std::string & wire) {\n"
                "    bool found = false;\n"
                "    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {\n"
                "        ggml_backend_reg_t reg = ggml_backend_reg_get(i);\n"
                "        auto set_config = (bool (*)(const char *, const char *, char *, size_t))\n"
                "            ggml_backend_reg_get_proc_address(reg, \"ggml_backend_comm_set_config\");\n"
                "        if (set_config == nullptr) {\n            continue;\n        }\n"
                "        found = true;\n        char err[256] = {};\n"
                "        if (!set_config(provider.c_str(), wire.c_str(), err, sizeof(err))) {\n"
                "            fprintf(stderr, \"error: %s\\n\", err);\n            exit(1);\n        }\n    }\n"
                "    if (!found && (provider != \"auto\" || wire != \"native\")) {\n"
                "        fprintf(stderr, \"error: --allreduce requires a CUDA/HIP build\\n\");\n        exit(1);\n    }\n}\n\n"
            ),
            guard=r"bench_apply_allreduce_config",
            expect_matches=1,
        ),
        Edit(
            id="bench-allreduce-usage",
            anchor=r'^    printf\("  -sm, --split-mode <none\|layer\|row\|tensor>         \(default: %s\)\\n", join\(transform_to_str\(cmd_params_defaults\.split_mode, split_mode_str\), ","\)\.c_str\(\)\);$',
            rationale="document benchmark-local scalar AllReduce controls beside split-mode",
            mode="insert_after",
            text=(
                "\n    printf(\"      --allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly> (default: auto)\\n\");\n"
                "    printf(\"      --allreduce-wire <native|q8>                 (default: native)\\n\");"
            ),
            guard=r"--allreduce-wire <native\|q8>",
            expect_matches=1,
        ),
        Edit(
            id="bench-allreduce-parser",
            anchor=(
                r"                params\.split_mode\.insert\(params\.split_mode\.end\(\), modes\.begin\(\), modes\.end\(\)\);\n"
                r"            \} else if \(arg == \"-lm\" \|\| arg == \"--load-mode\"\) \{"
            ),
            rationale="parse AllReduce provider/wire as scalar benchmark settings; comma-list syntax is invalid",
            mode="replace",
            text=(
                "                params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());\n"
                "            } else if (arg == \"--allreduce\") {\n"
                "                if (++i >= argc || strchr(argv[i], ',') != nullptr) {\n                    invalid_param = true;\n                    break;\n                }\n"
                "                bench_allreduce_provider = argv[i];\n"
                "            } else if (arg == \"--allreduce-wire\") {\n"
                "                if (++i >= argc || strchr(argv[i], ',') != nullptr) {\n                    invalid_param = true;\n                    break;\n                }\n"
                "                bench_allreduce_wire = argv[i];\n"
                "            } else if (arg == \"-lm\" || arg == \"--load-mode\") {"
            ),
            guard=r'arg == "--allreduce-wire"',
            expect_matches=1,
        ),
        Edit(
            id="bench-allreduce-apply",
            anchor=r"^    cmd_params params = parse_cmd_params\(argc, argv\);$",
            rationale="backends are already loaded; apply the parsed benchmark AllReduce configuration before device/model setup",
            mode="insert_after",
            text="\n    bench_apply_allreduce_config(bench_allreduce_provider, bench_allreduce_wire);",
            guard=r"bench_apply_allreduce_config\(bench_allreduce_provider, bench_allreduce_wire\)",
            expect_matches=1,
        ),
    ),
)

PATCHES = (CUDA, ARG_CPP, LLAMA_BENCH)
