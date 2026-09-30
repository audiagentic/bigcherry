"""0860: explicit AllReduce provider/wire/switch configuration seam."""

GROUP = "core"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="replace AllReduce env provider selection with explicit provider/wire/switch configuration",
    language="none",
    edits=(
        Edit(
            id="allreduce-provider-config",
            anchor=r"^static bool ggml_backend_cuda_comm_try_allreduce_internal\($",
            rationale="define process-wide CLI configuration before 0840's adaptive per-call dispatcher",
            mode="insert_before",
            text=(
                "struct ggml_backend_cuda_comm_config {\n"
                "    std::string provider = \"auto\";\n"
                "    std::string wire = \"native\";\n"
                "    size_t switch_bytes = 1048576;\n"
                "};\n\n"
                "static ggml_backend_cuda_comm_config g_ggml_backend_cuda_comm_config;\n\n"
                "static bool ggml_backend_comm_set_config(\n"
                "        const char * provider, const char * wire, size_t switch_bytes, char * err, size_t err_len) {\n"
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
                "    g_ggml_backend_cuda_comm_config.switch_bytes = switch_bytes;\n"
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
            rationale="select only from the validated explicit configuration; auto preserves the stock platform default",
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
                "        GGML_LOG_INFO(\"BIGCHERRY_PATCH_HIT patch=0860_allreduce_provider_cli provider=%s wire=%s switch_bytes=%zu\\n\",\n"
                "            provider.c_str(), wire.c_str(), g_ggml_backend_cuda_comm_config.switch_bytes);\n"
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
    description="add explicit AllReduce provider/wire/switch CLI options and apply them once after parsing",
    language="none",
    edits=(
        Edit(
            id="allreduce-config-helper",
            anchor=r"^static bool common_params_parse_ex\(int argc, char \*\* argv, common_params_context & ctx_arg\) \{$",
            rationale="callbacks record values; parse switch bytes and apply one backend config after parse_cli_args",
            mode="insert_before",
            text=(
                "static std::string common_allreduce_provider;\n"
                "static std::string common_allreduce_wire;\n"
                "static std::string common_allreduce_switch_bytes;\n\n"
                "static size_t common_parse_allreduce_switch_bytes(const std::string & text) {\n"
                "    if (text.empty() || text[0] < '0' || text[0] > '9') {\n"
                "        throw std::invalid_argument(\"--allreduce-switch-bytes must be a non-negative integer\");\n"
                "    }\n"
                "    size_t pos = 0;\n"
                "    unsigned long long parsed = 0;\n"
                "    try {\n"
                "        parsed = std::stoull(text, &pos, 10);\n"
                "    } catch (...) {\n"
                "        throw std::invalid_argument(\"--allreduce-switch-bytes must be a non-negative integer\");\n"
                "    }\n"
                "    const size_t value = (size_t) parsed;\n"
                "    if (pos != text.size() || (unsigned long long) value != parsed) {\n"
                "        throw std::invalid_argument(\"--allreduce-switch-bytes must be a non-negative integer\");\n"
                "    }\n"
                "    return value;\n"
                "}\n\n"
                "// Option callbacks only record values; applied once after all args/env are parsed.\n"
                "static void common_apply_allreduce_config() {\n"
                "    if (common_allreduce_provider.empty() && common_allreduce_wire.empty() && common_allreduce_switch_bytes.empty()) {\n"
                "        return;\n"
                "    }\n\n"
                "    const std::string effective_provider = common_allreduce_provider.empty() ? \"auto\" : common_allreduce_provider;\n"
                "    const std::string effective_wire = common_allreduce_wire.empty() ? \"native\" : common_allreduce_wire;\n"
                "    const size_t effective_switch_bytes = common_allreduce_switch_bytes.empty()\n"
                "        ? 1048576 : common_parse_allreduce_switch_bytes(common_allreduce_switch_bytes);\n\n"
                "    ggml_backend_load_all();\n"
                "    bool found = false;\n"
                "    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {\n"
                "        ggml_backend_reg_t reg = ggml_backend_reg_get(i);\n"
                "        auto set_config = (bool (*)(const char *, const char *, size_t, char *, size_t))\n"
                "            ggml_backend_reg_get_proc_address(reg, \"ggml_backend_comm_set_config\");\n"
                "        if (set_config == nullptr) {\n"
                "            continue;\n"
                "        }\n"
                "        found = true;\n"
                "        char err[256] = {};\n"
                "        if (!set_config(effective_provider.c_str(), effective_wire.c_str(), effective_switch_bytes, err, sizeof(err))) {\n"
                "            throw std::invalid_argument(err);\n"
                "        }\n"
                "    }\n\n"
                "    if (!found && (effective_provider != \"auto\" || effective_wire != \"native\" || !common_allreduce_switch_bytes.empty())) {\n"
                "        throw std::invalid_argument(\"--allreduce requires a CUDA/HIP build\");\n"
                "    }\n"
                "}\n\n"
            ),
            guard=r"common_apply_allreduce_config",
            expect_matches=1,
        ),
        Edit(
            id="allreduce-cli-options",
            anchor=r'add_opt\(common_arg\(\n        \{"-ts", "--tensor-split"\}',
            rationale="keep AllReduce controls adjacent to multi-GPU split-mode controls",
            mode="insert_before",
            text=(
                "add_opt(common_arg(\n"
                "        {\"--allreduce\"}, \"PROVIDER\",\n"
                "        \"multi-GPU AllReduce provider: auto|ccl|host|adaptive|p2p|root3|butterfly (default: auto)\",\n"
                "        [](common_params &, const std::string & value) {\n"
                "            common_allreduce_provider = value;\n"
                "        }\n"
                "    ).set_env(\"LLAMA_ARG_ALLREDUCE\"));\n"
                "    add_opt(common_arg(\n"
                "        {\"--allreduce-wire\"}, \"WIRE\",\n"
                "        \"AllReduce wire format: native|q8 (default: native; q8 requires p2p)\",\n"
                "        [](common_params &, const std::string & value) {\n"
                "            common_allreduce_wire = value;\n"
                "        }\n"
                "    ).set_env(\"LLAMA_ARG_ALLREDUCE_WIRE\"));\n"
                "    add_opt(common_arg(\n"
                "        {\"--allreduce-switch-bytes\"}, \"N\",\n"
                "        \"adaptive AllReduce host/RCCL crossover in bytes (default: 1048576; host below N, RCCL at/above N)\",\n"
                "        [](common_params &, const std::string & value) {\n"
                "            common_allreduce_switch_bytes = value;\n"
                "        }\n"
                "    ));\n"
                "    "
            ),
            guard=r'"--allreduce-switch-bytes"',
            expect_matches=1,
        ),
        Edit(
            id="allreduce-config-apply",
            anchor=r"^    parse_cli_args\(\);$",
            rationale="apply the recorded AllReduce configuration once after every CLI/env option is parsed",
            mode="insert_after",
            text="\n    common_apply_allreduce_config();",
            guard=r"^    common_apply_allreduce_config\(\);$",
            expect_matches=1,
        ),
    ),
)

LLAMA_BENCH = FilePatch(
    path="tools/llama-bench/llama-bench.cpp",
    description="add llama-bench AllReduce provider/wire/switch controls and apply after parsing",
    language="none",
    edits=(
        Edit(
            id="bench-allreduce-config-helper",
            anchor=r"^static void print_usage\(int /\* argc \*/, char \*\* argv\) \{$",
            rationale="keep benchmark scalar state and registry bridge immediately before usage/parser code",
            mode="insert_before",
            text=(
                "static std::string bench_allreduce_provider = \"auto\";\n"
                "static std::string bench_allreduce_wire = \"native\";\n"
                "static std::string bench_allreduce_switch_bytes = \"1048576\";\n\n"
                "static size_t bench_parse_allreduce_switch_bytes(const std::string & text) {\n"
                "    if (text.empty() || text[0] < '0' || text[0] > '9') {\n"
                "        fprintf(stderr, \"error: --allreduce-switch-bytes must be a non-negative integer\\n\");\n"
                "        exit(1);\n"
                "    }\n"
                "    size_t pos = 0;\n"
                "    unsigned long long parsed = 0;\n"
                "    try {\n"
                "        parsed = std::stoull(text, &pos, 10);\n"
                "    } catch (...) {\n"
                "        fprintf(stderr, \"error: --allreduce-switch-bytes must be a non-negative integer\\n\");\n"
                "        exit(1);\n"
                "    }\n"
                "    const size_t value = (size_t) parsed;\n"
                "    if (pos != text.size() || (unsigned long long) value != parsed) {\n"
                "        fprintf(stderr, \"error: --allreduce-switch-bytes must be a non-negative integer\\n\");\n"
                "        exit(1);\n"
                "    }\n"
                "    return value;\n"
                "}\n\n"
                "static void bench_apply_allreduce_config(\n"
                "        const std::string & provider, const std::string & wire, const std::string & switch_bytes) {\n"
                "    const size_t parsed_switch_bytes = bench_parse_allreduce_switch_bytes(switch_bytes);\n"
                "    bool found = false;\n"
                "    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {\n"
                "        ggml_backend_reg_t reg = ggml_backend_reg_get(i);\n"
                "        auto set_config = (bool (*)(const char *, const char *, size_t, char *, size_t))\n"
                "            ggml_backend_reg_get_proc_address(reg, \"ggml_backend_comm_set_config\");\n"
                "        if (set_config == nullptr) {\n"
                "            continue;\n"
                "        }\n"
                "        found = true;\n"
                "        char err[256] = {};\n"
                "        if (!set_config(provider.c_str(), wire.c_str(), parsed_switch_bytes, err, sizeof(err))) {\n"
                "            fprintf(stderr, \"error: %s\\n\", err);\n"
                "            exit(1);\n"
                "        }\n"
                "    }\n"
                "    if (!found && (provider != \"auto\" || wire != \"native\" || switch_bytes != \"1048576\")) {\n"
                "        fprintf(stderr, \"error: --allreduce requires a CUDA/HIP build\\n\");\n"
                "        exit(1);\n"
                "    }\n"
                "}\n\n"
            ),
            guard=r"bench_apply_allreduce_config",
            expect_matches=1,
        ),
        Edit(
            id="bench-allreduce-usage",
            anchor=r'^    printf\("  -sm, --split-mode <none\|layer\|row\|tensor>         \(default: %s\)\\n", join\(transform_to_str\(cmd_params_defaults\.split_mode, split_mode_str\), ","\)\.c_str\(\)\);$',
            rationale="document benchmark AllReduce controls beside split-mode",
            mode="insert_after",
            text=(
                "\n    printf(\"      --allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly> (default: auto)\\n\");\n"
                "    printf(\"      --allreduce-wire <native|q8>                 (default: native)\\n\");\n"
                "    printf(\"      --allreduce-switch-bytes <N>                  (default: 1048576)\\n\");"
            ),
            guard=r"--allreduce-switch-bytes <N>",
            expect_matches=1,
        ),
        Edit(
            id="bench-allreduce-parser",
            anchor=(
                r"                params\.split_mode\.insert\(params\.split_mode\.end\(\), modes\.begin\(\), modes\.end\(\)\);\n"
                r"            \} else if \(arg == \"-lm\" \|\| arg == \"--load-mode\"\) \{"
            ),
            rationale="parse AllReduce controls as scalar benchmark settings; comma-list syntax is invalid",
            mode="replace",
            text=(
                "                params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());\n"
                "            } else if (arg == \"--allreduce\") {\n"
                "                if (++i >= argc || strchr(argv[i], ',') != nullptr) {\n"
                "                    invalid_param = true;\n"
                "                    break;\n"
                "                }\n"
                "                bench_allreduce_provider = argv[i];\n"
                "            } else if (arg == \"--allreduce-wire\") {\n"
                "                if (++i >= argc || strchr(argv[i], ',') != nullptr) {\n"
                "                    invalid_param = true;\n"
                "                    break;\n"
                "                }\n"
                "                bench_allreduce_wire = argv[i];\n"
                "            } else if (arg == \"--allreduce-switch-bytes\") {\n"
                "                if (++i >= argc || strchr(argv[i], ',') != nullptr) {\n"
                "                    invalid_param = true;\n"
                "                    break;\n"
                "                }\n"
                "                bench_allreduce_switch_bytes = argv[i];\n"
                "            } else if (arg == \"-lm\" || arg == \"--load-mode\") {"
            ),
            guard=r'arg == "--allreduce-switch-bytes"',
            expect_matches=1,
        ),
        Edit(
            id="bench-allreduce-apply",
            anchor=r"^    cmd_params params = parse_cmd_params\(argc, argv\);$",
            rationale="backends are already loaded; apply the parsed AllReduce configuration before device/model setup",
            mode="insert_after",
            text=(
                "\n    bench_apply_allreduce_config("
                "bench_allreduce_provider, bench_allreduce_wire, bench_allreduce_switch_bytes);"
            ),
            guard=r"bench_apply_allreduce_config\(bench_allreduce_provider, bench_allreduce_wire, bench_allreduce_switch_bytes\)",
            expect_matches=1,
        ),
    ),
)

PATCHES = (CUDA, ARG_CPP, LLAMA_BENCH)
