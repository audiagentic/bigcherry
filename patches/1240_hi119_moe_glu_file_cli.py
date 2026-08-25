"""HI119: a --moe-glu-file CLI hook, so a Python evidence producer can
instantiate the new test_bigcherry_moe_glu_fusion class (patch 1239) with
runtime-derived parameters, without recompiling test-backend-ops for every
new dispatch shape.

Context: HI119's own step 7 required this -- the two instances patch 1239
registered in the static corpus (matching HI108's real routed dispatch
shape, plus a sanity baseline) prove the harness DESIGN works on real
hardware, but are not a general mechanism. test-backend-ops' existing
--test-file/test_generic_op escape hatch cannot help here either (confirmed
in HI108/HI119's own investigation: it builds exactly one op per line, not
a multi-node fused subgraph).

Mirrors make_test_cases_from_file()'s own established convention exactly
(one line, numeric fields via istringstream >>, same style as that
function's op/type/ne/op_params/sources fields) rather than inventing a new
format: `type glu_op k n m n_mats n_used broadcast`, one line, matching
bigcherry_moe_glu_fusion's constructor parameter order.

Adds a NEW parameter to test_backend()'s own signature (moe_glu_file_path)
rather than overloading test_file_path, so --test-file and --moe-glu-file
stay independently selectable/distinguishable in the argv parsing and in
any future caller -- test-backend-ops' own real semantics (a corpus can be
the eval switch, --test-file's parsed corpus, or now --moe-glu-file's
single case) are then an explicit three-way choice at the one call site
that builds the corpus, not two overloaded meanings on one variable."""

GROUP = "core"
# Verified offline: dry-run apply + idempotence against the real vendored
# checkout, composed with 1222+1236+1238+1239. Real-hardware validated on
# Brutus: compiled cleanly, and `--moe-glu-file <path>` with a real one-line
# param file produced the exact same PASS result and the exact same
# fusion=GATE/op=GLU recorded signature as the statically-registered
# instance in patch 1239, confirming the runtime-parameterized path is
# equivalent to the compiled-in one, not just that it doesn't crash.
STATE = "untested"

REQUIRES = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1238_hi119_deterministic_init_mul_mat_id_tensors",
    "1239_hi119_fused_moe_glu_test_case",
)

import re as _re

from bigcherry import csource as _csource
from bigcherry.patcher import Edit, FilePatch

_FUNC_ANCHOR_SOURCE = '''static bool test_backend(ggml_backend_t backend, ggml_backend_dev_t dev, test_mode mode, const char * op_names_filter, const char * params_filter,
                         printer * output_printer, const char * test_file_path, int parallel_workers) {'''

_FUNC_NEW = '''// bigcherry (HI119): analogous to make_test_cases_from_file() but for the
// registered test_bigcherry_moe_glu_fusion class -- a single line of
// numeric fields (type glu_op k n m n_mats n_used broadcast), same
// convention as make_test_cases_from_file()'s own numeric-field format.
// This is the runtime-parameterization hook HI119 needs: a Python
// evidence producer can derive these eight values from a real BigCherry
// dispatch signature (ne0/ne1/ned/n_expert/n_expert_used/src0_type/glu_op)
// without recompiling test-backend-ops for every new shape. See
// patches/1240_hi119_moe_glu_file_cli.py.
static std::vector<std::unique_ptr<test_case>> make_test_cases_from_moe_glu_file(const char * path) {
    std::ifstream f(path);

    if (!f.is_open()) {
        throw std::runtime_error("Unable to read moe-glu-file");
    }

    std::vector<std::unique_ptr<test_case>> test_cases;

    std::string line;
    if (std::getline(f, line)) {
        std::istringstream iss(line);
        uint64_t tmp;

        iss >> tmp;
        ggml_type type = (ggml_type) tmp;
        iss >> tmp;
        ggml_glu_op glu_op = (ggml_glu_op) tmp;
        int64_t k, n, m;
        int n_mats, n_used;
        int broadcast;
        iss >> k >> n >> m >> n_mats >> n_used >> broadcast;

        if (!iss.fail()) {
            test_cases.emplace_back(new test_bigcherry_moe_glu_fusion(type, glu_op, k, n, m, n_mats, n_used, broadcast != 0));
        }
    }

    return test_cases;
}

static bool test_backend(ggml_backend_t backend, ggml_backend_dev_t dev, test_mode mode, const char * op_names_filter, const char * params_filter,
                         printer * output_printer, const char * test_file_path, const char * moe_glu_file_path, int parallel_workers) {'''

_FUNC_ANCHOR = _re.escape(_csource.strip_noise(_FUNC_ANCHOR_SOURCE, "c"))

_SELECT_ANCHOR_SOURCE = '''    if (test_file_path == nullptr) {
        switch (mode) {
        case MODE_TEST:
        case MODE_GRAD:
        case MODE_SUPPORT:
            test_cases = make_test_cases_eval();
            break;
        case MODE_PERF:
            test_cases = make_test_cases_perf();
            break;
        }
    } else {
        test_cases = make_test_cases_from_file(test_file_path);
    }'''

_SELECT_NEW = '''    if (moe_glu_file_path != nullptr) {
        test_cases = make_test_cases_from_moe_glu_file(moe_glu_file_path);
    } else if (test_file_path == nullptr) {
        switch (mode) {
        case MODE_TEST:
        case MODE_GRAD:
        case MODE_SUPPORT:
            test_cases = make_test_cases_eval();
            break;
        case MODE_PERF:
            test_cases = make_test_cases_perf();
            break;
        }
    } else {
        test_cases = make_test_cases_from_file(test_file_path);
    }'''

_SELECT_ANCHOR = _re.escape(_csource.strip_noise(_SELECT_ANCHOR_SOURCE, "c"))

_USAGE_ANCHOR_SOURCE = '''    printf("    --test-file reads test operators from a test file generated by test-export-graph-ops\\n");'''
_USAGE_NEW = (
    _USAGE_ANCHOR_SOURCE + "\n"
    '    printf("    --moe-glu-file <path> reads one line (type glu_op k n m n_mats n_used broadcast) and runs\\n");\n'
    '    printf("        exactly one test_bigcherry_moe_glu_fusion instance (HI119)\\n");'
)
_USAGE_ANCHOR = _re.escape(_csource.strip_noise(_USAGE_ANCHOR_SOURCE, "c"))

_DECL_ANCHOR_SOURCE = '''    const char * test_file_path = nullptr;
    int parallel_workers = 1;'''
_DECL_NEW = '''    const char * test_file_path = nullptr;
    const char * moe_glu_file_path = nullptr;
    int parallel_workers = 1;'''
_DECL_ANCHOR = _re.escape(_csource.strip_noise(_DECL_ANCHOR_SOURCE, "c"))

_ARG_ANCHOR_SOURCE = '''        } else if (strcmp(argv[i], "--test-file") == 0) {
            if (i + 1 < argc) {
                test_file_path = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-j") == 0) {'''
_ARG_NEW = '''        } else if (strcmp(argv[i], "--test-file") == 0) {
            if (i + 1 < argc) {
                test_file_path = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--moe-glu-file") == 0) {
            if (i + 1 < argc) {
                moe_glu_file_path = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-j") == 0) {'''
_ARG_ANCHOR = _re.escape(_csource.strip_noise(_ARG_ANCHOR_SOURCE, "c"))

_CALL_ANCHOR_SOURCE = '''        bool ok = test_backend(backend.get(), dev, mode, op_names_filter, params_filter, output_printer.get(), test_file_path, parallel_workers);'''
_CALL_NEW = '''        bool ok = test_backend(backend.get(), dev, mode, op_names_filter, params_filter, output_printer.get(), test_file_path, moe_glu_file_path, parallel_workers);'''
_CALL_ANCHOR = _re.escape(_csource.strip_noise(_CALL_ANCHOR_SOURCE, "c"))

PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description="--moe-glu-file <path> CLI hook: runs a single, runtime-parameterized "
                "test_bigcherry_moe_glu_fusion instance, so a Python evidence producer "
                "can drive it from a real dispatch signature (HI119)",
    edits=(
        Edit(
            id="hi119-moe-glu-file-function",
            anchor=_FUNC_ANCHOR,
            mode="replace",
            rationale="insert make_test_cases_from_moe_glu_file() immediately before "
                       "test_backend(), and extend test_backend()'s own signature with "
                       "the new moe_glu_file_path parameter in the same edit",
            text=_FUNC_NEW,
            guard=r"bigcherry \(HI119\): analogous to make_test_cases_from_file\(\)",
            max_span_lines=6,
        ),
        Edit(
            id="hi119-moe-glu-file-selection",
            anchor=_SELECT_ANCHOR,
            mode="replace",
            rationale="moe_glu_file_path takes priority over test_file_path and the "
                       "ordinary eval/perf corpus switch -- an explicit three-way choice",
            text=_SELECT_NEW,
            guard=r"if \(moe_glu_file_path != nullptr\) \{",
            max_span_lines=15,
        ),
        Edit(
            id="hi119-moe-glu-file-usage",
            anchor=_USAGE_ANCHOR,
            mode="replace",
            rationale="document the new flag in --help output",
            text=_USAGE_NEW,
            guard=r"--moe-glu-file <path> reads one line",
            max_span_lines=3,
        ),
        Edit(
            id="hi119-moe-glu-file-decl",
            anchor=_DECL_ANCHOR,
            mode="replace",
            rationale="main()'s own local variable for the new flag's value",
            text=_DECL_NEW,
            guard=r"const char \* moe_glu_file_path = nullptr;",
            max_span_lines=3,
        ),
        Edit(
            id="hi119-moe-glu-file-arg",
            anchor=_ARG_ANCHOR,
            mode="replace",
            rationale="argv parsing for --moe-glu-file, inserted right after --test-file's "
                       "own branch",
            text=_ARG_NEW,
            guard=r'strcmp\(argv\[i\], "--moe-glu-file"\) == 0',
            max_span_lines=15,
        ),
        Edit(
            id="hi119-moe-glu-file-call",
            anchor=_CALL_ANCHOR,
            mode="replace",
            rationale="thread moe_glu_file_path through to the (now six-argument-plus) "
                       "test_backend() call",
            text=_CALL_NEW,
            guard=r"test_file_path, moe_glu_file_path, parallel_workers\);",
            max_span_lines=3,
        ),
    ),
)

PATCHES = [PATCH]
