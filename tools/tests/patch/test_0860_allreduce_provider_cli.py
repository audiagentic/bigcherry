"""Hardware-free mechanics tests for 0860_allreduce_provider_cli."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]

_BENCH_SOURCE = r'''static const cmd_params cmd_params_defaults = {};

static void print_usage(int /* argc */, char ** argv) {
    printf("  -sm, --split-mode <none|layer|row|tensor>         (default: %s)\n", join(transform_to_str(cmd_params_defaults.split_mode, split_mode_str), ",").c_str());
}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params;
    std::string arg;
    bool invalid_param = false;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-sm" || arg == "--split-mode") {
            std::vector<llama_split_mode> modes;
                params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-lm" || arg == "--load-mode") {
        }
    }
    return params;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    cmd_params params = parse_cmd_params(argc, argv);
    return 0;
}
'''

_ARG_SOURCE = r'''static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    auto parse_cli_args = [&]() {
    };

    // parse all CLI args now
    parse_cli_args();

    postprocess_cpu_params(params.cpuparams, nullptr);
}

static void add_rpc_devices(const std::string & servers) {
}

void common_params_add_all(common_params_context & ctx_arg) {
    add_opt(common_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU",
        [](common_params & params, const std::string & value) {}
    ));
}
'''


class Patch0860Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = patch_registry.load_registry(_REPO / "patches")
        descriptor = registry.get("0860_allreduce_provider_cli")
        patches = patch_registry.load_implementation(descriptor, root=_REPO / "patches")
        cls.bench_patches = tuple(p for p in patches if p.path == "tools/llama-bench/llama-bench.cpp")
        cls.arg_patches = tuple(p for p in patches if p.path == "common/arg.cpp")
        if len(cls.arg_patches) != 1:
            raise AssertionError(f"expected one common/arg.cpp FilePatch, got {len(cls.arg_patches)}")
        if len(cls.bench_patches) != 1:
            raise AssertionError(f"expected one llama-bench FilePatch, got {len(cls.bench_patches)}")

    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "tools/llama-bench/llama-bench.cpp"
        path.parent.mkdir(parents=True)
        path.write_text(_BENCH_SOURCE, encoding="utf-8")
        return td, root, path

    def test_apply_and_idempotent(self):
        td, root, path = self._tree()
        with td:
            first = apply_all(self.bench_patches, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = path.read_text(encoding="utf-8")
            self.assertIn("--allreduce", text)
            self.assertIn("ggml_backend_comm_set_config", text)
            before = text
            second = apply_all(self.bench_patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        td, root, path = self._tree()
        with td:
            path.write_text(_BENCH_SOURCE.replace("static void print_usage", "static void wrong_usage", 1), encoding="utf-8")
            results = apply_all(self.bench_patches, root)
            self.assertFalse(all(r.ok for r in results))

    def _arg_tree(self, source=_ARG_SOURCE):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "common/arg.cpp"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return td, root, path

    def test_arg_apply_and_idempotent(self):
        td, root, path = self._arg_tree()
        with td:
            first = apply_all(self.arg_patches, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            before = path.read_text(encoding="utf-8")
            second = apply_all(self.arg_patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_arg_config_is_applied_once_after_parsing_so_option_order_is_irrelevant(self):
        td, root, path = self._arg_tree()
        with td:
            apply_all(self.arg_patches, root)
            text = path.read_text(encoding="utf-8")
            # Callbacks only record; the single apply call sits right after parse_cli_args().
            self.assertIn("common_allreduce_provider = value;", text)
            self.assertIn("common_allreduce_wire = value;", text)
            self.assertEqual(text.count("common_apply_allreduce_config()"), 2)  # definition + one call
            self.assertIn("    parse_cli_args();" + chr(10) + "    common_apply_allreduce_config();", text)
            self.assertNotIn("common_apply_allreduce_config(value", text)
            # pristine b11233 order: parse_ex (the call site) precedes add_rpc_devices; the
            # definition must come before the call or arg.cpp does not compile.
            self.assertLess(text.index("static void common_apply_allreduce_config()"),
                            text.index("    common_apply_allreduce_config();"))
            # --allreduce-wire q8 without a provider must reach the backend as auto/q8 and fail closed there.
            self.assertNotIn("current_wire == ", text)

    def test_arg_missing_parse_anchor_fails_closed(self):
        td, root, path = self._arg_tree(_ARG_SOURCE.replace("    parse_cli_args();", "    run_parse();", 1))
        with td:
            self.assertFalse(all(r.ok for r in apply_all(self.arg_patches, root)))

    def test_arg_duplicate_parse_anchor_fails_closed(self):
        td, root, path = self._arg_tree(_ARG_SOURCE.replace("    parse_cli_args();", "    parse_cli_args();" + chr(10) + "    parse_cli_args();", 1))
        with td:
            self.assertFalse(all(r.ok for r in apply_all(self.arg_patches, root)))


if __name__ == "__main__":
    unittest.main()
