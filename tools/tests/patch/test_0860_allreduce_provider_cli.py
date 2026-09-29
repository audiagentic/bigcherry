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


class Patch0860Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = patch_registry.load_registry(_REPO / "patches")
        descriptor = registry.get("0860_allreduce_provider_cli")
        patches = patch_registry.load_implementation(descriptor, root=_REPO / "patches")
        cls.bench_patches = tuple(p for p in patches if p.path == "tools/llama-bench/llama-bench.cpp")
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


if __name__ == "__main__":
    unittest.main()
