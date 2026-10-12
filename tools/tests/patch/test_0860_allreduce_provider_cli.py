"""Hardware-free mechanics tests for 0860_allreduce_provider_cli."""

from __future__ import annotations

import sys
import re
from collections import Counter
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402
from bigcherry.core import paths  # noqa: E402

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
        registry = patch_registry.load_registry("llamacpp", _REPO / "engines" / "llamacpp" / "patches")
        descriptor = registry.get("0860_allreduce_provider_cli")
        patches = patch_registry.load_implementation("llamacpp", descriptor, root=_REPO / "engines" / "llamacpp" / "patches")
        cls.cuda_patches = tuple(p for p in patches if p.path == "ggml/src/ggml-cuda/ggml-cuda.cu")
        cls.cuda_owned = tuple(p for p in patches if p.path == "ggml/src/ggml-cuda/bc-allreduce-provider-cli.cuh")
        cls.bench_patches = tuple(p for p in patches if p.path == "tools/llama-bench/llama-bench.cpp")
        cls.bench_owned = tuple(p for p in patches if p.path == "tools/llama-bench/bc-allreduce-provider-cli.h")
        cls.arg_patches = tuple(p for p in patches if p.path == "common/arg.cpp")
        cls.arg_owned = tuple(p for p in patches if p.path == "common/bc-allreduce-provider-cli.h")
        cls.bench_apply_patches = cls.bench_owned + cls.bench_patches
        cls.arg_apply_patches = cls.arg_owned + cls.arg_patches
        if len(cls.cuda_patches) != 1:
            raise AssertionError(f"expected one CUDA FilePatch, got {len(cls.cuda_patches)}")
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
            first = apply_all(self.bench_apply_patches, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = path.read_text(encoding="utf-8")
            self.assertIn("--allreduce", text)
            self.assertIn("--allreduce-switch-bytes", text)
            self.assertIn("bench_allreduce_switch_bytes = argv[i];", text)
            self.assertIn("ggml_backend_comm_set_config", (root / "tools/llama-bench/bc-allreduce-provider-cli.h").read_text(encoding="utf-8"))
            before = text
            second = apply_all(self.bench_apply_patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        td, root, path = self._tree()
        with td:
            path.write_text(_BENCH_SOURCE.replace("static void print_usage", "static void wrong_usage", 1), encoding="utf-8")
            results = apply_all(self.bench_apply_patches, root)
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
            first = apply_all(self.arg_apply_patches, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            before = path.read_text(encoding="utf-8")
            second = apply_all(self.arg_apply_patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_arg_config_is_applied_once_after_parsing_so_option_order_is_irrelevant(self):
        td, root, path = self._arg_tree()
        with td:
            apply_all(self.arg_apply_patches, root)
            text = path.read_text(encoding="utf-8")
            owned = (root / "common/bc-allreduce-provider-cli.h").read_text(encoding="utf-8")
            self.assertIn("common_allreduce_provider = value;", text)
            self.assertIn("common_allreduce_wire = value;", text)
            self.assertIn("common_allreduce_switch_bytes = value;", text)
            self.assertIn("common_parse_allreduce_switch_bytes", owned)
            self.assertIn("? 98304 : common_parse_allreduce_switch_bytes", owned)
            self.assertEqual(owned.count("static void common_apply_allreduce_config()"), 1)
            self.assertEqual(text.count("    common_apply_allreduce_config();"), 1)
            self.assertIn("    parse_cli_args();" + chr(10) + "    common_apply_allreduce_config();", text)
            self.assertNotIn("common_apply_allreduce_config(value", text)
            self.assertNotIn("current_wire == ", text)

    def test_switch_bytes_has_no_environment_binding(self):
        options = next(e for e in self.arg_patches[0].edits if e.id == "allreduce-cli-options").text
        self.assertIn('{"--allreduce-switch-bytes"}, "N"', options)
        self.assertNotIn("LLAMA_ARG_ALLREDUCE_SWITCH", options)
        helper = self.arg_owned[0].edits[0].text
        self.assertIn("must be a non-negative integer", helper)
        self.assertIn("effective_switch_bytes", helper)

    def test_single_named_switch_default_constant(self):
        source = (_REPO / "engines/llamacpp/patches/0860_allreduce_provider_cli/patch.py").read_text(encoding="utf-8")
        self.assertIn("ADAPTIVE_SWITCH_BYTES_DEFAULT = 96 << 10", source)
        self.assertEqual(source.count("98304"), 0)

    def test_cuda_config_owns_default_switch_and_is_visible_before_adaptive_dispatch(self):
        edit = next(e for e in self.cuda_patches[0].edits if e.id == "allreduce-provider-config")
        owned = self.cuda_owned[0].edits[0].text
        self.assertEqual(edit.anchor, r"^static bool ggml_backend_cuda_comm_try_allreduce_internal\($")
        self.assertIn('#include "bc-allreduce-provider-cli.cuh"', edit.text)
        self.assertIn("size_t switch_bytes = 98304;", owned)
        self.assertIn("const char * provider, const char * wire, size_t switch_bytes", owned)
        self.assertIn("g_ggml_backend_cuda_comm_config.switch_bytes = switch_bytes;", owned)

    def test_arg_missing_parse_anchor_fails_closed(self):
        td, root, path = self._arg_tree(_ARG_SOURCE.replace("    parse_cli_args();", "    run_parse();", 1))
        with td:
            self.assertFalse(all(r.ok for r in apply_all(self.arg_apply_patches, root)))

    def test_arg_duplicate_parse_anchor_fails_closed(self):
        td, root, path = self._arg_tree(
            _ARG_SOURCE.replace("    parse_cli_args();", "    parse_cli_args();" + chr(10) + "    parse_cli_args();", 1)
        )
        with td:
            self.assertFalse(all(r.ok for r in apply_all(self.arg_apply_patches, root)))


_ID = "0860_allreduce_provider_cli"
_COMPOSE_INPUTS = ("ggml/src/ggml-cuda/ggml-cuda.cu", "common/arg.cpp", "tools/llama-bench/llama-bench.cpp")
_CREATED = {
    "ggml/src/ggml-cuda/bc-allreduce-provider-cli.cuh": ("ggml/src/ggml-cuda/ggml-cuda.cu", '#include "bc-allreduce-provider-cli.cuh"  // BigCherry 0860 owned definitions\n\n', "static bool ggml_backend_comm_set_config("),
    "common/bc-allreduce-provider-cli.h": ("common/arg.cpp", '#include "bc-allreduce-provider-cli.h"  // BigCherry 0860 owned definitions\n\n', "static void common_apply_allreduce_config()"),
    "tools/llama-bench/bc-allreduce-provider-cli.h": ("tools/llama-bench/llama-bench.cpp", '#include "bc-allreduce-provider-cli.h"  // BigCherry 0860 owned definitions\n\n', "static void bench_apply_allreduce_config("),
}


def _owned_lines(*texts):
    return Counter(line.strip() for text in texts for line in text.splitlines() if line.strip())


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch0860OwnedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = patch_registry.load_registry("llamacpp", _REPO / "engines" / "llamacpp" / "patches")
        cls.patches = tuple(patch_registry.load_implementation("llamacpp", registry.get(_ID), root=_REPO / "engines" / "llamacpp" / "patches"))

    def test_composed_shape_guards_identity_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source("llamacpp", "bigcherry", _COMPOSE_INPUTS, td, stop_before=_ID)
            before = {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS}
            for fp in self.patches:
                if fp.create:
                    self.assertFalse((root / fp.path).exists())
                    continue
                for edit in fp.edits:
                    self.assertIsNone(re.search(edit.guard_pattern(), before[fp.path], re.MULTILINE), (fp.path, edit.id))
            res = apply_all(self.patches, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            after = {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS}
            owned = {p: (root / p).read_text(encoding="utf-8") for p in _CREATED}
            for path, (upstream, include, definition) in _CREATED.items():
                self.assertIn(definition, owned[path])
                self.assertEqual(after[upstream].count(include.strip()), 1)
                self.assertNotIn(definition, after[upstream])
            self.assertIn('return (void *) ggml_backend_comm_set_config;', after["ggml/src/ggml-cuda/ggml-cuda.cu"])
            self.assertIn("    common_apply_allreduce_config();", after["common/arg.cpp"])
            self.assertIn("bench_apply_allreduce_config(bench_allreduce_provider", after["tools/llama-bench/llama-bench.cpp"])
            legacy = dict(after)
            for path, (upstream, include, _) in _CREATED.items():
                legacy[upstream] = legacy[upstream].replace(include, owned[path], 1)
            new_lines = _owned_lines(*(after[p] for p in _COMPOSE_INPUTS), *(owned[p] for p in _CREATED))
            old_lines = _owned_lines(*(legacy[p] for p in _COMPOSE_INPUTS))
            self.assertEqual(new_lines, old_lines + _owned_lines(*(v[1] for v in _CREATED.values())))
            self.assertEqual(sum(new_lines.values()), sum(old_lines.values()) + 3)
            snapshot = {p: (root / p).read_text(encoding="utf-8") for p in (*_COMPOSE_INPUTS, *_CREATED)}
            again = apply_all(self.patches, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(snapshot, {p: (root / p).read_text(encoding="utf-8") for p in snapshot})

    def test_missing_owned_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source("llamacpp", "bigcherry", _COMPOSE_INPUTS, td, stop_before=_ID)
            p = root / "common/arg.cpp"
            source = p.read_text(encoding="utf-8")
            anchor = "static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {"
            self.assertEqual(source.count(anchor), 1)
            p.write_text(source.replace(anchor, anchor.replace("common_params_parse_ex", "common_params_parse_ex_drifted"), 1), encoding="utf-8")
            self.assertFalse(all(r.ok for r in apply_all(self.patches, root)))


if __name__ == "__main__":
    unittest.main()
