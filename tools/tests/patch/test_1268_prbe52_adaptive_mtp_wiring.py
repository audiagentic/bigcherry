"""Mechanics tests for 1268_prbe52_adaptive_mtp_wiring."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1268_prbe52_adaptive_mtp_wiring/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1268", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_COMMON_H = """struct common_params_speculative_draft {
    int32_t n_max = 3; // maximum number of tokens to draft during speculative decoding
    int32_t n_min = 0; // minimum number of draft tokens to use for speculative decoding
"""
_ARG = """    add_opt(common_arg(
        {\"--spec-draft-n-min\"}, \"N\",
        string_format(\"minimum number of draft tokens to use for speculative decoding (default: %d)\", params.speculative.draft.n_min),
        [](common_params & params, int value) {
            params.speculative.draft.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env(\"LLAMA_ARG_SPEC_DRAFT_N_MIN\"));
"""
_SPEC = """#include <algorithm>
#include <cassert>

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;

    void ctor_tail() {
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};
"""


class Patch1268Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "common").mkdir(parents=True)
        (root / "common/common.h").write_text(_COMMON_H, encoding="utf-8")
        (root / "common/arg.cpp").write_text(_ARG, encoding="utf-8")
        (root / "common/speculative.cpp").write_text(_SPEC, encoding="utf-8")
        return td, root

    def test_apply_and_idempotent(self):
        td, root = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            self.assertIn("n_min_adaptive = 0", (root / "common/common.h").read_text())
            self.assertIn("LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE", (root / "common/arg.cpp").read_text())
            text = (root / "common/speculative.cpp").read_text()
            self.assertIn("effective_n_max", text)
            self.assertIn("adaptive_state[seq_id].update", text)
            before = {p: (root / p).read_text() for p in ("common/common.h", "common/arg.cpp", "common/speculative.cpp")}
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(before, {p: (root / p).read_text() for p in before})

    def test_missing_depth_anchor_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "common/speculative.cpp"
            path.write_text(_SPEC.replace("params.n_max <= (int) result.size()", "false"), encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
