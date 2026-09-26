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
    int32_t n_chain_heads = 0; // lower-order composed field
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

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    std::vector<std::vector<float>> pending_g_last;
    std::vector<int32_t> verify_g_rows;

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            other_batch_add(seq_id);

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_g_last[seq_id][0] = (float) i_g;
    }
};

struct another_draft_impl : public common_speculative_impl {
    void draft(common_speculative_draft_params_vec & dparams) override {
            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            other_batch_add_again(seq_id);
    }
};

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

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            warn();
        }
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    chain_h[seq_id].push_back(0.0f);
                }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};
"""


class Patch1268Mechanics(unittest.TestCase):
    def _tree(self, *, cosmetic_noise=False):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "common").mkdir(parents=True)
        common_h = _COMMON_H
        arg = _ARG
        spec = _SPEC
        if cosmetic_noise:
            common_h = common_h.replace("minimum number of draft tokens", "wording changed upstream")
            arg = arg.replace("--spec-draft-n-min", "--string-is-noise").replace("minimum number of draft tokens to use for speculative decoding (default: %d)", "string literal changed")
            spec = spec.replace("[n_seq][n_embd]", "comment changed").replace("/*is_other*/", "/* renamed comment */")
        (root / "common/common.h").write_text(common_h, encoding="utf-8")
        (root / "common/arg.cpp").write_text(arg, encoding="utf-8")
        (root / "common/speculative.cpp").write_text(spec, encoding="utf-8")
        return td, root

    def test_apply_and_idempotent(self):
        td, root = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            common_h = (root / "common/common.h").read_text()
            self.assertIn("n_chain_heads = 0", common_h)
            self.assertIn("n_min_adaptive = 0", common_h)
            self.assertIn("LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE", (root / "common/arg.cpp").read_text())
            text = (root / "common/speculative.cpp").read_text()
            self.assertEqual(text.count("adaptive_state.at(seq_id).reset"), 1)
            self.assertEqual(text.count("last_n_draft[seq_id] = 0"), 2)  # begin reset + MTP draft reset only
            self.assertEqual(text.count("effective_n_max"), 4)
            self.assertEqual(text.count("adaptive_state[seq_id].update"), 1)
            eagle3_text = text.split("struct common_speculative_impl_draft_mtp", 1)[0]
            self.assertNotIn("effective_n_max", eagle3_text)
            self.assertNotIn("last_n_draft", eagle3_text)
            before = {p: (root / p).read_text() for p in ("common/common.h", "common/arg.cpp", "common/speculative.cpp")}
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(before, {p: (root / p).read_text() for p in before})

    def test_noise_stripped_comments_and_strings_do_not_define_anchors(self):
        td, root = self._tree(cosmetic_noise=True)
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])

    def test_missing_depth_anchor_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "common/speculative.cpp"
            mtp_only = _SPEC.replace("params.n_max <= (int) result.size()", "false", 1)
            # First occurrence is eagle3; remove only MTP's depth cap so the MTP-only
            # lookahead has no valid match while the eagle3 lookalike remains.
            mtp_only = mtp_only.replace("params.n_max <= (int) result.size()", "false", 1)
            path.write_text(mtp_only, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
