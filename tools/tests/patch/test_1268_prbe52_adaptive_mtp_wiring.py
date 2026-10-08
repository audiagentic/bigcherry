"""Mechanics tests for 1268_prbe52_adaptive_mtp_wiring."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch import rebase as patch_rebase  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1268_prbe52_adaptive_mtp_wiring/patch.py"
_V = paths.llama_root()
_FILES = (
    "common/common.h",
    "common/arg.cpp",
    "common/speculative.cpp",
    "common/speculative.h",
    "tools/server/server-context.cpp",
)
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
_SPEC_H = """int32_t common_speculative_n_max(const common_speculative * spec);
"""
_SERVER = """    int get_n_draft_max() const {
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\\n", n_draft_max);
        return n_draft_max;
    }

    bool load_model() {
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }
    }
"""
_SPEC = """#include <algorithm>
#include <cassert>

struct common_speculative_impl {
    int32_t n_max = 4;
    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;
};
struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_impl>> impls;
};
int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

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
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());

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
            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!params.probabilistic) {
                dp.result_q = nullptr;
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q) {
                spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (!dp.result_q) {
                common_sampler_reset(smpls[seq_id].get());
            }

            const int32_t idx = batch.add(dp.id_last, dp.pos0, seq_id, true);
            batch.set_embd(idx, { pending_h[seq_id].data(), 1, (size_t) n_embd });

                result.push_back(id);

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                    if (bc_front <= result.size() && dp.n_tail <= 0) {
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

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
};
"""


class Patch1268Mechanics(unittest.TestCase):
    def _tree(self, *, cosmetic_noise=False):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "common").mkdir(parents=True)
        (root / "tools/server").mkdir(parents=True)
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
        (root / "common/speculative.h").write_text(_SPEC_H, encoding="utf-8")
        (root / "tools/server/server-context.cpp").write_text(_SERVER, encoding="utf-8")
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
            self.assertNotIn("last_n_draft", text)
            self.assertIn("this->params.n_min_adaptive < this->params.n_min", text)
            self.assertNotIn("adaptive MTP depth cannot be combined with BIGCHERRY_MTP_AHEAD=1", text)
            self.assertIn("dp.n_tail <= 0", text)
            self.assertIn("n_rows - 1", text)
            self.assertEqual(text.count("adaptive_state[seq_id].update"), 1)
            self.assertIn("bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override", text)
            self.assertIn("void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override", text)
            self.assertIn("std::memcpy(data.data(), h.data(), data.size())", text)
            self.assertIn("std::memcpy(h.data(), data.data(), data.size())", text)
            self.assertIn("current_n_max(llama_seq_id seq_id) const override", text)
            self.assertIn("event=depth_change", text)
            self.assertEqual(text.count("bc_front_cap <= result.size()"), 1)
            self.assertIn("if (params.n_max <= (int) result.size()) {", text)
            self.assertIn("if (dp.result->size() < (size_t) params.n_min) {", text)
            self.assertIn("common_speculative_n_max(const common_speculative * spec, llama_seq_id seq_id)", text)
            self.assertIn("common_speculative_n_max(const common_speculative * spec, llama_seq_id seq_id)",
                          (root / "common/speculative.h").read_text())
            server = (root / "tools/server/server-context.cpp").read_text()
            self.assertIn("common_speculative_n_max(spec, id)", server)
            self.assertIn("requested speculation must never silently degrade", server)
            self.assertNotIn("if (params_base.speculative.has_synth())", server)
            eagle3_text = text.split("struct common_speculative_impl_draft_mtp", 1)[0]
            self.assertNotIn("n_min_adaptive", eagle3_text)
            self.assertNotIn("last_n_draft", eagle3_text)
            before = {p: (root / p).read_text() for p in (
                "common/common.h", "common/arg.cpp", "common/speculative.cpp",
                "common/speculative.h", "tools/server/server-context.cpp",
            )}
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
            target = "bc_front <= result.size()"
            self.assertEqual(_SPEC.count(target), 1)
            broken = _SPEC.replace(target, "false", 1)
            path.write_text(broken, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


@unittest.skipUnless(all((_V / rel).exists() for rel in _FILES), "pinned vendor checkout not present")
class Patch1268ProductionComposition(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for rel in _FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / rel, root / rel)
        return root

    def _apply_production_and_1255(self, root):
        # Exercise the exact selector/apply path used by:
        #   patch-rebase-check --source bigcherry --experiment adaptive-mtp-no1210
        selected = patch_rebase.resolve_selection(
            source_name="bigcherry",
            all_patches=False,
            experiment="adaptive-mtp-no1210",
        )
        ids = [m.patch_id for m in selected.modules]
        self.assertIn("1321_mtp_ahead_primitives", ids)
        self.assertIn("1255_nro06_adaptive_mtp_depth", ids)
        self.assertIn("1268_prbe52_adaptive_mtp_wiring", ids)
        self.assertLess(ids.index("1321_mtp_ahead_primitives"), ids.index("1268_prbe52_adaptive_mtp_wiring"))
        self.assertLess(ids.index("1255_nro06_adaptive_mtp_depth"), ids.index("1268_prbe52_adaptive_mtp_wiring"))

        texts = patch_rebase._overlay_texts()
        overlay_paths = frozenset(texts)
        found = False
        for module in selected.modules:
            if module.patch_id == "1268_prbe52_adaptive_mtp_wiring":
                found = True
                break
            probe = patch_rebase.probe_patch(
                module,
                _V,
                texts,
                context_lines=3,
                previous_revision=None,
                revision="mechanics-test",
                overlay_paths=overlay_paths,
            )
            self.assertIn(
                probe.status,
                (patch_rebase.STATUS_CLEAN, patch_rebase.STATUS_CLEAN_NOOP),
                (module.patch_id, probe.to_dict()),
            )
        self.assertTrue(found)
        for rel in _FILES:
            if rel in texts:
                (root / rel).write_text(texts[rel], encoding="utf-8")

    def test_full_production_then_1268_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply_production_and_1255(root)
            before = (root / "common/speculative.cpp").read_text(encoding="utf-8")
            self.assertIn("const size_t bc_front", before)
            self.assertIn("if (bc_front <= result.size() && dp.n_tail <= 0)", before)

            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(x.ok for x in first), [e.detail for x in first for e in x.failed])
            spec = (root / "common/speculative.cpp").read_text(encoding="utf-8")
            self.assertIn("size_t bc_front_cap = bc_front;", spec)
            self.assertIn("bc_front_cap = std::min<size_t>", spec)
            self.assertIn("if (bc_front_cap <= result.size() && dp.n_tail <= 0)", spec)
            self.assertNotIn("if (params.n_max <= (int) result.size())", spec.split("struct common_speculative_impl_draft_mtp", 1)[1])

            snapshot = {rel: (root / rel).read_text(encoding="utf-8") for rel in _FILES}
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(x.ok for x in second), [e.detail for x in second for e in x.failed])
            self.assertEqual(snapshot, {rel: (root / rel).read_text(encoding="utf-8") for rel in _FILES})


if __name__ == "__main__":
    unittest.main()
