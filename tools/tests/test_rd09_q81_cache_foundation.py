"""RD09 stage 1: source-contract tests for the per-graph Q8_1
activation-quantization cache foundation
(src/ggml/src/ggml-cuda/hip-q81-cache.{h,cpp}).

Source-contract only, matching this repo's existing pattern for .cu/.cuh/.cpp
files under src/ggml/src/ggml-cuda (see test_hi99_tuner_config_macro.py) --
no HIP compiler is assumed available offline. This stage adds no caller in
mmvq.cu, so there is nothing to real-hardware-validate yet; these tests only
confirm the structural invariants the design (docs/planning/active/
rdna-boost-experiments/RD09.md, dev-gpt-agent session ses_76b0fef0c94c434a,
req_60a41664e0de43d6) requires before any wiring happens:

  - the cache key includes the exact view data address/offset (the fork
    source's own key omits this -- a real same-root/different-offset
    false-hit bug this stage must not reproduce)
  - backing storage never reallocates/relocates an existing slab (growth is
    always an additional allocation, appended, never a realloc/copy/free of
    an existing one) -- required for HIP/CUDA graph-capture pointer
    stability
  - the runtime mode gate defaults to off and is independent of
    GGML_HIP_DISPATCH_MODE
  - stage 1 adds no caller: mmvq.cu (the one real call site of
    quantize_row_q8_1_cuda) must not reference this cache yet
  - the new files are wired into the HIP build via patches/
    1235_rd09_q81_activation_cache_foundation.py, not via
    0100_cmake_options.py (keeping the production build surface untouched
    for this first, zero-behavioral-risk slice)
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-q81-cache.h"
IMPL = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-q81-cache.cpp"
MMVQ = ROOT / "vendor" / "llama.cpp" / "ggml" / "src" / "ggml-cuda" / "mmvq.cu"
PATCH_PATH = ROOT / "patches" / "1235_rd09_q81_activation_cache_foundation.py"
CMAKE_0100 = ROOT / "patches" / "0100_cmake_options.py"

sys.path.insert(0, str(ROOT / "tools"))


class Rd09CacheFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header_src = HEADER.read_text(encoding="utf-8")
        cls.impl_src = IMPL.read_text(encoding="utf-8")
        cls.mmvq_src = MMVQ.read_text(encoding="utf-8")
        cls.patch_src = PATCH_PATH.read_text(encoding="utf-8")

    def test_files_exist(self):
        self.assertTrue(HEADER.is_file())
        self.assertTrue(IMPL.is_file())

    def test_key_struct_has_the_offset_field_the_fork_omits(self):
        # The specific field whose absence is the fork's real bug: the exact
        # view byte start, not just the view-root tensor pointer.
        m = re.search(r"struct ggml_hip_q81_cache_key \{(.*?)\n\};", self.header_src, re.DOTALL)
        self.assertIsNotNone(m, "could not find ggml_hip_q81_cache_key struct body")
        body = m.group(1)
        self.assertIn("view_root", body)
        self.assertIn("const void * data", body,
                      "key must carry the exact view data address -- its absence "
                      "is the fork's real same-root/different-offset false-hit bug")

    def test_key_equality_compares_every_field(self):
        m = re.search(r"bool operator==\(const ggml_hip_q81_cache_key & other\) const \{(.*?)\n    \}",
                       self.header_src, re.DOTALL)
        self.assertIsNotNone(m)
        body = m.group(1)
        for field in ("generation", "view_root", "data", "stream_no",
                      "ne0", "ne0_padded", "ne1", "ne2", "ne3", "s1", "s2", "s3"):
            self.assertRegex(body, rf"\b{field}\b\s*==\s*other\.{field}\b",
                              f"operator== must compare {field!r} against other.{field!r}")

    def test_mode_env_var_and_default(self):
        self.assertIn('std::getenv("GGML_HIP_Q8_1_CACHE_MODE")', self.impl_src)
        # off is the fallback for null AND for any unrecognized string --
        # fail closed on a typo rather than silently enabling the
        # experimental path.
        parse_fn = re.search(r"ggml_hip_q81_cache_mode parse_mode\(const char \* s\) \{(.*?)\n\}",
                              self.impl_src, re.DOTALL)
        self.assertIsNotNone(parse_fn)
        body = parse_fn.group(1)
        self.assertIn("GGML_HIP_Q81_CACHE_OFF", body)
        self.assertIn('strcmp(s, "on")', body)
        self.assertIn('strcmp(s, "verify")', body)
        # last statement before the closing brace must be the off fallback
        self.assertTrue(body.rstrip().endswith("return GGML_HIP_Q81_CACHE_OFF;"))

    def test_mode_is_checked_once_not_per_call(self):
        # Same checked-once-static shape as ggml_hip_tuner's own env lookups
        # (hip-autotune-tuner.cu) -- not re-read on every call.
        self.assertRegex(
            self.impl_src,
            r"static const ggml_hip_q81_cache_mode mode = parse_mode\(std::getenv\(",
        )

    def test_stats_env_var_exists_and_is_separate_from_mode(self):
        self.assertIn('std::getenv("GGML_HIP_Q8_1_CACHE_STATS")', self.impl_src)

    def test_mode_is_independent_of_dispatch_mode(self):
        # RD09 must not fold its gate into GGML_HIP_DISPATCH_MODE: that
        # would couple an activation-materialization optimization to an
        # unrelated dispatch-selection experiment and poison the native
        # control other work relies on. A design-rationale comment
        # mentioning the name is fine (and present, deliberately); the
        # actual gate logic must never read or branch on it.
        mode_fn = re.search(
            r"ggml_hip_q81_cache_mode ggml_hip_q81_cache_mode_get\(\) \{.*?\n\}",
            self.impl_src, re.DOTALL,
        )
        self.assertIsNotNone(mode_fn)
        self.assertNotIn("GGML_HIP_DISPATCH_MODE", mode_fn.group(0))

    def test_slab_growth_never_reallocates_or_frees_an_existing_slab(self):
        # The whole point of "stable slabs": ggml_hip_q81_cache_reserve must
        # grow by appending a brand-new allocation, never by realloc'ing or
        # freeing an existing one (which would invalidate any pointer a HIP
        # graph has already captured).
        reserve_fn = re.search(
            r"ggml_hip_q81_cache_reservation ggml_hip_q81_cache_reserve\(.*?\n\}",
            self.impl_src, re.DOTALL,
        )
        self.assertIsNotNone(reserve_fn)
        body = reserve_fn.group(0)
        self.assertNotIn("realloc", body)
        self.assertNotIn("hipFree", body,
                          "reserve() must never free an existing slab -- growth is "
                          "always an additional allocation")
        self.assertIn("slabs.push_back", body)

    def test_hipfree_appears_only_in_destructor_and_test_reset(self):
        # hipFree must exist (slabs are eventually released), but only in
        # the two places that are safe: the cache's own destructor
        # (process/context teardown) and the explicit test-only reset --
        # never on the hot reserve/find/publish path.
        self.assertEqual(self.impl_src.count("hipFree("), 2,
                          "expected exactly 2 hipFree call sites (destructor + reset_for_test)")

        destructor = re.search(r"~ggml_hip_q81_cache\(\) \{.*?\n    \}", self.impl_src, re.DOTALL)
        self.assertIsNotNone(destructor)
        self.assertIn("hipFree(", destructor.group(0))

        reset_fn = re.search(
            r"void ggml_hip_q81_cache_reset_for_test\(.*?\n\}", self.impl_src, re.DOTALL,
        )
        self.assertIsNotNone(reset_fn)
        self.assertIn("hipFree(", reset_fn.group(0))

        # The two functions above account for every hipFree call site;
        # nothing else in the file may call it.
        other = self.impl_src.replace(destructor.group(0), "").replace(reset_fn.group(0), "")
        self.assertNotIn("hipFree(", other)

    def test_reserve_never_grows_while_capture_active(self):
        reserve_fn = re.search(
            r"ggml_hip_q81_cache_reservation ggml_hip_q81_cache_reserve\(.*?\n\}",
            self.impl_src, re.DOTALL,
        )
        body = reserve_fn.group(0)
        self.assertIn("capture_active", body)
        self.assertIn("capture_capacity_bypasses", body)
        # The capture_active check must come before any hipMalloc in this
        # function, so growth during capture is refused, not attempted.
        capture_pos = body.index("if (cache.capture_active)")
        malloc_pos = body.index("hipMalloc")
        self.assertLess(capture_pos, malloc_pos,
                         "capture_active must gate growth BEFORE the allocation, not after")

    def test_find_reserve_publish_are_separate_calls(self):
        # A miss must never become a visible cache entry before the
        # producer kernel is at least enqueued -- find/reserve/publish must
        # be three distinct functions, not one that publishes on reserve.
        for name in ("ggml_hip_q81_cache_find", "ggml_hip_q81_cache_reserve",
                     "ggml_hip_q81_cache_publish"):
            self.assertIn(f"{name}(", self.header_src)
        reserve_fn = re.search(
            r"ggml_hip_q81_cache_reservation ggml_hip_q81_cache_reserve\(.*?\n\}",
            self.impl_src, re.DOTALL,
        ).group(0)
        self.assertNotIn("entries[", reserve_fn,
                          "reserve() must not publish into the entries map itself")
        publish_fn = re.search(
            r"void ggml_hip_q81_cache_publish\(.*?\n\}", self.impl_src, re.DOTALL,
        ).group(0)
        self.assertIn("entries[key] = reservation.ptr", publish_fn)

    def test_begin_generation_clears_entries_but_keeps_slabs(self):
        fn = re.search(
            r"void ggml_hip_q81_cache_begin_generation\(.*?\n\}", self.impl_src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group(0)
        self.assertIn("cache.entries.clear()", body)
        self.assertIn("cache.cursor = 0", body)
        self.assertNotIn("hipFree", body,
                          "begin_generation must never free backing slab memory -- "
                          "only reset the logical bump cursor and clear the map")
        self.assertNotIn("slabs.clear()", body)

    def test_stage1_adds_no_caller_in_mmvq(self):
        # This stage must be entirely inert: nothing in mmvq.cu (the sole
        # existing call site of quantize_row_q8_1_cuda) may reference the
        # new cache yet. Wiring it in is a separate, future stage 2 patch.
        self.assertNotIn("hip-q81-cache", self.mmvq_src)
        self.assertNotIn("ggml_hip_q81_cache", self.mmvq_src)
        # Exactly one quantize_row_q8_1_cuda call site still exists,
        # unmodified -- this is the seam stage 2 will change.
        self.assertEqual(self.mmvq_src.count("quantize_row_q8_1_cuda("), 1)

    def test_patch_is_isolated_and_untested(self):
        self.assertIn('GROUP = "rdna-boosts"', self.patch_src)
        self.assertIn('STATE = "untested"', self.patch_src)

    def test_patch_does_not_touch_production_cmake_options(self):
        # gpt's design explicitly called for NOT modifying
        # 0100_cmake_options.py or the production build surface for this
        # first slice -- the new files must be wired in by RD09's own
        # patch only.
        self.assertNotIn("hip-q81-cache", CMAKE_0100.read_text(encoding="utf-8"))
        self.assertIn("hip-q81-cache.cpp", self.patch_src)

    def test_patch_targets_only_the_hip_cmakelists(self):
        from bigcherry import patchset as _patchset  # noqa: E402

        module = _patchset._load_module(PATCH_PATH)
        patches = module.PATCHES
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].path, "ggml/src/ggml-hip/CMakeLists.txt")

    def test_provenance_cites_the_rebased_commit_not_the_stale_short_hash(self):
        from bigcherry import patchset as _patchset  # noqa: E402

        module = _patchset._load_module(PATCH_PATH)
        provenance = module.PROVENANCE
        self.assertEqual(provenance["fork-commit"], "299f6eaf73b5eeb888bd94eaa66122d003136e6a")
        self.assertEqual(provenance["original-commit"], "ff6fde5046ffb86672e05da640d2bfb20d4bfdfc")
        self.assertTrue(provenance["adaptations"], "must document the departures from the fork source")


if __name__ == "__main__":
    unittest.main()
