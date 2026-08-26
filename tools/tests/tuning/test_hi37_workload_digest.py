"""HI37 Part 2: source-contract tests for the workload digest/label.

The workload digest is a blake2b hash over the sorted set of observed
signature digests (presence, not call frequency) -- an identity for "what
shapes did this model produce" that needs no model metadata crossing the
ggml/llama boundary. It is advisory only: recorded and reported in the
measurements header, never gating a cache load.

Source-contract only, matching this file's existing testing pattern -- no
HIP compiler is available in this dev environment.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
SIGNATURE_H = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-signature.h"


class Hi37WorkloadDigestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")

    def test_blake2b_header_is_included(self):
        self.assertIn('#include "hip-autotune-blake2b.h"', self.tuner)

    def test_digest_is_over_sorted_signature_digests_not_insertion_order(self):
        self.assertIn(
            "std::sort(sorted.begin(), sorted.end(),\n"
            "              [](const ggml_hip_digest & a, const ggml_hip_digest & b) {\n"
            "                  return memcmp(a.bytes, b.bytes, GGML_HIP_DIGEST_BYTES) < 0;",
            self.tuner,
        )

    def test_digest_hashes_signature_digests_not_dispatch_or_call_counts(self):
        # Presence, not frequency: pushing entry.second.signature_digest
        # (never a call count) is what makes the digest stable across two
        # runs of the same model at different token counts.
        self.assertIn("sorted.push_back(entry.second.signature_digest);", self.tuner)

    def test_person_string_matches_the_python_mirror(self):
        self.assertIn('"llama-workload"', self.tuner)

    def test_workload_label_read_from_the_documented_env_var(self):
        self.assertIn('getenv("GGML_HIP_TUNE_WORKLOAD")', self.tuner)

    def test_header_emits_both_workload_and_workload_label(self):
        self.assertIn('"\\"workload\\":\\"%s\\",\\"workload_label\\":\\"%s\\"}\\n"', self.tuner)
        self.assertIn("ggml_hip_digest_hex(workload).c_str(),", self.tuner)

    def test_missing_label_env_var_emits_empty_string_not_null(self):
        # A missing GGML_HIP_TUNE_WORKLOAD must still produce valid JSON
        # (workload_label:"") rather than a null/garbage %s argument.
        self.assertIn('workload_label_env ? workload_label_env : "");', self.tuner)


class Hi37DiagnosticsStructRemovedTests(unittest.TestCase):
    """HI37 notes: ggml_hip_diagnostics was declared since HI05, never
    instantiated anywhere in this tree, and could only be populated by
    plumbing model metadata across the ggml/llama boundary this overlay
    exists to avoid -- deleted rather than left as a struct that was either
    a plan or a lie. Its intended purpose is now served by the workload
    digest (this file) and dispatch_db's real, populated
    observation.diagnostics_json column."""

    def test_struct_is_gone(self):
        text = SIGNATURE_H.read_text(encoding="utf-8")
        self.assertNotIn("struct ggml_hip_diagnostics", text)


if __name__ == "__main__":
    unittest.main()
