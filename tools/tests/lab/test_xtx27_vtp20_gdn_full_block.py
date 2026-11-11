"""Card-free VTP20 original BF16 full GDN GPU-block provenance and safety tests."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))

import xtx27_vtp20_gdn_full_block_gpu as full


class OriginalCheckpointGdnFullBlockTests(unittest.TestCase):
    def test_default_never_launches_gpu_and_reports_limits(self):
        with (redirect_stdout(io.StringIO()) as output,
              patch.object(full.subprocess, "run",
                           side_effect=AssertionError("would execute external command"))):
            self.assertEqual(full.main([]), 0)
        doc = json.loads(output.getvalue())
        self.assertEqual(doc["phase"], "card_free")
        self.assertTrue(doc["not_full_model"])
        self.assertTrue(doc["not_native_radiance"])
        self.assertIsNone(doc["original_reference_sources"])

    def test_all_original_checkpoint_input_oracles_required(self):
        for cli in (
            ["--phase", "gpu"],
            ["--phase", "gpu", "--run"],
            ["--phase", "compile"],
            ["--iterations", "0"],
            ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                full.main(cli)

    def test_fail_closed_on_changed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = pathlib.Path(tmp) / "incomplete.json"
            wrong.write_text("{}")
            with self.assertRaises((KeyError, ValueError)):
                full.validate(wrong, wrong, wrong, wrong, wrong, wrong)

    def test_three_gpu_real_checkpoint_full_block_chain_source(self):
        src = full.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> KEYS{4,4,8}",
            "constexpr std::array<int,3> VALUES{12,12,24}",
            "project_original_bf16<<<",
            "conv4<<<",
            "qk_norm<<<",
            "delta_qwen<<<",
            "gdn_rank_to_out_bf16<<<",
            "gdn_out_projection<<<",
            "output[(static_cast<size_t>(h)*rows+t)*H+e]",
            "out[i]=bf16rn(in[(static_cast<size_t>(h)*rows+t)*H+d])",
            "hipHostMallocPortable",
            "for(int i=0;i<3;++i)launch(ranks[i]);",
            "v+=ranks[1].pinned_partial[i]",
            "v+=ranks[2].pinned_partial[i]",
            "pinned_final[i]=round_host(fb16host(round_host(v))+fb16host(residual[i]))",
            "VTP20 three GPU broadcast did not agree bitwise",
            "VTP20 full checkpoint-GDN output differs from independent CPU oracle",
            "VTP20 final BF16 residual exceeds independent CPU tolerance",
        ):
            self.assertIn(token, src)
        self.assertNotIn("hipMemcpyPeer", src)
        self.assertNotIn("hipDeviceEnablePeerAccess", src)

    def test_same_source_sha_reference_modules_as_validated_partial_stages(self):
        self.assertTrue(full.SOURCE.is_file())
        self.assertEqual(set(full.SOURCES), {"gdn_full_block"})
        self.assertTrue(full.PURPOSE.startswith("VTP20"))
        self.assertTrue(full.RECEIPT_TYPE.startswith("VTP20"))
        self.assertIn("one-layer original-weight microkernel proof", full.__doc__)
        self.assertIn("NOT 64-layer model", full.__doc__)
        self.assertIsNotNone(full.core_gpu.validate)
        self.assertIsNotNone(full.output_gpu.validate)


if __name__ == "__main__":
    unittest.main()
