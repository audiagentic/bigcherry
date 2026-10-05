"""Universal patches default on; runtime profiles hold only the flags they switch (owner rule, 2026-10-05).

The fused Q8_1 decode path (1307 cache + 1309/1310/1311/1312/1313 writers) helps any quantized model, so its flags
are off switches: unset means on. A model profile that listed them would hide the benefit from every other model.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

# flag -> patches that read it with getenv
_DEFAULT_ON = {
    "BIGCHERRY_RMS_Q81": ("1309_rms_norm_mul_q81",),
    "BIGCHERRY_ACT_Q81": ("1310_act_q81", "1313_scale_act_fuse"),
    "BIGCHERRY_HC_Q81": ("1311_hc_pre_q81",),
    "BIGCHERRY_SCALE_ACT_FUSE": ("1313_scale_act_fuse",),
}
_UNIVERSAL_FLAGS = tuple(_DEFAULT_ON) + ("GGML_HIP_Q8_1_CACHE_MODE",)


class UniversalDefaults(unittest.TestCase):
    def test_fusion_flags_are_off_switches(self):
        for flag, patch_ids in _DEFAULT_ON.items():
            for patch_id in patch_ids:
                src = (_REPO / "patches" / patch_id / "patch.py").read_text(encoding="utf-8")
                self.assertIn(f'getenv("{flag}") == nullptr || atoi(getenv("{flag}")) != 0', src, (flag, patch_id))
                self.assertNotIn(f'getenv("{flag}") != nullptr && atoi(', src, (flag, patch_id))

    def test_q81_cache_is_on_when_unset_and_fails_closed_on_a_typo(self):
        src = (_REPO / "src/ggml/src/ggml-cuda/hip-q81-cache.cpp").read_text(encoding="utf-8")
        body = re.search(r"ggml_hip_q81_cache_mode parse_mode\(const char \* s\) \{(.*?)\n\}", src, re.DOTALL).group(1)
        self.assertRegex(body, r"if \(s == nullptr\) \{\s+return GGML_HIP_Q81_CACHE_ON;")
        self.assertTrue(body.rstrip().endswith("return GGML_HIP_Q81_CACHE_OFF;"))

    def test_documented_defaults_say_on(self):
        for patch_id, flag in (("1307_q81_activation_cache_mmvq", "GGML_HIP_Q8_1_CACHE_MODE"),
                               ("1309_rms_norm_mul_q81", "BIGCHERRY_RMS_Q81"),
                               ("1310_act_q81", "BIGCHERRY_ACT_Q81"),
                               ("1311_hc_pre_q81", "BIGCHERRY_HC_Q81"),
                               ("1312_mul_q81", "BIGCHERRY_ACT_Q81"),
                               ("1313_scale_act_fuse", "BIGCHERRY_SCALE_ACT_FUSE")):
            src = (_REPO / "patches" / patch_id / "patch.py").read_text(encoding="utf-8")
            doc = re.search(rf"EnvDoc\('{flag}', '[^']+', '([^']+)'", src)
            self.assertIsNotNone(doc, (patch_id, flag))
            self.assertTrue(doc.group(1).startswith(("on", "1")), (patch_id, flag, doc.group(1)))

    def test_profiles_do_not_list_universal_flags(self):
        for ini in sorted((_REPO / "src/profile").glob("*.ini")):
            for line in ini.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for flag in _UNIVERSAL_FLAGS:
                    self.assertFalse(re.match(rf"\s*{flag}\s*=", line), f"{ini.name}: {line.strip()}")

    def test_model_profiles_include_no_generic_bundle(self):
        text = (_REPO / "src/profile/flashnext.ini").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?m)^@")


if __name__ == "__main__":
    unittest.main()
