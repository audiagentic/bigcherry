"""Universal patches default on; runtime profiles hold only the flags they switch (owner rule, 2026-10-05).

The fused Q8_1 decode path (1307 cache + 1309/1310/1311/1312/1313 writers) helps any quantized model, so its flags
are off switches: unset means on. A model profile that listed them would hide the benefit from every other model.
"""

from __future__ import annotations

import re
import importlib.util
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "tools"))
_Q81 = "1307_q81_activation_cache_mmvq"  # the merged Q8_1 family: cache, RMS, activation, HC_PRE and MUL producers

# flag -> patches whose inserted code reads it with getenv
_DEFAULT_ON = {
    "BIGCHERRY_RMS_Q81": (_Q81,),
    "BIGCHERRY_ACT_Q81": (_Q81, "1313_scale_act_fuse"),
    "BIGCHERRY_HC_Q81": (_Q81,),
    "BIGCHERRY_SCALE_ACT_FUSE": ("1313_scale_act_fuse",),
    "BIGCHERRY_FA_SPARSE": ("1334_hip_sparse_flash_attn",),
    "BIGCHERRY_PREFILL_PIPELINE": ("1359_prefill_pipeline",),
}
_UNIVERSAL_FLAGS = tuple(_DEFAULT_ON) + ("GGML_HIP_Q8_1_CACHE_MODE",)


def _module(patch_id):
    path = _REPO / "engines" / "llamacpp" / "patches" / patch_id / "patch.py"
    spec = importlib.util.spec_from_file_location("universal_" + patch_id[:4], path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inserted_text(patch_id):
    """Everything the package writes into the source tree (a merged package holds its parts as data, not as text)."""
    return "\n".join(edit.text or "" for file_patch in _module(patch_id).PATCHES for edit in file_patch.edits)


def _env_docs(patch_id):
    module = _module(patch_id)
    return tuple(getattr(module, "ENV_DOCS", ())) + tuple(getattr(module, "DOCUMENTED_ENV_DOCS", ()))


class UniversalDefaults(unittest.TestCase):
    def test_fusion_flags_are_off_switches(self):
        for flag, patch_ids in _DEFAULT_ON.items():
            for patch_id in patch_ids:
                src = _inserted_text(patch_id)
                self.assertIn(f'getenv("{flag}") == nullptr || atoi(getenv("{flag}")) != 0', src, (flag, patch_id))
                self.assertNotIn(f'getenv("{flag}") != nullptr && atoi(', src, (flag, patch_id))

    def test_q81_cache_is_on_when_unset_and_fails_closed_on_a_typo(self):
        src = (_REPO / "engines/llamacpp/overlay/ggml/src/ggml-cuda/hip-q81-cache.cpp").read_text(encoding="utf-8")
        body = re.search(r"ggml_hip_q81_cache_mode parse_mode\(const char \* s\) \{(.*?)\n\}", src, re.DOTALL).group(1)
        self.assertRegex(body, r"if \(s == nullptr\) \{\s+return GGML_HIP_Q81_CACHE_ON;")
        self.assertTrue(body.rstrip().endswith("return GGML_HIP_Q81_CACHE_OFF;"))

    def test_documented_defaults_say_on(self):
        for patch_id, flag in ((_Q81, "GGML_HIP_Q8_1_CACHE_MODE"),
                               (_Q81, "BIGCHERRY_RMS_Q81"),
                               (_Q81, "BIGCHERRY_ACT_Q81"),
                               (_Q81, "BIGCHERRY_HC_Q81"),
                               ("1313_scale_act_fuse", "BIGCHERRY_SCALE_ACT_FUSE")):
            defaults = [doc.default for doc in _env_docs(patch_id) if doc.name == flag]
            self.assertTrue(defaults, (patch_id, flag))
            for default in defaults:
                self.assertTrue(default.startswith(("on", "1")), (patch_id, flag, default))

    def test_profiles_do_not_list_universal_flags(self):
        for ini in sorted((_REPO / "engines/llamacpp/overlay/profile").glob("*.ini")):
            for line in ini.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for flag in _UNIVERSAL_FLAGS:
                    self.assertFalse(re.match(rf"\s*{flag}\s*=", line), f"{ini.name}: {line.strip()}")

    def test_model_profiles_include_no_generic_bundle(self):
        text = (_REPO / "engines/llamacpp/overlay/profile/flashnext.ini").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?m)^@")


if __name__ == "__main__":
    unittest.main()
