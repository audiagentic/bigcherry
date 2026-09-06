"""Check the production fingerprint against word loads for every storage byte."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "src/ggml/src/ggml-cuda"


class RuntimeFingerprintCompiledTests(unittest.TestCase):
    def test_alias_safe_hash_preserves_word_hash_and_field_sensitivity(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        types = (ROOT / "hip-autotune-types.h").read_text(encoding="utf-8")
        dispatch = (ROOT / "hip-autotune-dispatch.cu").read_text(encoding="utf-8")
        signature = re.search(r"struct ggml_hip_dispatch_signature_v1 \{.*?\n\};",
                              types, re.S).group(0)
        fingerprint = re.search(r"static inline uint64_t signature_fingerprint\(.*?\n\}",
                                dispatch, re.S).group(0)
        source = "#include <stdint.h>\n#include <string.h>\n" + signature + "\n" + fingerprint
        source += """
int main() {
    ggml_hip_dispatch_signature_v1 signature;
    memset(&signature, 0, sizeof(signature));
    const uint64_t zero = signature_fingerprint(signature);
    unsigned char * bytes = reinterpret_cast<unsigned char *>(&signature);
    for (size_t offset = 0; offset < sizeof(signature); ++offset) {
        bytes[offset] = 0x5a;
        uint64_t words[sizeof(signature) / sizeof(uint64_t)];
        memcpy(words, &signature, sizeof(signature));
        uint64_t expected = 1469598103934665603ull;
        for (uint64_t word : words) { expected ^= word; expected *= 1099511628211ull; }
        uint64_t actual = signature_fingerprint(signature);
        if (actual != expected || actual == zero) return 1;
        bytes[offset] = 0;
    }
    return signature_fingerprint(signature) != zero;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "fingerprint.cpp"
            exe = Path(directory) / "fingerprint.exe"
            cpp.write_text(source, encoding="utf-8")
            result = subprocess.run([compiler, "-std=c++17", "-O3", "-fstrict-aliasing",
                                     str(cpp), "-o", str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(subprocess.run([str(exe)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
