"""HI163: extract the REAL L2 cache logic from the production dispatch file
(not a hand-maintained mirror -- see test_runtime_fingerprint_compiled.py for
the same pattern) and prove its collision-safety invariant: a fingerprint
collision costs an extra compare, never a wrong binding. dev-gpt-agent's
sign-off (req_535fc57a30964a0d) required this be a real, compiled, run test,
not a code-review-only claim.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src/ggml/src/ggml-cuda"


def _extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.S)
    if not match:
        raise AssertionError(f"pattern not found (source drifted?): {pattern!r}")
    return match.group(0)


class L2CacheCollisionSafetyTests(unittest.TestCase):
    def test_l2_fingerprint_collision_never_returns_wrong_binding(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")

        types = (ROOT / "hip-autotune-types.h").read_text(encoding="utf-8")
        dispatch = (ROOT / "hip-autotune-dispatch.cu").read_text(encoding="utf-8")

        hardware_key = _extract(r"struct ggml_hip_hardware_key_v1 \{.*?\n\};", types)
        signature = _extract(r"struct ggml_hip_dispatch_signature_v1 \{.*?\n\};", types)
        fingerprint = _extract(
            r"static inline uint64_t signature_fingerprint\(.*?\n\}", dispatch)
        l2_entry = _extract(r"struct L2Entry \{.*?\n\};", dispatch)
        bindings_map = _extract(
            r"std::unordered_map<uint64_t, std::vector<L2Entry>> g_bindings;", dispatch)
        hw_equal = _extract(
            r"static bool hardware_key_equal\(.*?\n\}", dispatch)
        l2_find = _extract(
            r"static bool l2_find_locked\(.*?\n\}", dispatch)
        l2_insert = _extract(
            r"static void l2_insert_locked\(.*?\n\}", dispatch)

        # A stand-in for Binding: the real struct also carries a candidate
        # pointer/variant/transform, none of which this invariant needs --
        # only that lookups return the RIGHT entry, not necessarily its
        # exact real payload type.
        binding_stub = "struct Binding { int tag; };"

        source = "\n".join([
            "#include <cstdint>",
            "#include <cstring>",
            "#include <mutex>",
            "#include <thread>",
            "#include <unordered_map>",
            "#include <vector>",
            hardware_key,
            signature,
            binding_stub,
            fingerprint,
            l2_entry,
            bindings_map,
            "std::mutex g_bindings_mutex;",
            hw_equal,
            l2_find,
            l2_insert,
        ])
        source += """
static ggml_hip_dispatch_signature_v1 make_sig(int64_t ne0_0) {
    ggml_hip_dispatch_signature_v1 s{};
    s.schema_version = 1;
    s.op = 42;
    s.src0_type = 8;
    s.ne0[0] = ne0_0;
    return s;
}

static ggml_hip_hardware_key_v1 make_hw(uint16_t arch) {
    ggml_hip_hardware_key_v1 hw{};
    hw.schema_version = 1;
    hw.architecture_code = arch;
    hw.wave_size = 32;
    hw.compute_units = 96;
    return hw;
}

static void reset() {
    std::lock_guard<std::mutex> lock(g_bindings_mutex);
    g_bindings.clear();
}

int main() {
    // 1) Deliberate fingerprint collision across two different (hw, sig)
    //    pairs must not cross-contaminate.
    reset();
    {
        const uint64_t X = 0xDEADBEEFCAFEBABEull;
        const auto sig_a = make_sig(100);
        const auto sig_b = make_sig(200);
        const auto hw    = make_hw(1100);
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        l2_insert_locked(X, hw, sig_a, Binding{111});
        l2_insert_locked(X, hw, sig_b, Binding{222});
        Binding out{};
        if (!(l2_find_locked(X, hw, sig_a, &out) && out.tag == 111)) return 1;
        if (!(l2_find_locked(X, hw, sig_b, &out) && out.tag == 222)) return 2;
        const auto sig_c = make_sig(300);
        if (l2_find_locked(X, hw, sig_c, &out)) return 3;
    }

    // 2) Same signature, different hardware class -> miss.
    reset();
    {
        const uint64_t fp = signature_fingerprint(make_sig(42));
        const auto sig = make_sig(42);
        const auto hw_a = make_hw(1100);
        const auto hw_b = make_hw(1201);
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        l2_insert_locked(fp, hw_a, sig, Binding{1});
        Binding out{};
        if (l2_find_locked(fp, hw_b, sig, &out)) return 4;
        if (!(l2_find_locked(fp, hw_a, sig, &out) && out.tag == 1)) return 5;
    }

    // 3) Identical hardware CLASS from separately-constructed instances
    //    (no device ordinal in the key) shares one entry -- HI64's
    //    invariant, which HI163 must preserve.
    reset();
    {
        const auto sig = make_sig(7);
        const uint64_t fp = signature_fingerprint(sig);
        const auto hw_gpu0 = make_hw(1100);
        const auto hw_gpu1 = make_hw(1100);
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        l2_insert_locked(fp, hw_gpu0, sig, Binding{9});
        Binding out{};
        if (!(l2_find_locked(fp, hw_gpu1, sig, &out) && out.tag == 9)) return 6;
    }

    // 4) Duplicate exact insertion is a no-op (first-insert-wins).
    reset();
    {
        const auto sig = make_sig(55);
        const uint64_t fp = signature_fingerprint(sig);
        const auto hw = make_hw(1030);
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        l2_insert_locked(fp, hw, sig, Binding{1});
        l2_insert_locked(fp, hw, sig, Binding{2});
        Binding out{};
        if (!(l2_find_locked(fp, hw, sig, &out) && out.tag == 1)) return 7;
        if (g_bindings[fp].size() != 1) return 8;
    }

    // 5) Ordinary distinct-signature lookups still work.
    reset();
    {
        const auto sig1 = make_sig(1);
        const auto sig2 = make_sig(2);
        const auto hw = make_hw(1100);
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        l2_insert_locked(signature_fingerprint(sig1), hw, sig1, Binding{10});
        l2_insert_locked(signature_fingerprint(sig2), hw, sig2, Binding{20});
        Binding out{};
        if (!(l2_find_locked(signature_fingerprint(sig1), hw, sig1, &out) && out.tag == 10)) return 9;
        if (!(l2_find_locked(signature_fingerprint(sig2), hw, sig2, &out) && out.tag == 20)) return 10;
    }

    // 6) Concurrent duplicate insert yields exactly one entry.
    reset();
    {
        const auto sig = make_sig(999);
        const uint64_t fp = signature_fingerprint(sig);
        const auto hw = make_hw(1201);
        auto racer = [&](int tag) {
            std::lock_guard<std::mutex> lock(g_bindings_mutex);
            l2_insert_locked(fp, hw, sig, Binding{tag});
        };
        std::thread t1(racer, 1);
        std::thread t2(racer, 2);
        t1.join();
        t2.join();
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        if (g_bindings[fp].size() != 1) return 11;
        Binding out{};
        if (!(l2_find_locked(fp, hw, sig, &out) && (out.tag == 1 || out.tag == 2))) return 12;
    }

    return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "l2_invariant.cpp"
            exe = Path(directory) / "l2_invariant.exe"
            cpp.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [compiler, "-std=c++17", "-O0", "-pthread", str(cpp), "-o", str(exe)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = subprocess.run([str(exe)])
            self.assertEqual(run.returncode, 0,
                              f"invariant check failed (exit code {run.returncode})")


if __name__ == "__main__":
    unittest.main()
