"""Upstream backport: make -funsafe-math-optimizations opt-in for HIP builds.

Cherry-picked from unmerged upstream PR
(https://github.com/ggml-org/llama.cpp/pull/26696, still ``OPEN`` as of this
writing). The PR's own CI test names the exact failure mode (bug ID
AIESW-40114): ``-funsafe-math-optimizations`` reassociates floating-point
reductions, which can flip a greedy argmax -- breaking the byte-identical
output guarantee MTP speculative decoding requires at temperature 0. Upstream
reports this confirmed on gfx1151 (RDNA3.5) and untested elsewhere.

This project's own tree had the flag unconditionally on for every HIP build
(``ggml/src/ggml-hip/CMakeLists.txt``, previously no gate at all). Verified
directly on hardware before backporting rather than trusting the gfx1151-only
upstream evidence: on gfx1201 (RDNA4), a temp-0 MTP speculative completion
was byte-identical to the non-speculative baseline across the same build.
On gfx1100 (RDNA3, this project's own local hardware), the same test
diverged -- both completions agreed word-for-word through most of the
response and then produced a genuinely different closing sentence, while two
independent non-speculative baseline runs on the same hardware were
byte-identical to each other. That rules out general run-to-run noise: the
divergence is specifically speculative-vs-non-speculative, reproducing the
PR's own described mechanism on a third architecture upstream never tested.

Adds ``GGML_HIP_UNSAFE_MATH`` (default OFF, matching upstream's default and
this project's own precedent for backported CMake toggles) gating the flag.
An OFF build is now IEEE-conformant by default; the fast-math speedup is
still available via ``-DGGML_HIP_UNSAFE_MATH=ON`` for workloads that don't
run MTP and want the throughput back.
"""

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"

PATCH = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="Gate -funsafe-math-optimizations behind GGML_HIP_UNSAFE_MATH "
                "(default OFF) -- confirmed on hardware to break MTP temp-0 "
                "determinism on gfx1100 (upstream PR #26696)",
    edits=(
        Edit(
            id="hip-unsafe-math-opt-in",
            # Anchored on the code line only -- the preceding comment is blanked
            # in the noise-stripped copy the patcher matches against (CMake
            # dialect treats `#` comments as noise), so it cannot be part of the
            # anchor even though it stays in the file, unchanged, right above.
            anchor=r"^set\(CMAKE_HIP_FLAGS \"\$\{CMAKE_HIP_FLAGS\} "
                   r"-funsafe-math-optimizations\"\)",
            rationale="the unconditional fast-math flag for HIP builds",
            mode="replace",
            text=(
                "# bigcherry: opt-in, not on by default -- confirmed on real gfx1100\n"
                "# hardware that this flag breaks MTP speculative decoding's temp-0\n"
                "# byte-identical guarantee (upstream PR #26696, bug AIESW-40114).\n"
                "# gfx1201 tested clean; gfx1100 reproduced the divergence directly.\n"
                "option(GGML_HIP_UNSAFE_MATH "
                "\"ggml: use -funsafe-math-optimizations for HIP (breaks MTP determinism "
                "on some architectures)\" OFF)\n"
                "if (GGML_HIP_UNSAFE_MATH)\n"
                "    set(CMAKE_HIP_FLAGS \"${CMAKE_HIP_FLAGS} -funsafe-math-optimizations\")\n"
                "endif()"
            ),
            guard=r"option\(GGML_HIP_UNSAFE_MATH",
        ),
    ),
)

PATCHES = [PATCH]
