# Archived RCS13 large-K card-free admission report

Original ignored report was preserved locally with SHA-256 d4ce26e57bf6b5fdd63163f4c3d86d16e4125a88a9b3645e9d9230874cf61c42. Two machine-local scratch paths were normalized to a work-root placeholder; no tests were rerun during this relocation.

---

# RCS13 large-K card-free admission — 2026-11-18

## Result

PR: https://github.com/audiagentic/bigcherry/pull/1088
Branch: `test/rcs13-largek-model-admission`
Head: `e77c5b5af808cf028005d907043ecea94fb3367a`
Commit titles: `test(moe-remote): pin large-K model admission decisions`, `style(moe-remote): format large-K admission fixture`.
Auto-merge enabled. PR title check passed after replacing the original PR body (the first run failed on a work-root path in the PR description, not on code). Main remains untouched locally.

## Coverage inventory

Merged `gemv_geometry.h` and existing `gemv_geometry_test.cpp` already covered each large-K condition separately:
- variants' geometry below/at/above 2560, including K=2688 and index bijections at K=3072;
- paired parity 2560/3072 dynamic LDS;
- down K=3072 exceeding a 2560-sized LDS cap with fallback;
- hidden=4096 with generic GEMM available;
- no grouped GEMM, typed `Refuse`;
- zero, misalignment, overflow, static/dynamic LDS, occupancy and kernel absence.

New `test_large_k_model_stage_admission` ties the real model shape hidden=2560 and intermediate=3072 together: under an LDS budget fitting only the hidden projection, gate/up chooses GEMV and down chooses grouped GEMM; hidden=4096 can still use grouped GEMM; disabling generic fallback makes down return typed `Refuse`; a 3072-capable LDS budget admits a full-size down GEMV. All through the same `choose_stage` function consumed by production, no fake planner. Existing declared `r3_gemv_geometry_test` gets CTest `TIMEOUT 60`; no new target and no GPU.

Production source audit (`moe_remote_gpu.h`): `Runner::open` validates both hidden and intermediate through `set_shape` prior to lane preparation. In `enqueue_lane`, both projection choices are checked before stage kernels and `lane_publish_done`; a `Refuse` returns `-ENOTSUP` before stage publication. The card-free fixture proves selection and the refusal type, not a live startup/publish path or real GPU output.

## Local validation

Scratch source: `<work-root>/tmp/rcs13-largek-pr1088`.
Scratch component build: `<work-root>/tmp/rcs13-largek-pr1088-build/component`; CMake configured real moe-remote against an already-built, scratch-installed Radiance ABI with HIP compiler and `R3MOE_TARGETS=gfx1100`. No GPU use, hardware, or lease.
- `g++ -std=c++17 -Wall -Wextra -Werror` standalone geometry test: compiled, passed.
- `cmake --build ... --target r3_gemv_geometry_test`: passed.
- `ctest --test-dir ... -R '^r3_gemv_geometry_test$' --output-on-failure --timeout 60`: 1/1 passed.
- Additional declared card-free tests: `r3_moe_backend_test`, `r3_moe_fp8_ref_test`, `r3_worker_footprint_test`, `r3_v26_gpu_contract_test`, `r3_admission_ledger_test`, topology, SHM, policy input/live, share, heat, replay bootstrap, combine epoch, segment version, mailbox and owner generation: 19/19 passed; 20/20 overall.
- Partial broader component build encountered the shell tool's per-command time boundary compiling the generated serving codec. No declaration of a full-suite pass. `combine_epoch_test.cpp` emitted an existing GCC `-Wstringop-overflow` warning but its test passed.
- `clang-format` applied to the new test using the repository's 4-space convention; `git diff --check` passed. Changed production headers remain C++17.
- PR title check passed. PR CI status should be checked at hand-off.

## Hardware gates still open

1. Actual model with intermediate >2560: exercise real grouped GEMM or typed startup refusal, sentinel/unwritten result prevention and output verification.
2. gfx1100 and gfx1030 byte identity and decode parity independently.
3. Deterministic multi-stream oracle if strict concurrency identity is required.

No claim of hardware qualification or GPU output parity. RCS13 remains `in_progress`; Notes updated in the PR.

## Final CI and merge

- PR #1088: auto-merged with squash; merge commit a424990ac1531290322e7a130d4c7773bdd81f06.
- PR title: latest check SUCCESS (original scan failure was fixed by PR body edit).
- PR CI: SUCCESS. Radiance build and card-free component-tests PASS: moe-remote 38/38, serving 33/33, libr3 4/4. Core offline, line endings, forbidden-term scan and final ci-pass all succeeded.
- Scratch git worktree and component build/prefix removed; main tracked tree remains clean; this note is ignored under out/.
