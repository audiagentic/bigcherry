# RAD20 experiment: rad20-blocking-lanes

Archived from branch `feat/rad20-blocking-lanes` (tip `e0c1d6110e7f37f9180a18c35557c5f6dbc4d346`, base `31c282c62745e0d402dc2f1e7d0d8feaf3442105`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD20 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad20-blocking-lanes <current main> && git apply --3way experiments/radiance/RAD20-rad20-blocking-lanes/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-10-27 e0c1d6110 lab(radiance): barrier-value-bench pending changes
- 2026-10-26 b5969d25e revert: drop the RAD_HIPCC stopgap and the ad-hoc ROCm comparison script; ROCm selection becomes a build-system stack
- 2026-10-26 bd4cb2850 fix(build): radiance builds use the selected ROCm's hipcc and headers, not /opt/rocm
- 2026-10-26 178dc0335 lab(radiance): ROCm 10.1 vs 7.2.4 comparison; stage-A barrier/idle-power harness; gfx11 MES kernel backport
- 2026-10-25 409f4673a fix(radiance): RAD20 lanes wait on the request word with their own one-wave kernel, not hipStreamWaitValue32
- 2026-10-25 96cc5972e lab(radiance): RAD20 discriminator - R3_MOE_WORKER_IDLE_WAIT arms one idle wait of a chosen kind on an extra stream; 
- 2026-10-25 5806d1bb2 lab(radiance): RAD20 experiment B harness (IPC signal vs engine relay); profile arm values may hold commas as +
- 2026-10-25 cbb7a8f08 fix(radiance): RAD20 large lane sleeps on an engine host hint, no GPU wait armed through the decode; arming locked ag
- 2026-10-25 3f4fcd176 lab(radiance): RAD20 diag nolargewait (large lane polls as main, no armed GPU wait)
- 2026-10-25 a4e6ed5da lab(radiance): rad20-profile profiler options as the fifth argument
- 2026-10-25 9f40ca9e9 lab(radiance): RAD20 diag noevent (poll without the per-request done event)
- 2026-10-25 c8bca5639 lab(radiance): rad20-profile arm specs as the fourth argument
- 2026-10-25 749f57590 lab(radiance): rad20-profile takes arm specs with per-arm environment and can build HEAD; idle-gap micro-benchmark
- 2026-10-25 0f26f2a0c lab(radiance): rad20-profile samples XTX clock levels every 10 ms; run label
- 2026-10-25 61359e405 fix(lab): rad20-profile keeps its arguments from the env script
- 2026-10-25 1d47fd5b4 fix(lab): rad20-profile takes build ids and sources the project environment
- 2026-10-25 3b37e78db lab(radiance): RAD20 targeted profile (rocprofv3 on helper 0, clocks) and event/host-read micro-benchmarks
- 2026-10-25 fabec1e35 fix(lab): RAD20 diag string literal
- 2026-10-25 9adf67456 lab(radiance): RAD20 decode diagnosis switch and 3-arm A/Bs (lab only, removed before merge)
- 2026-10-25 700771e56 fix(lab): rad20-rad18-e2e uses the build directory, not runtime-prefix
- 2026-10-25 633c9fd64 feat(bench): ab-benchmark cells record MTP draft totals; RAD20/RAD18 lab A/B
- 2026-10-25 4372bc647 feat(radiance): helper lane threads sleep in GPU completion events, no polling (RAD20)
