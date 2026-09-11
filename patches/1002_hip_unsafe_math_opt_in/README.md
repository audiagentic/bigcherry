# 1002: make `-funsafe-math-optimizations` opt-in for HIP builds

## Scope

Adds `GGML_HIP_UNSAFE_MATH` (default OFF) gating `-funsafe-math-optimizations`
in `ggml/src/ggml-hip/CMakeLists.txt`, which this project's tree previously
enabled unconditionally for every HIP build.

## Why

Unsafe math reassociates floating-point reductions, which can flip a greedy
argmax and break byte-identical output for MTP speculative decoding at
temperature 0 (upstream PR #26696's CI test names this bug ID AIESW-40114,
confirmed there on gfx1151/RDNA3.5 only).

## Upstream / provenance

Cherry-picked from unmerged upstream PR
https://github.com/ggml-org/llama.cpp/pull/26696. Upstream later merged its
own version (`e79e4bf6`) by deleting the flag outright rather than gating
it; this backport's anchor was re-adapted to the surviving
`GGML_HIP_EXPORT_METRICS` block so its own opt-in escape hatch is preserved
(effective default build behavior unchanged by the pin bump -- see
`patch.py`'s docstring for the full anchor-migration rationale).

## Real hardware evidence

Two independent real-hardware verifications, both already recorded in the
release ledger (2026-08-21):

1. **Mechanism confirmation (byte-identical divergence test)**, this
   project's own hardware, before backporting: on gfx1201 (RDNA4), a temp-0
   MTP speculative completion was byte-identical to the non-speculative
   baseline. On gfx1100 (RDNA3, this project's local hardware), the same
   test diverged -- both completions agreed word-for-word through most of
   the response, then produced a genuinely different closing sentence,
   while two independent non-speculative baseline runs on the same hardware
   were byte-identical to each other. This rules out general run-to-run
   noise (the divergence is specifically speculative-vs-non-speculative)
   and reproduces the PR's own described mechanism on a third architecture
   upstream never tested.
2. **Build-configuration verification (deterministic, not statistical)**,
   ledger event `chg_20260821_155302`: confirmed on Linux/Brutus via
   `ninja -t commands` that `-funsafe-math-optimizations` is genuinely
   absent from the real per-file HIP compile command when
   `GGML_HIP_UNSAFE_MATH` is OFF (default) and genuinely present when
   explicitly set ON -- an exact match to the patch's documented intent.
   This investigation also found (filed and fixed separately as HI81, not a
   defect in this patch) that the identical `CMAKE_HIP_FLAGS` mechanism
   silently failed to propagate to compile commands on Windows/Ninja --
   fixed by patch 1232 (`hi81_windows_cxx_hipcc_flags_reach_compile`),
   re-verified on gfx1100/ROCm 7.1/Ninja: the flag correctly appears in the
   real Windows HIP compile line when ON and is correctly absent when OFF.

Both checks are stronger signal than a statistical benchmark comparison --
the compile-command check is an exact deterministic match, and the
divergence test directly reproduces the documented failure mechanism on
real hardware.

## Lifecycle note

`patch.toml`'s `state` field was `"untested"`, which no longer reflects the
real, already-recorded evidence above -- corrected to `"validated"` in this
change as a metadata-sync fix, not a new promotion decision (the promotion
was already made and evidenced across the 2026-08-21/23 sessions; this only
brings the package's own state field in line with history it had not yet
recorded).

## Known limitations

- No `validation.toml` adapter exists. This patch is a CMake build-flag
  gate with no runtime code path of its own to trace-mark; its evidence is
  the compile-command check and the mechanism-reproduction test above, both
  produced by manual real-hardware investigation rather than the generic
  campaign harness. `origin = "upstream-pr"` means it is not eligible for
  the `is_framework_configuration_patch` local-framework adapter path, and
  no Experiment Contract is bound (none has been authored for this patch).
- `retirement` in `patch.toml` notes this patch should be removed once PR
  #26696 lands in the pinned base in its upstream form -- it has not yet
  (upstream instead merged a different, flag-deleting version, `e79e4bf6`,
  which this backport was explicitly adapted to coexist with).
