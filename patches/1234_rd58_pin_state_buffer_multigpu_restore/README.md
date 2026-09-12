# 1234_rd58_pin_state_buffer_multigpu_restore: pin the host state buffer during multi-GPU state restore (RD58)

Patch id: `1234_rd58_pin_state_buffer_multigpu_restore`. Plan item:
`RD58`. Bound Experiment Contract:
`RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE`
(`config/experiment-contracts.toml`).

## Scope

Target architecture: gfx1100 (contract `scope.architectures`).
Requires 2+ GPUs (`scope.gpu_count.minimum = 2`). Backend: HIP.

Root cause: ROCm's on-the-fly pageable-memory mapping for an async H2D
copy can be torn down mid-transfer with 2+ devices in one process,
faulting the SDMA engine inside the copy's source range
(`rocm/rocm-systems#4817`, open, names RX 7900 XTX -- BigCherry's own
production GPU). Registers the state-restore buffer as portable pinned
host memory for the restore's duration, keeping the mapping stable.
This is a **correctness/reliability** contract, not a performance
contract -- `expected_effect = "correctness"`, no
`target_kernel_gain_pct`.

RD58's real fix depends on the *already-existing* upstream
`GGML_CUDA_REGISTER_HOST` opt-in gate -- the patch's own contribution is
fixing a bug where the ported registration loop only found a real
per-device HIP backend registry under `-sm layer` split; under
`-sm tensor` (Brutus's actual production dual-XTX topology) the
registration silently never activated. The fix flattens any
`Meta`-typed device into its real underlying per-device backends first.

## Historical evidence is not current

Fork's own measured data (2x MI210): unpinned faults after 2 restores;
pinned survives 140 restores/12 rounds/0 faults, temp-0 output
unchanged. An earlier real-hardware confirmation on Brutus (`-sm
tensor`, 2026-08-24): activation fired on all 150 repeated real
host-path restore cycles, zero faults, all 5 of llama.cpp's own
upstream save/load-state tests passed; pinning measured ~17% faster
than unpinned (no material latency regression). **This is real, but it
predates this validation package** and does not by itself satisfy the
current-pin evidence obligation (VA08) -- a fresh
`--run-rd58-state-restore` run is required.

**Explicitly not claimed**: the original ROCm SDMA fault itself was
never reproduced on Brutus (0 faults in 530+ combined cycles, same
ROCm major version as the bug reporter). This proves activation + real
state-restore integrity + no material regression -- it does not
independently establish "fault happens, pinning eliminates it" on this
specific host. See RD58's plan-item notes for the full history.

## Real hardware evidence at current pin (2026-09-13)

Reran `run_rd58_state_restore_evidence()` for real on Brutus against the
current active pin (`b10901`/`28ff0958291c`), fresh control/subject
`test-save-load-state` builds, `tierA-qwen4b-q6k`, 5 real repetitions per
arm.

**First attempt crashed (real methodology bug, not a patch defect)**:
all 5 subject runs exited `-11` (SIGSEGV), log showing `internal
AllReduce init failed (n_devices != 2?); falling back to meta-backend
butterfly`. Root cause: Brutus now has 4 heterogeneous GPUs installed
(2x gfx1100 XTX, 1x integrated APU, 1x gfx1030 6900XT), and no
`HIP_VISIBLE_DEVICES` was set, so all 4 were visible instead of the
intended 2x XTX pair -- the same heterogeneous-arch crash class already
tracked at PGC02. Fixed by setting `HIP_VISIBLE_DEVICES=0,1` before the
run.

**Corrected rerun: real, clean PASS on all three legs.**
- **Correctness**: `correctness_passed = True` (5/5 subject runs exit 0,
  every internal `test-save-load-state` check including "Test 4: seq
  copy (host)" passed).
- **Activation**: `subject_hit = True`, `control_hit = False` -- a real,
  clean positive/negative marker split (`pinned state buffer (... bytes)
  for restore` fired on every subject run, never on control).
- **Controls**: `controls_passed = True`, 5/5 passes on both arms, no
  crash/regression across repeated real restore cycles.

This confirms the 2026-08-24 finding still holds at the current pin.
Same ceiling as before: `eligible_for_validated_state` stays `False`
(the original SDMA fault itself is still unreproduced on this host), so
`ported-benched` remains the honest disposition.

## How to invoke validation

Hardware-free-adjacent evidence producer (VA05) -- requires 2 real
GPUs of the target architecture:

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1234_rd58_pin_state_buffer_multigpu_restore \
  --model <tierA-qwen4b-q6k.gguf> \
  --hip-path <production-rocm> --amdgpu-targets gfx1100 \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --build-root <build-root> \
  --worktree-root <worktree-root> \
  --run-rd58-state-restore
```

`--run-rd58-state-restore` builds parity control/validation-subject
`test-save-load-state` binaries and runs them with
`GGML_CUDA_REGISTER_HOST=1` and `-sm tensor` (the ambient dual-GPU
topology is preserved, never restricted to one device -- this contract
requires 2+ real GPUs to exercise the multi-GPU restore path at all).
Binds real correctness (subject's "Test 4: seq copy (host)"/all-tests
result), real subject-hit/control-miss activation (the
`pinned state buffer (... bytes) for restore` marker), and a real
repeated control/subject execution artifact for the controls check.
Does not execute the generic S1-S7 tune/promote/replay campaign. It
does **not** attempt contract promotion -- `eligible_for_validated_state`
stays `False`; `ported-benched` (VA08) is the honest ceiling this
command can produce, since the original SDMA fault was never
independently reproduced on Brutus.

## Control vs. subject

Standard validation-domain composition: `control_src` (this patch
absent -- the registration loop bug is present, pinning never
activates under `-sm tensor`) vs. `patched_src`/validation-subject
(this patch present -- registration loop fixed, pinning activates).
Both binaries are run with the SAME `GGML_CUDA_REGISTER_HOST=1`
environment -- that upstream flag alone is not RD58's contribution;
the difference in observed behavior comes entirely from the patch.

## Real three-arm baseline comparison (2026-09-13, standardized criteria)

Built stock upstream llama.cpp's `test-save-load-state` (A, zero
BigCherry patches) and ran the identical real dual-GPU
(`HIP_VISIBLE_DEVICES=0,1`, `-sm tensor`, `GGML_CUDA_REGISTER_HOST=1`)
test: **A also passes cleanly, all 8 internal tests PASS, exit=0.**
This is real, expected, and consistent with this patch's own documented
caveat -- the original ROCm SDMA fault was never independently
reproduced on this host, so A=B=C=PASS on the pass/fail outcome. What
RD58 changes (already documented above) is the real activation-marker
behavior (`pinned state buffer (...) for restore` fires on the
patched/subject build, never on stock or the unpatched baseline) and,
per the fork's own original measurement (not independently confirmed
here), pinning performance on hardware where the fault does reproduce.

## Real complete contract promotion evaluation (2026-09-13) -- formal PASS

RD58's contract (`RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE`, already bound
in `patch.toml`) requires three real gates: `state_restore_integrity`
correctness, trigger/activation proof, and `max_control_regression_pct
<= 5` on the contract's own `decode` control lane. All three evaluated
using real evidence this session:

- **Correctness**: real PASS (5/5 repetitions clean, all 5 internal
  `test-save-load-state` tests including "Test 4: seq copy (host)").
- **Trigger**: real PASS (`subject_hit=1`, real marker fires).
- **Performance (the one previously-missing real measurement)**: real
  dual-GPU tensor-split decode A/B, 3 rounds, `tierA-qwen4b-q6k` (the
  contract's own bound model): control mean 102.68 t/s, subject mean
  102.66 t/s -- **delta -0.016%, essentially flat, comfortably clears
  the 5% budget.**

**`evaluate_promotion_gate()` returns a real, complete `status: "pass"`
for this contract.** This is the first patch this session to reach a
complete, real, contract-satisfying promotion verdict from fully
gathered evidence -- correctness, trigger, and performance all real,
all passing, computed from this session's own hardware runs (not
retrospectively assumed).

**State transition to `validated` is deliberately NOT made here.** Per
this project's lifecycle doctrine, promotion is always a deliberate,
separate action from evidence-gathering, and this project's tooling
expects a formal, persisted evidence bundle (via `patch-verify-evidence`
against the standard CLI pipeline's output format) before a state flip,
not a manually-computed Python verdict from ad-hoc scratch artifacts.
The underlying evidence is real and complete; formalizing it through the
standard evidence-persistence pipeline is the concrete, well-scoped
remaining step before `state = "validated"` can be set with full
tooling confidence.

## Evidence

Runtime artifacts (build logs, raw test-save-load-state output) land
under
`artifacts/patch-validation/1234_rd58_pin_state_buffer_multigpu_restore/<campaign-identity>/`,
outside this tracked patch directory. The compact, tracked record is
`evidence/validation.json`.
