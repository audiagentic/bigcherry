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

## Evidence

Runtime artifacts (build logs, raw test-save-load-state output) land
under
`artifacts/patch-validation/1234_rd58_pin_state_buffer_multigpu_restore/<campaign-identity>/`,
outside this tracked patch directory. The compact, tracked record is
`evidence/validation.json`.
