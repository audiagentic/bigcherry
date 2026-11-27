# IQ6R-V4 adaptive checkpoint handoff

17 November 2026. Keep the original sealed Stage 1, micro-checkpoint receipts, and the failed original 20-tensor Stage 2 unchanged. Production W8A8 is not involved.

## Measured starting point

The first one-tensor checkpoint, micro-stage 2, sealed cumulative GPTQ 5/292 after **464.44 seconds** and wrote an approximately 26 GB RAD2. Its predecessor is the original Stage 1 (4/292). Micro-stage 3 (6/292) started separately under an exclusive R9700 lease.

The 20-tensor Stage 2 timed out at 10,800 seconds and left an invalid RAD2 header; keeping batches at 20 risks losing hours of work. Conversely, rewriting a full 26 GB container after every one of 288 additional tensors would write roughly 7.5 TB. Batching 2–4 tensors can roughly halve or quarter the repeated container writes, with correspondingly longer recovery intervals.

## Conservative adaptive policy

- Newly handed-off adaptive job starts with **two GPTQ tensors** if the latest sealed one-tensor checkpoint took at most 900 seconds. Otherwise it starts with one.
- Subsequent jobs use one if the immediately preceding job took more than 1,800 seconds; two if the last job completed in 900 seconds; four only after **two consecutive adaptive jobs** each complete within 600 seconds.
- Never exceed four new tensors or the remaining 292/292 count. No changes to IQ6R parameters or the final pinned recipe.
- Every job obtains a separate exclusive GPU lease, validates its predecessor SHA-256 and recipe, runs plan-only, writes a complete RAD2, verifies all encoding counts and hashes, seals it, then records elapsed runtime.
- Explicit minimum free-space guard of 85 GiB before another output; retain only the two newest sealed adaptive RAD files. The immutable original Stage 1, failed Stage 2 diagnostic, and inherited micro checkpoint remain intact. The exact disk write volume depends on actual batch sizes.
- A single pathological GPTQ tensor may still exceed the timeout; only a verified prior sealed checkpoint is reused, never an incomplete one. Any failure blocks the controller without retry or production promotion.

## Handoff safety

Do not replace a recipe or batch size inside the **already running** single-tensor controller. `iq6r_v4_adaptive_handoff.py` checks that the micro controller is actively waiting for micro-stage 3 and the stage-3 worker exists. It stops **only the controller parent process** via a PID-specific SIGSTOP, allowing the already-launched stage-3 GPU worker to finish. After stage 3 exits with a hash-verified receipt, it retires the paused controller and starts the new adaptive controller. On failure or timeout it resumes the old controller; it never kills or interrupts an active converter. The handoff itself consumes no GPU.

`iq6r_v4_adaptive_controller.py` holds the old micro exclusion lock for its full life, and its child converter verifies an active controller token. The controller is mutually exclusive with the old controller and holds an independent controller lock; all conversion work remains under per-job GPU leases.

If an operator must recover a paused old controller from a lost handoff supervisor, verify the old PID and current GPU worker first, then resume it with SIGCONT. Do not launch a new controller while any old worker or exclusive lease is active. Status is in `conversion-adaptive-handoff.json`, `conversion-controller-adaptive.json`, and the source micro and adaptive receipts under `${BC_WORK_ROOT}/iq6r-v4-qualification/`.

## Qualification

This is a throughput/durability policy only, not a completed full model or verified RDNA3 inference. Numerical quality, full-model hashing, RDNA3 kernels and TP3 remain separate gates.
