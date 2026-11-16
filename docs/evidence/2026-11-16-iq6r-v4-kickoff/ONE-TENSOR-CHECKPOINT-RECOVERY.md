# IQ6R-V4 one-tensor checkpoint recovery

17 November 2026. Experimental conversion only; no model qualification or production promotion.

## Failure mechanism

The old 20-tensor Stage 2 logged five newly completed GPTQ weights before its 10,800-second timeout (exit 124). At the final check, the process consumed roughly two CPU cores but there was no additional completed GPTQ log output. Two large weights can be worked concurrently; the last completed log line does not identify the operation that subsequently stalled.

The converter seals the RAD2 container only after every weight finishes. `rad-info` reports invalid magic `00000000` for `staged-conversion/stage-02.incomplete.rad`. It cannot be safely used with `--reuse`. Keep that failed output and its logs for investigation. The old SHA-verified 26 GB `stage-01.rad`, all four vocabulary-head GPTQ weights, recipes, calibration and Hessians survive.

## Replacement schedule (isolated, no source mutation)

The separate `micro-conversion/` path starts with the existing, SHA-verified Stage 1 and adds **one** new trunk GPTQ tensor per independent sealed RAD2 checkpoint. Micro-stage 2 has 5/292 GPTQ rules, stage 3 has 6/292, and stage 289 is the exact pinned 292/292 final recipe. This is a checkpoint *between tensors*, not inside the tensor.

`iq6r_v4_micro_convert.py` uses the pinned managed converter with `--reuse` of the previous sealed checkpoint, then checks RAD2 structure, trellis counts, output size, recipe hashes and SHA-256 before sealing the new checkpoint. It refuses changed prior artifact hashes and all preexisting outputs. Stage 1 and the original failed Stage 2 remain unmodified. Only the two most recent micro RAD files remain once the next is sealed; prior micro receipts remain.

`iq6r_v4_micro_controller.py` acquires a **separate exclusive GPU lease** for each micro-stage and allows other jobs between leases. It fails closed without automatically retrying or promoting anything. GPU work pins the R9700 through `HIP_VISIBLE_DEVICES=2`.

The extra safety costs disk traffic: each checkpoint writes/copies a roughly 26 GB model container. A single tensor can still exceed the three-hour timeout; if that occurs, the prior sealed checkpoint remains safe but the tensor needs separate profiling.

## Commands and evidence

- `tools/lab/radiance/iq6r_v4_micro_convert.py --stage 2 --preflight`: no GPU, validates original Stage 1
- `tools/lab/radiance/iq6r_v4_micro_controller.py --max-stages 1`: run one checkpoint under an exclusive lease
- `tools/lab/radiance/iq6r_v4_micro_controller.py`: sequential, separately leased checkpoints through Stage 289
- Same environment as original controller: `BC_WORK_ROOT`, `BC_MODEL_ROOT`, `BC_RADIANCE_SOURCE`, `BC_RELEASE_MANIFEST`

Tests: all 57 IQ6R offline tests pass; lint passes. Real Stage 1 micro preflight passed and identified original SHA-256 `77e403172bfe625ea62ed635081be1f612b20df5324a20e89a2b51142f25eda4`, next recipe SHA-256 `76df15b441d2b1bfae6a13f459f7db73f5978ac214bff122f7c50129e267fa11`. GPU execution and end-to-end inference quality remain unverified.
