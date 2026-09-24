# b10901 -> b11126 throughput regression

Plan item: bump b11126 follow-up (post-bump smoke showed pp512 -18%, tg128 -14% on Brutus gfx1100)
Status: complete (no regression; see docs/evidence/2026-09-24-bump-b11126-throughput/)
Owner: pin-bump process
Question state: one-shot investigation

## Question

Is the lower throughput of the b11126 release build real, and if so is it
caused by the upstream llama.cpp change or by BigCherry's patch composition?

## Inputs

Four llama-bench binaries on the same GPU, model and flags:
`U10901`/`U11126` (llama-native stock, upstream only, identical CMake
configuration) and `B10901`/`B11126` (bigcherry:native release builds).
Model: tierA Qwen3.5-4B Q6_K from `$BC_MODEL_ROOT`.

## Outputs

`results.jsonl` (one row per arm/round/test with position and build commit)
under the run's `work/runs/` directory; `run_arms.py analyse` prints per-arm
means, per-position drift, deltas, complete separation and an exact
Mann-Whitney p.

## Runtime

GPU required: yes (one device, exclusive). Real compilation: no (binaries are
built beforehand). Mutates canonical BigCherry state: no.

## Safety

llama-bench exits normally per run (no server, no kill). Do not run builds or
other GPU jobs on the host while it runs. Arm order rotates every round;
rounds must be a multiple of the arm count.

## Disposition

One-shot. Keep until the regression is attributed and recorded; then ARCHIVE.
