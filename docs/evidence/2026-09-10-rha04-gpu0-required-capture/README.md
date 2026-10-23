# RHA04 GPU0 required-attestation capture

Date: 2026-09-10  
Role: Brutus environment, resolved through environment settings  
Status: **invalid before measurement**

## Purpose

This was the first fresh RHA04 production-role cell after the retained HI168
bundle. It used the current pushed `planning-refactor` checkout and the
maintained `bigcherry ab-benchmark` server-capture path, with
`execution_evidence=required`. It was a bounded GPU0 9B native-arm probe, not
a parity result.

## Configuration

- Model: `$BC_MODEL_ROOT/qwen3.5-9B/gguf/mtp/Qwen3.5-9B-Q6_K.gguf`
- Arm: BC native binary from the existing HI171 build cache
- Devices: `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`
- Expected backend/identity: ROCm, `gfx1100`, `0000:03:00.0`
- Server mode: `-sm none`, `--fit off`
- Runner: maintained `bench/run_bench.py --bench-type server-bench`
- Capture mode: production role, required execution evidence

The run was started from a temporary remote configuration derived from the
existing HI171 GPU0 configuration; no host paths or credentials were added to
the repository.

## Result

The server loaded the model, became healthy, and shut down cleanly. The cell
was rejected before the server-bench request because the server log contained
no parseable physical-device attestation. The recorded error was:

```text
execution attestation failed [ATTESTATION_MISSING]
```

No throughput metrics were produced and `performance_admitted` remained
`false`. This is an attestation-path failure, not evidence of native
performance or a GPU failure.

The captured startup log reported the default verbosity (`3`) and model
materialization, but no `using device ... (PCI-BDF)` line. This confirms the
current gap: `run_server_arm_capture()` uses `ServerRunner` directly and does
not force the higher verbosity used by `AttestedServerSession`; with
`--fit off`, the normal production path does not provide the parser's physical
device locator evidence.

## Raw artifact hashes

Artifacts remain on the Brutus temporary run directory:

| Artifact | SHA-256 |
|---|---|
| Temporary config | `de2e90f5f74b1224bb92ac82369869211c3ab4b4848816af3a3ed6a4c70c6f40` |
| `run.json` | `bb8e20633739114ae4bf207da354497da3af17327414d0636b95b5bc7f5b31d3` |
| Native `server.log` | `7c5e55d25e1841146e0ec4e8971b3f58fba749b6a913e62df443a36b28390e25` |
| Native `cell.json` | `72deaf9b8e76fb1928df5587f0d11b21ef86cf6a0a0340630cdb1a614823bc81` |

## Disposition

Retain this failed cell. Do not rerun the complete matrix until the capture
path has a documented solution that preserves diagnostics-off production timing
as a separate arm from diagnostic attestation. The next implementation slice
is an execution-attestation design decision (for example, a process-bound
physical-device probe), not a relaxation from `required` to `observe`.
