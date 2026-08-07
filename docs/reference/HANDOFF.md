# Handoff — state of play and what to do next

Read this first when picking work up. For build commands, see [BUILD_AND_TEST.md](BUILD_AND_TEST.md). For why things are shaped the way they are, see [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

---

## Where things stand

Phases 0–3 are complete and verified on hardware. The dispatch layer runs end to
end: signature construction, blake2b hashing, dispatch-key derivation, process
cache, replay lookup, native fallback, miss recording, coverage reporting.

| Item | State |
| --- | --- |
| HI01 audit | done — 32/32, on pristine and patched trees |
| HI01 baseline | done — gfx1100, 1545/1545 `test-backend-ops` MUL_MAT |
| HI02 CMake/ABI | done — all 5 rejection rules verified |
| HI03 catalog | done — 26 AMD architectures, deterministic manifest hash |
| HI04 dispatch hook | done, exercised |
| HI05 signature/blake2b | done — 7/7 vectors match Python `hashlib` |
| HI06 MMQ forced-J | done, verified |
| HI07 MMVF forced block size | done, verified |
| HI08 MMF forced nwarps | done, verified |
| HI09 MMVQ geometry | done, verified — routing built and tuned; forced geometries beat native |
| HI11 replay cache | done — loader and writer; full chain verified |
| HI13 collection points | done — coverage complete on all tested configurations |
| HI10 record mode | done — JSONL, inventory, SQLite |
| HI12 tuning engine | full sweep completes end to end |
| HI09b resource blacklist | not started |
| HI14 multi-GPU | partly done — tensor split + RCCL verified |
| HI15 production hardening | mostly done — cache export and `replay-slim` work; slim runtime needs testing |
| HI16 test suite | partial — patcher tests, no C++ suite yet |

Full patch set: 16 file patches, applies cleanly from a pristine checkout, verified against multiple upstream revisions with no adjustment.

## Key findings so far

- **Tuning produces real gains on prefill.** See tuning database for numbers.
- **The real workload differs sharply from synthetic sweeps.** The target workload prefers a tuned candidate roughly nine times as often as `test-backend-ops` suggests. Do not judge candidate value from synthetic sweeps.
- **MMVQ `small_k` geometries are the standout dimension** on two independent workloads — and was unreachable until `_variant_initialiser` was fixed to carry it through the variant params struct.
- **Five defects found only by hardware runs**, each hidden behind the one before it (RV01–RV05). The structural fix: derive eligibility from the instantiation set the build already knows, not hand-maintained predicates. Pair with an HI16 test that launches every registered candidate against a signature it claims to serve.
- **End-to-end throughput is within noise of one standard deviation** — matmul saving is real but small relative to decode wall time and benchmark spread. Measure matmul time directly, not tokens/sec.

## Next actions, in order

### 1. HI09b — resource blacklist

Build `full-max` with `GGML_HIP_EXPORT_METRICS=ON` (already an upstream option;
sets `-Rpass-analysis=kernel-resource-usage`), parse the remarks, map mangled
kernel symbols back to stable names, emit a blacklist the catalog consumes.
Must precede tuning — a spilling geometry that reaches the tuner costs a full
measurement cycle to learn what the compiler already said.

### 2. HI19 / HI17 / HI18 — the taxonomy and the remaining opaque paths

See [FAMILY_MODEL.md](FAMILY_MODEL.md) for source-level verification; do not re-derive it.

**HI19 first** — the four-record separation (signature / context / candidate / observation).
Establishing it before HI17 and HI18 add fields is cheaper than retrofitting it onto two new families. It also adds `dispatch_status`, which matters for any enumerated-but-unreachable candidates.

**HI17** — BLAS decomposition: conversion routes, `compute_type`, output conversion, `api_strategy`, provider policies. Start with `compute_type`: the template switch already exists, so it is close to free.

**HI18** — three allreduce implementations, one chosen at backend construction, never revisited. Implement the no-fallback candidates and the actual-path telemetry first — a fallback-enabled candidate that silently falls through makes the winner label false.

Rejected: WMMA as a family. `rocwmma` is absent from the ggml sources entirely.

### 3. HI14 — multi-GPU validation

Validate on the remaining architectures (gfx1030 RDNA2 untouched). Use
`HIP_VISIBLE_DEVICES` to isolate; the multi-GPU build (`~/bc-build-multi`) is
already configured for all three targets.

## Architectural context — what drives the remaining design

Four things about the target configuration drive the remaining work:

**Tensor split across identical GPUs.** Standards 5.2 requires signatures to be built
from the *device-local* slice, after splitting. With `tensor_split 1,1` each GPU sees
half of each tensor, so a signature built from the global shape would be wrong. Standards
10.2 then says the two devices — same hardware key, same local shape — should share one
winner used twice, not store two identical winners separately.

**MTP speculative decoding** produces small, irregular batch widths — draft widths of 1..5
alongside the verify pass. That lands squarely in MMVQ/MMVF territory at exactly the
widths the explicit geometry matrix covers (`MMVQ_WIDTHS = 1..8`). It is also the workload
most likely to expose signatures the dense selector never sees, which is what HI13's
collection points were built for.

**Long context (200k+)** means attention shapes dominate late in a sequence and the
hot-signature ranking will look very different at 200k than at 1k. Record over a
representative long run, not a short one.

**Tight VRAM.** Q8_0 at 27B across 2×24GB is tight. Workspace filtering
(`max_workspace_bytes`) is not academic — a candidate that wins on time but needs more
scratch may not be usable at all.

## Architectural context — what drives the remaining design

Four things about the target configuration drive the remaining work:

**Tensor split across identical GPUs.** Standards 5.2 requires signatures to be built
from the *device-local* slice, after splitting. With `tensor_split 1,1` each GPU sees
half of each tensor, so a signature built from the global shape would be wrong. Standards
10.2 then says the two devices — same hardware key, same local shape — should share one
winner used twice, not store two identical winners separately.

**MTP speculative decoding** produces small, irregular batch widths — draft widths of 1..5
alongside the verify pass. That lands squarely in MMVQ/MMVF territory at exactly the
widths the explicit geometry matrix covers (`MMVQ_WIDTHS = 1..8`). It is also the workload
most likely to expose signatures the dense selector never sees, which is what HI13's
collection points were built for.

**Long context (200k+)** means attention shapes dominate late in a sequence and the
hot-signature ranking will look very different at 200k than at 1k. Record over a
representative long run, not a short one.

**Tight VRAM.** Q8_0 at 27B across 2×24GB is tight. Workspace filtering
(`max_workspace_bytes`) is not academic — a candidate that wins on time but needs more
scratch may not be usable at all.

## Working on brutus — quick reference

- **SMB mapping:** `J:\development\llmhosts\bigcherry == /mnt/vault/development/llmhosts/bigcherry`
- **Device indices:** 0,1 = gfx1100 XTX, 2 = gfx1201, 3 = gfx1030
- **No stale copies** — `~/bigcherry` is old; the live tree is under `/mnt/vault`
- **Server-side files can be invisible from Windows** — produce repo files from the Windows side, or copy them back with `scp`

For full build commands and test procedures, see [BUILD_AND_TEST.md](BUILD_AND_TEST.md).
