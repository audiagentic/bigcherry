# Standardized multi-arch/multi-model patch validation criteria

GPT-designed (`dev-gpt-agent`, `req_5b07ec3d063a4231`, 2026-09-13), adopted
per explicit user instruction to define and apply a standardized test
criteria across the patch registry. This is a practical, executable-today
methodology built on this project's EXISTING tooling (`patch-lint`,
`patch-rebase-check`, `patch-gates`, per-patch contract producers) -- it
does not invent a new benchmark framework.

**gfx1151/RDNA3.5 and Vulkan are explicitly parked** (user instruction,
2026-09-13) -- never inferred as covered from other architectures, never
substituted with heterogeneous hardware. Record as `deferred-hardware`
when a contract requires them and stop there.

## Universal minimum per patch

Every patch, regardless of its own bespoke evidence, requires:

- G0 exact composition, G1 docs, G2 current-pin apply/rebase (all via
  existing `patch-lint`/`patch-rebase-check`/`patch-gates`).
- G3 complete validation definition (`validation.toml` or the documented
  reason one doesn't exist yet).
- A successful, identity-bound build (`capture_completed_build_evidence()`)
  on every applicable real architecture/topology (see below).
- Process execution attestation (this project's
  `bigcherry.experiment.attestation` machinery).
- Every correctness check the bound Experiment Contract/`validation.toml`
  declares -- the contract remains authoritative for `bit_identical`,
  `backend_reference`, `ppl_equality`, thresholds, etc.
- Positive activation/trigger evidence whenever the patch changes a
  conditional runtime path.
- A performance A/B ONLY when the patch makes a performance claim -- a
  diagnostic/framework patch with no runtime/performance claim does not
  need invented benchmarking.

**A missing required check is BLOCKED/incomplete, never silently treated
as PASS.**

## What "multi-arch/topology" means (applicability coverage, not blind coverage)

Real hardware available: 2x gfx1100 (Brutus, can run `-sm tensor`
together), 1x gfx1201, 1x gfx1030.

- Generic HIP/runtime patch, not architecture-restricted: single-GPU on
  **gfx1030 + gfx1100 + gfx1201** (all three).
- Architecture-scoped patch: every available IN-scope architecture, plus
  at least one available OUT-of-scope architecture proving real
  fallback/non-selection (not just "we didn't test it there").
- Multi-GPU/tensor/collective patches: additionally **2x gfx1100
  `-sm tensor`**.
- Heterogeneous pairs (gfx1100+gfx1201, gfx1100+gfx1030): ONLY when the
  patch itself claims heterogeneous operation, device selection/admission,
  or a fail-closed heterogeneity guard. Do not force unrelated patches
  through mixed-GPU runs.
- If an acceptance criterion genuinely requires N>=3 same-architecture
  GPUs: record `deferred-hardware`/unavailable. Never substitute Brutus's
  heterogeneous 3/4-GPU set to manufacture "N>=3 coverage."

## Baseline comparison: BOTH comparisons, different purposes

Three arms:

- **A** = stock pinned llama.cpp (no BigCherry patches at all).
- **B** = BigCherry baseline composition with the focal patch absent.
- **C** = B + the focal patch.

`B<->C` is the causal patch qualification and is what feeds the focal
Experiment Contract -- this is NOT replaced by stock-vs-patched, which
cannot attribute an effect to the focal patch alone. `A<->B` measures
BigCherry's accumulated baseline cost/benefit versus upstream. `A<->C`
shows the final user-visible state versus upstream. For
performance/optimization patches, retain all three arms -- this matches
BigCherry's existing validation policy, not a new requirement.

## Standardized model matrix (by applicability, not blanket "run everything")

- `tierA-qwen4b-q6k` (Qwen3.5-4B): general coverage, AND the GDN/hybrid
  fixture (confirmed 2026-09-13 via PRBE102 -- 24 of 32 layers are Gated
  DeltaNet recurrent layers, not purely dense as its own notes previously
  said).
- `tierM-gptoss20b-q6k`: MoE coverage.
- `tierM-ministral14b-q4km`: an independent dense-transformer family
  (distinct from the Qwen family).
- `tierM-qwen35b-a3b-moe-mtp`: MoE/MTP-specific paths only.
- `tierL-qwen27b-q8`: the dual-gfx1100 large-model topology only.

## Execution order for the next patch

1. `PYTHONPATH=tools python -m bigcherry patch-lint --json`
2. `patch-explain <patch-id>` / read the bound Experiment Contract and
   `validation.toml` -- these define the required checks and
   architectures, not invented fresh per patch.
3. Fresh `patch-rebase-check --source <canonical-source> --json <report>`.
4. `patch-gates <patch-id> --intent validate --rebase-report <report>`.
5. Materialize clean B/C compositions at the current pin (plus stock A
   where a runtime/performance comparison applies).
6. Build each required architecture/topology via `build_tree()`, capture
   `capture_completed_build_evidence()` immediately.
7. Run the EXISTING contract-specific producer where one exists
   (`--run-rd08-contract`, RD13's PPL producer, RD26/RD43/RD58/RD73's
   equivalents, etc.) -- never substitute manual judgment for its gates.
8. Require activation/trigger evidence before accepting correctness or
   performance results from a conditional optimization.
9. Run contract correctness FIRST; only spend benchmark time after
   correctness + activation pass.
10. Persist identity-bound evidence; run
    `patch-gates <id> --intent promote` -- promote only if G0-G5 and
    every applicable contract obligation pass. Production/build admission
    then adds G6/G7.

**Do not build a new universal benchmark framework.** Standardize the
matrix and decision order above; reuse each patch's existing
contract/producer; add only the missing thin producer for a
contract-required check when a patch currently lacks one.
