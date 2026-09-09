# RHA10 admission-gate checkpoint

This checkpoint records a fresh exact-source replay activation run while
keeping the production admission gate fail-closed. The diagnostics binary was
built from the retained RHA04 source slice and used the same model, dual-XTX
topology, signature inventory and replay cache as the production replay binary.

The diagnostic server log reports 54 replay winners, 32 exact cache matches,
25 misses, and 57 replay-hit records including 12 non-native candidates. Its
coverage artifact records 21,566/21,566 measured dispatches and, critically,
13,342 final tuned launches versus 8,224 native launches after revalidation.
The paired maintained
`run_bench.py` server-bench capture used the diagnostics-off production binary
and returned successfully with pp512 923.85, pp2048 1283.26, tg128 34.06 and
tg512 34.12 tokens/s. Diagnostic activation and production timing are kept
separate.

The final tuned-launch gate is satisfied, but correctness is not: a three-prompt
deterministic probe matched stock/native on all prompts while replay differed on
one prompt. This is retained as a blocking correctness investigation, not
silently treated as harmless nondeterminism. `final_tuned_launches` is sourced
from the compile-time diagnostic witness; production timing remains from the
diagnostics-off binary, which is the documented zero-overhead design.

RHA11 supersedes the blocking capture: both implicated winners were seeded to
`mmvq:native:v1` through the supported replay-cache exporter. The corrected
activation/timing record is
`replay-diagnostic-activation-rha11-corrected.json`; its raw logs and cache are
retained under `docs/evidence/2026-09-10-rha11-correctness/`. The corrected
three-prompt corpus passes; RHA10 remains fail-closed only for the explicit
24 replay-miss policy decision.

The final complete-cache record supersedes that intermediate state:
`replay-diagnostic-activation-rha11-corrected.json` now reports 79 entries,
57 exact matches, zero misses, and `production_admitted=true`.
