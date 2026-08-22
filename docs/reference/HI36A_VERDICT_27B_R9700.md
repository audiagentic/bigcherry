# HI36a verdict — winner generalisation, 27B R9700 evidence

Date: 2026-08-21. Evidence: `h36-campaign-27b-r9700/` (same directory).
This is the **evidence/decision** half of HI36 per the 2026-08-21 GPT
adjudication (req_09656b782a5d449a). The runtime implementation (HI36b)
starts only from this verdict, on replay v5 (HI74).

## Frozen ship gates vs measured (all from committed artifacts)

| gate | limit | measured | verdict |
| --- | --- | --- | --- |
| max cross-run regret (S5 A/B) | ≤5.0% | 1.10% | PASS |
| p95 regret (S5) | ≤1.0% | 0.00% | PASS |
| call-weighted mean regret (S5) | ≤0.5% | 0.014% | PASS |
| candidate availability at target Y | 100% | every converted holdout miss had the representative present in the target's measured set (49/49) | PASS |
| correctness/promotion at every tested Y | 100% | S6: 40 promoted through the CPU-reference gate (schema 6), 2 confirmation-rejected — the gate rejected, it did not pass | PASS |
| S7 same-workload replay coverage | 100% | 19,077/19,077 executed, 0 misses, 58 exact / 59 entries, rerun_required=0 | PASS |
| holdout misses resolvable, same-family key, call-weighted | ≥95% | GO families (mmq+mmvq): **49/49 = 100%** across both holdouts (S7 12k: 11 mmvq; S7b parallel-2: 38 mmq + 10 mmvq) | PASS |
| wrong-family representative | structurally impossible | `family` (and `src0_type`) is in every key | PASS |
| outlier migration >5% (family NO-GO trigger) | ≤5% | regret localised to the same two mmvq keys in both runs (K=5120, M=10240/12288); the historical 6.02% outlier is **reproducible and localised**, not migrating | PASS |

All gates pass. No exact-only blacklist is required (worst regret 1.10% is
inside the gate, not pathological).

## Per-family verdict

| family | verdict | evidence |
| --- | --- | --- |
| mmq | **GO** | 100% cross-run stability (worst regret 0.00%); S7b converted 38/38 batched-decode signatures that did not exist in the parallel-1 record — generalisation is what makes them covered at all |
| mmvq | **GO** (was CONDITIONAL) | the condition was width invariance: 21/21 width-perturbed misses across both holdouts resolved by same-ne0 tuned siblings (12k holdout: width-3 decode; parallel-2 holdout: batched decode); cross-run regret ≤1.10%, localised to two prefill-width keys |
| mmvf | **NO-GO** | no mmvf candidates exist for a q8_0 dense workload (0/2, 0/1) — there is nothing to generalise from; stays native |
| blas | **NO-GO** (reinforced) | 0/26 + 0/31; the 12k and parallel-2 workloads each surface *new* tiny native GEMMs (64×64, 256×256) absent from the 8k record — the target set moves with the workload; stays native, zero risk |

## Frozen representative-selection rule (frozen with this verdict)

1. **Keys** (per family; `family` and `src0_type` always included, so a
   wrong-family representative is structurally impossible):
   - mmq: `(family, src0_type, K, M)`
   - mmvq: `(family, src0_type, ne0)` — full ne0, i.e. K and M both kept
   - (mmvf/blas: no key — NO-GO)
2. **Representative** = the promoted winner of the group's
   highest-call-count member signature in the tuning run; ties broken by
   lower median. "Promoted" means it passed the CPU-reference correctness
   promotion gate — an un-promoted candidate can never be a representative.
3. **At the target Y the generalised key yields a recommendation, never an
   authorization.** Dispatch requires the standard v5 runtime guards: the
   candidate exists in Y's binary catalog and passes `can_execute` against
   the real target signature. This is the adjudicated correctness boundary.
4. **Blacklist**: empty for this verdict.

## What this does NOT establish

- Nothing about float workloads (the evidence is q8_0 dense + built-in MTP,
  widths 1–5). mmvf and float-path claims need their own evidence.
- Nothing about the 245,760-context configuration: the holdouts perturb
  batch/width/length around ctx 8192–16384. A long-context run may surface
  further shapes; the gates would be re-applied to that miss log.
- The S6 promotion ran on pre-HI67-slice-3 tooling (J: tree 5020c68); the
  correctness gate is the older gate. Re-running promotion on HI67-final
  tooling is part of HI36b, not this verdict.

## Disposition

- HI36 (this item) closes with the verdict.
- HI36b — runtime/cache implementation on v5 — is tracked separately
  (blocked by HI74; the v5 `match_kind` discriminator is the extension
  point for the generalised route).
