---
id: PRBE104
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-12T19:27:37.075621+00:00'
breadth: ''
skill: ''
created-by: agent
---

# RD08 contract-design review: is bit_identical the right correctness bar for VDR=2's reduction-order change

## Description

RD08's (patch 1204) real, confirmed, deterministic correctness-gate FAIL (config/experiment-contracts.toml's RD08-Q6K-MMVQ-VDR2 contract, required check bit_identical) has been investigated twice now: once on 2026-09-11 (GPT req_3c98f154389148bb, ruled out RD25 as cause) and again on 2026-09-13 (PRBE103, confirmed the divergence is deterministic, not GPU/process nondeterminism). Neither investigation resolves the actual open question both sessions independently landed on: VDR=2 intentionally changes lane assignment and warp-reduction structure versus VDR=1 (processing both 8-element chunks of a Q6_K dot product per call instead of one), so a different-but-still-numerically-valid floating-point reduction order is an expected, deliberate consequence of the optimization -- not necessarily evidence the implementation is wrong. Whether exact bit-for-bit output identity is the scientifically appropriate acceptance bar for this class of kernel, versus a numerical-tolerance/backend-reference-style criterion, is a deliberate contract-authority decision this project's own lifecycle rules require be made independently of any specific patch's outcome (GPT, req_d76911c6fc814809: 'Do not change the contract merely to rescue RD08... The review must justify the intended correctness semantics independently').

## Steps

1. Review the fork's own original claim and evidence for VDR=2 being bit-identical to VDR=1 (upstream commit 4591cc980) -- was their own claim measured on different hardware/compiler, and does their evidence actually support exact identity or just numerical equivalence?
2. Determine the actual magnitude/nature of the observed divergence (already measured: ~1e-7 relative in the err metric, well under the 0.0005 backend-reference tolerance used elsewhere) and whether it's consistent with expected floating-point reassociation error from a genuinely different-but-correct reduction order.
3. Decide, as a deliberate contract-authority action (not a validation-producer side effect): does RD08-Q6K-MMVQ-VDR2's bit_identical requirement stay as-is, or does it get revised to a numerical-tolerance check (e.g. backend_reference-style)?
4. If the contract is revised: rerun RD08's full qualification under the new contract before any promotion decision.
5. If bit_identical stays required: transition patch 1204 to state=rejected (the claim is now genuinely disproved, not just untested).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Both 2026-09-11 and 2026-09-13 investigations independently converged on this same open question without resolving it -- do not re-litigate whether the divergence is real/deterministic (already confirmed twice); focus directly on the contract-design decision.

**RESOLVED (2026-09-13).** GPT made the actual contract-design decision (req_8163325eb9c545ea): revised RD08-Q6K-MMVQ-VDR2's correctness requirement from bit_identical to backend_reference (NMSE within test-backend-ops' existing 0.0005 threshold) -- floating-point addition is non-associative and VDR=2 deliberately changes accumulation grouping, so exact bit-identity was never the scientifically appropriate bar. Contract hash change voided the legacy point-estimate waiver (VA24); migrated to ci95_threshold_bound_v1 with min_paired_rounds=10, removed the stale waiver entry.

Re-evaluated using already-gathered PRBE103 evidence: correctness gate now PASSES (all 15 rows ~19-21x below threshold). Extended performance to the required 10 paired rounds (4 additional real rounds): performance gate FAILS honestly -- target_kernel_gain_pct point estimate +0.261%, but 95% CI lower bound -0.062% (crosses zero), below the required 0.3% threshold. RD08's real gain is too small to statistically distinguish from zero at n=10.

Final disposition: correctness passes, performance does not -- RD08 is not promotable to validated, but for a real, honest, now-fully-resolved reason rather than an open contract-design ambiguity. state stays untested. See patches/1204_rd08_q6k_mmvq_vdr2/README.md's 'PRBE104 resolved' section for full detail.

## Change Log

- 2026-09-12T19:27:37.075621+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260912_192807_confirmed-patch-1204-rd08s_9661
- 2026-09-12T19:28:07.083361+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T22:47:17.610915+00:00 (updated-by): Updated: section:notes
- 2026-09-12T22:47:22.977085+00:00 (state-transition): State: pending → completed
- chg_20260912_224753_fully-resolved-patch-1204-rd0_6022
- 2026-09-12T22:47:53.075691+00:00 (updated-by): Updated: section:ledger-events
