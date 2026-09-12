---
id: PRBE109
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-12T23:01:11.676984+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Wire RD58's real contract_promotions into validation_campaign.py's run() -- exact gap identified

## Description

RD58 (patch 1234) now has a real, complete, passing evaluate_promotion_gate() verdict computed manually this session (2026-09-13, docs in patches/1234_rd58_pin_state_buffer_multigpu_restore/README.md's 'Real complete contract promotion evaluation' section): correctness PASS, trigger PASS, performance PASS (real dual-GPU tensor-split decode A/B, delta -0.016%, well under the 5% budget). But this verdict can never be reached via the actual CLI path (--run-rd58-state-restore in tools/bigcherry/patch/validation_campaign.py's run() function) because contract_promotions (the dict feeding compute_persisted_validation_eligible()) is only ever populated for rd08_qualification and rd73_qualification (lines 4808 and 5230) -- RD58's real correctness gate IS computed (contract_correctness_gate, line 5379) but its promotion verdict is never computed or added to contract_promotions. This means eligible_for_validated_state structurally can never be True for RD58 through this CLI path, regardless of real evidence -- a real, precisely located code gap, not a hardware or evidence gap.

## Steps

1. In validation_campaign.py's run() function, after rd58_result is computed (~line 5084) and rd58_contract_correctness_named_results is built (~line 5111), add a real aggregated_effects computation for RD58's decode control lane -- this needs a real LaneEffect built from paired decode benchmark rounds (control vs subject, real llama-bench runs against the contract's bound tierA-qwen4b-q6k model, -sm tensor), analogous to how RD08's lanes are built via run_rd08_validation_lanes(), NOT a scalar percentage.
2. Call experiment_contract.evaluate_promotion_gate() with the real correctness_gate, aggregated_effects, and trigger_proof (trigger_proof can reuse the existing trace_evidence marker-hit data, converted to TriggerEvidence).
3. Populate contract_promotions[rd58_contract.id] = <that promotion verdict>, mirroring line 4808's RD08 pattern exactly.
4. Verify the full offline test suite still passes, and run --run-rd58-state-restore for real on Brutus to confirm eligible_for_validated_state now reports True given the same real evidence already gathered manually this session.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

This is precise, scoped, real engineering work -- not a hardware/evidence gap. The real evidence needed to prove RD58 passes already exists (this session's manual computation); this item is purely about making the standard CLI/evidence-persistence pipeline able to reach the same real conclusion automatically, so state="validated" can be set with full tooling confidence per this project's lifecycle doctrine.

## Change Log

- 2026-09-12T23:01:11.676984+00:00 (created-by): Created by agent
