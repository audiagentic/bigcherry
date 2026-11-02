---
id: RHP15
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-02T19:52:38.746266+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Admission covers media encoder passes (encoder-only passes need no remote MoE ticket)

## Description

Found by GPT review of RHP11 slice C round 6 (2026-11-02). Radiance Engine::step_loop runs run_encodes() before step admission when encoded media is pending (core/engine.cpp:578-582); run_encodes releases encoder batches straight to the rank threads (core/engine_mm.cpp:147-163). rad0002 adds bc_moe_remote_before_pass() unconditionally before run_pass (patch.py ~795-817), and the plugin's before_pass requires a live, bound admission ticket (moe_remote_hook.cpp before_pass). So an encoder-only pass without a scheduler-parent ticket is rejected. That breaks multimodal serving with remote MoE; text-only serving is unaffected. Predates round 6.

## Steps

1. Distinguish encoder-only passes from target-model MoE passes at the admission bridge (the pass knows whether it runs the remote MoE model).
2. Skip before_pass for encoder-only passes; a decoder pass without admission still fails closed.
3. Never reuse a stale ticket.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free bridge test: an encoder pass with no ticket proceeds without touching the mailbox; a remote-MoE pass with no ticket is refused. Hardware: one multimodal request through the production launcher with helpers up; identical text vs main.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Do not allow arbitrary zero-ticket passes through a remote-capable model.

## Change Log

- 2026-11-02T19:52:38.746266+00:00 (created-by): Created by agent
