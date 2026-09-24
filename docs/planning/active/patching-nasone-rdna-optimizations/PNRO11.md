---
id: PNRO11
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:56.925530+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Replicate lm_head for DFlash2/DSpark under tensor split

## Description

TODO, NOT-READY (target-load ordering unresolved, bias assumption wrong). Replicate lm_head for DFlash2/DSpark full-vocabulary ranking under tensor split -- CORRECTED per GPT review: (1) target-load ordering is explicitly unresolved (already flagged as NEEDS-VERIFICATION in this item's prior notes -- DFlash metadata is currently consumed after the target model already exists, too late to affect the target's own tensor-split decision); (2) the prior plan's bias assumption is WRONG: verified at b11126 src/llama-model.cpp:595-598, `pattern_output_bias` unconditionally returns `GGML_BACKEND_SPLIT_AXIS_0` -- it does NOT read or mirror `output.weight`'s own split-axis config/MIRRORED state (it only asserts output_weight exists via GGML_ASSERT); (3) the pre-load VRAM-fit check is unspecified.

## Steps

1. Add a pre-target-load draft GGUF capability/tensor-inventory probe (read the draft's dflash.*/dspark.* metadata and output.weight presence BEFORE target model construction begins) and thread an explicit replication flag through model params/model construction -- resolving the construction-ordering problem flagged in this item's own prior notes (DFlash metadata today is only available after ctx_tgt/target model already exist, per common/speculative.cpp's own constructor assert). A practical alternative if probing before load proves impractical: a CLI-level opt-in `--replicate-output-for-draft` flag set before target load.
2. In the tensor-config lambda (src/llama-model.cpp, `pattern_output_weight` branch at ~588-594), extend the MIRRORED condition to `is_dsv4 || requires_replicated_output`.
3. CORRECTED: in the `pattern_output_bias` branch (verified src/llama-model.cpp:595-598, currently unconditionally `return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);`), make it ALSO return `GGML_BACKEND_SPLIT_AXIS_MIRRORED` when `requires_replicated_output` is set (matching the weight's new mirrored state) -- otherwise preserve the existing unconditional AXIS_0 behavior. Do not assume the bias branch already reads or mirrors the weight's config; it does not.
4. Specify the exact pre-load VRAM check/error site (currently unspecified): add an explicit VRAM delta estimate at load time (full output.weight copy per rank instead of a 1/N shard) and fail the load clearly, not silently truncate, if the replicated copy does not fit the declared per-device budget.
5. Enable replication only when selector/Markov metadata proves full-output ranking and the draft lacks its own output.weight.
6. Cover shared-target lm_head and draft-owned lm_head; the latter must not force target replication.
7. Test 2+ GPU tensor split, layer-split, ordinary MTP/Eagle/no-spec controls, candidate selection vs single-GPU/reference, and VRAM fit failure.

## Detailed Solution & Technical Design

Full-vocabulary ranking cannot use a single shard without distributed top-k; this fix chooses replication. Weight and bias mirror states must stay coherent. Replication cost is an operational correctness constraint; distributed top-k is separate future work.

Reuse `GGML_BACKEND_SPLIT_AXIS_MIRRORED` (src/llama-model.cpp) rather than introducing a new `output_replicated` bool/axis kind. In the tensor-config lambda (the one containing the `is_dsv4` check at ~line 590, `pattern_output_weight` match), extend the condition so `output.weight` (and its bias via the existing `pattern_output_bias` branch which already reads output_weight's config) selects `GGML_BACKEND_SPLIT_AXIS_MIRRORED` when EITHER `is_dsv4` (existing) OR a new `requires_replicated_output` flag is true. That flag must be threaded into the tensor-config lambda's captured context (`ud->model` or a sibling struct) from a new field set during model construction/load -- set only when: this is the TARGET model of a speculative setup (not standalone), the attached draft model's metadata proves full-vocabulary ranking (mirror the `is_dflash2`/`selector_top_k` detection already used in common/speculative.cpp -- read the same `dflash.*`/`dspark.*` GGUF keys during target-model construction, or thread the already-parsed draft capability from common_speculative_impl_draft_dflash's construction into llama_model's split-tensor-config path), AND the draft does NOT own its own `output.weight` (check the draft GGUF's own tensor inventory before deciding to replicate the target's). Load-time VRAM accounting: replication means N tensor-split ranks each hold a full output.weight copy instead of a 1/N shard -- add an explicit VRAM delta estimate at load time and fail the load clearly (not silently truncate) if the replicated copy does not fit the declared per-device budget.

## Code Samples & Guidance

Target file: `src/llama-model.cpp` (b11126). Verified anchor (exact text from source): the block\n```\n        // output\n        if (std::regex_match(tensor_name, pattern_output_weight)) {\n            if (is_dsv4) {\n                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);\n            }\n            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);\n        }\n```\n(lines ~588-594) -- mode="replace", new text changes the inner condition to `if (is_dsv4 || requires_replicated_output) {`. NEEDS-VERIFICATION before authoring: (1) exact name/scope to thread `requires_replicated_output` through -- read the full lambda capture list and `ud` struct definition around line ~370-400 to find where a new bool can live; (2) where draft-capability detection happens relative to target-model construction order (today `common_speculative_impl_draft_dflash` builds AFTER `ctx_tgt`/target model per its constructor assert at speculative.cpp ~952 'DFlash requires ctx_tgt and ctx_dft to be set' -- so the target's tensor split decision may need to happen at a later re-plan point, or the flag needs to come from CLI/params set before target load rather than discovered from the draft GGUF after the fact; this ordering question is the single most important thing for the next agent to resolve before writing patch.py, and may require a CLI-level opt-in `--replicate-output-for-draft` as a practical alternative to inferring it late).\nPatch package sketch: `patches/1264_pnro11_lmhead_replication_tp/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}` (next free id -- adjust for actual availability at authoring time). patch.toml: kind="enhancement", state="untested", requires=[].

## Files

Speculative model params/load; Meta split-state; DFlash model loading/selector metadata; lm_head weight+bias placement; TP/reference tests; VRAM accounting.

src/llama-model.cpp (split-axis tensor-config lambda); common/speculative.cpp (existing DFlash/DSpark metadata detection, reference only); include/llama.h if a new param is needed; patches/1264_pnro11_lmhead_replication_tp/*; TP/reference fixtures; VRAM accounting evidence.

## Validation

Exact candidate selection under TP; full output on every rank; bias placement; no replication when draft owns output; layer/no-spec controls; VRAM delta and fit rejection.

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1264_pnro11_lmhead_replication_tp`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1264_pnro11_lmhead_replication_tp --source bigcherry-tuning`; unit-level split-config fixtures: output.weight selects MIRRORED when the new flag is set (matching existing is_dsv4 path's behavior, which is already exercised by DeepSeek-v4 models), still selects AXIS_1 (sharded) when flag is unset, draft-owned-output case never sets the flag. Hardware (Brutus, 2+ GPU tensor split, not run here): compare candidate selection under TP against single-GPU/reference for DFlash2/DSpark drafts with target lacking replication (control, expect wrong/degraded ranking or existing behavior) vs with replication (subject); ordinary MTP/Eagle/no-spec configs must be unaffected (non-selection controls); VRAM-fit-failure case must fail load with a clear explicit error, not truncate.

## Effort & Risk

Work=L (revised down from L given a matching split-axis primitive, GGML_BACKEND_SPLIT_AXIS_MIRRORED, and draft-metadata detection precedent, is_dflash2/selector_top_k, both already exist upstream -- this is wiring, not new-mechanism invention). Main open risk is construction-order (target model split-config decided before the draft model that would inform it is even loaded); resolving that ordering question is the prerequisite design step, flagged NEEDS-VERIFICATION above.

## Standards

Architecture metadata, not names; topology proof; exact selection correctness; resource accounting.

## Acceptance Criteria

Full-vocabulary ranking matches reference under TP; unnecessary replication is absent; weight+bias states are coherent; memory cost is within declared limit or load fails clearly.

## Notes

Supersedes: NRO12
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro12

2026-09-24 relevance at b11126: TODO. Grounded directly against b11126 (GGML_BACKEND_SPLIT_AXIS_MIRRORED + is_dsv4 output.weight handling in src/llama-model.cpp; is_dflash2/selector_top_k GGUF-metadata draft detection in common/speculative.cpp) -- anchors grep-verified, not guessed. GPT design request req_fd0a2a33c0804146 (batched PNRO11+PNRO12, dev-gpt-agent) was still running/had not returned a terminal response by the time this item needed to be finalized in this session; design was completed directly from source instead. If/when that GPT response lands later, a follow-up session should read it and reconcile against this design, particularly on the construction-ordering question flagged above.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: confirmed target-load ordering remains genuinely unresolved (real prerequisite work, not yet a design). Verified via direct read that src/llama-model.cpp's pattern_output_bias branch (lines 595-598) unconditionally returns GGML_BACKEND_SPLIT_AXIS_0 and does NOT read output.weight's own config/MIRRORED state as previously assumed -- corrected step 3 to explicitly extend the bias branch's own condition rather than relying on inherited state. Added the previously-unspecified pre-load VRAM check/failure requirement.

## Change Log

- 2026-09-09T10:52:56.925530+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:25.733744+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.099464+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.751906+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:19.935492+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.164554+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:51.105317+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:code_samples, section:files, section:validation
- 2026-09-24T02:35:05.511646+00:00 (updated-by): Updated: section:effort_risk, section:notes
- 2026-09-24T04:50:35.412744+00:00 (updated-by): Updated: section:description, section:steps, section:notes
