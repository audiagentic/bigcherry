---
id: QFP23
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T22:01:19.661087+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# QFP23: deterministic runtime auto-profile admission

## b11552 audit (2026-10-11)

**Disposition:** `0910_feature_sets` is already validated. Do not create another loader, parser, placement solver, scheduler or telemetry system. QFP23 owns only deterministic `auto` admission, flag timing and profile identity. No new performance measurement.

**Implementation:** `engines/llamacpp/patches/0910_feature_sets/patch.py` implements `bc_load_profiles`, `bc_parse_profile_file`, `bc_flatten`, `bc_apply_request`, `ggml_bigcherry_features_init` and `ggml_bigcherry_features_apply_arch`. Explicit overrides, include validation, rollback and startup hooks exist. Source profiles are `engines/llamacpp/overlay/profile/{base,flashnext}.ini`; `tools/bigcherry/build/builds.py::resolve_runtime_artifacts` hashes shipped `profile/*.ini`. Historical `config/runtime-profiles/` and `overlay/src/profile/` proposals are obsolete.

### Source-proven gaps

1. C `bc_load_profiles` uses unsorted `readdir`; `ggml_bigcherry_features_apply_arch` chooses the first matching `arch`. Python `runtime_profiles.py::check` does not reject two profiles claiming the same arch. A reversed-order host fixture selects different profiles; this is not a reproduced GPU failure.
2. `flashnext.ini` binds `arch=qwen4exp` to `BIGCHERRY_ATTN_TS=1,1,0` and MTP/router settings tuned for 2×gfx1100 + gfx1201 with gfx1030 drafter. Architecture does not prove model hash, quant, GPU count/order, split, VRAM or P2P. Do not auto-admit topology-bound profiles without positive attestation.
3. C parsing lacks Python's ENV_DOCS whitelist, repeated-key rejection and strict arch grammar for external `BIGCHERRY_PROFILES`. External profile bytes are not in the shipped bundle hash. Record external SHA-256 and effective flags.
4. `auto` runs in `llama_model_create` after backend init. `EnvDoc` has no read-phase field; early getters may already run. Server handles init error statuses; other tools' init hooks discard them.

### Bounded implementation

**A:** Extend `tools/bigcherry/patch/runtime_profiles.py::check` with global `arch -> profile` uniqueness; add two-file/reversed-order, unique-arch and bad-key fixtures in `test_runtime_profiles.py`.

**B:** Extend existing 0910 C post-parse admission, before env mutation, to reject cross-profile arch collision, invalid arch, unknown ENV_DOCS flags, duplicate key and unmatched explicit `auto`. Add C compile/run cases to `test_0910_feature_sets.py`: two file orders, bad external override, explicit-wins, help/error and rollback. Preserve current include semantics.

**C:** Keep `BIGCHERRY_FEATURES=flashnext` explicit and model/device-attested. Remove/reject `auto` binding for topology-specific profiles until exact GGUF/quant and physical device split are verified. Census actual getter sites for shipped flags as pre-model/model-load/graph-build/lazy; only late-read flags qualify for auto. No second configuration system.

**D:** Compare explicit and admitted auto in separate processes with identical pinned binary/model. Record profile digest, effective env, physical GPUs, no-P2P, graph allocation and transfers. Negative controls: single R9700, dual XTX, three-card topology, different qwen4exp quant and bad profile. Require logits/KLD, greedy/MTP acceptance, multi-ubatch, long-context and repeated-request parity. If no safe auto use case remains, retain explicit profiles and close; no loader speedup is claimed.

### Consolidation and sources

QFP23/0910 owns profile admission; QFP18 owns flag docs; QFP/MET/MMQ/MTP owners retain tuning. Upstream llama.cpp [#17859](https://github.com/ggml-org/llama.cpp/pull/17859) and [#26118](https://github.com/ggml-org/llama.cpp/pull/26118) already implement model/system CLI presets; reuse for ordinary llama arguments, not patch-private env. [vLLM](https://github.com/vllm-project/vllm/blob/main/docs/configuration/serve_args.md) and [SGLang](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/server_arguments.md) are config-precedence references, not AMD gains.

Selected `build/pin-b11552` by independent `9fa502648e4f00ea334d4f783750417edd8f275d` (Oct 11 16:57 UTC). QFP23 last substantive update Oct 5, profile last Oct 10 11:49 UTC; 12h exclusion passed. Protected PA49, MET07, QFP41, QFP48, QFP49, PGC17 untouched. Eight static checks and two-order host fixture passed; no pytest, C/HIP build or GPU run.
