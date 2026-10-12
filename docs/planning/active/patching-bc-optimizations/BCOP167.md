---
id: BCOP167
order: 167
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T17:04:14+00:00'
created-by: agent
priority: P1
work: S
---

# QFP23: make 0910 runtime profile parity and auto failure deterministic

**Discovery:** Validated 0910 already implements shipped INI profiles and early initialization, superseding QFP23's original build-out TODO. C runtime accepts repeated same-value flags/includes and malformed/repeated arch declarations rejected by Python lint, does not enforce `ENV_DOCS` membership for custom overrides, selects the first architecture match in directory order, and discards the late auto-apply error at model creation. Source-proven conditional integrity gaps; no observed GPU failure or new performance gain.

**Authoritative owner/consolidation:** QFP23 owns the 0910 parser, Python validator, model-load hook and release profile identity; QFP18 owns historical stack promotion. Reuse existing `bc_env_docs`, `runtime_profiles.py`, `test_0910_feature_sets.py` and `test_runtime_profiles.py`; no second parser, selector, allocator or scheduler. Upstream llama.cpp router `--models-preset` provides per-child model-specific options rather than mutable process-global environment.

**Already acting:** 0910 patch marked validated; profile folder and C/Python tests already shipped. Last material independent QFP23 work 2026-10-05; no independent work/PR/queued experiment in the prior 12 hours. Protected recent Radiance prefill, GDN, MoE, AllReduce, n-gram, QFP15 and release work is untouched.

**Bounded action:** Table-driven C/Python grammar parity; reject duplicate architecture ownership; reject undocumented override flags; propagate auto-application error through existing model-load failure without exit/abort; retain first-model-only and explicit env precedence; run Linux/MSVC and release-bundle tests where available. No GPU benchmark until correctness is settled.

**Terminal disposition:** ACCEPT with zero parser acceptance mismatches, deterministic unique arch, no healthy serving after failed requested auto, and unchanged named-profile/release identity. Otherwise restrict to named single-process profiles and do not claim `auto` qualification. Host-model 8/8 and source-static 12/12 only; no repository pytest, C/HIP build or GPU execution. Full technical design belongs to QFP23.

**References:** QFP23, QFP18, 0910_feature_sets, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-models.cpp .
