# VTP43 design review 1 (GPT via Codex, read-only, plan and source excerpts supplied in the prompt)

Reviewed plan revision 1 (bf16-declared group). Findings below; the plan was revised to declare the group as E4M3 (option b) in response (see VTP43 revision 2).

--------
workdir: /work/bigcherry
model: gpt-6-luna
provider: openai
approval: never
sandbox: read-only
reasoning effort: medium
reasoning summaries: none
session id: 01a1e127-db3d-7e41-9ef0-7fb2a530e102
--------
user
You are reviewing an implementation PLAN (not code) for a new KV cache layout in the radiance inference engine, before any implementation starts. Read-only: do not edit any file.

Plan: docs/planning/active/run-dense-27b-virtual-tp/VTP43.md

The plan and the relevant source excerpts (with line numbers) are inlined below. Your shell tools may not work in this environment (sandbox); review from the inlined material. Where you need something not shown, say what to check.

Target: Qwen3.8-27B (qwen35_w8a8 build of qwen35_fp8.cpp) at TP3 on gfx1100 (RX 7900 XTX, no native fp8). Today it runs attn_{prefill,decode}_h256_gqa6_bf16kv, attn_decode_h128_gqa4_bf16kv (drafter?), qk_norm_rope_gate_kv_store_bf16.

Please review in depth and report:
1. Correctness of the design: layout, declaring the group as RAD_BF16 with head_dim 3HD/4, drafter group behaviour, bringup refusal, streaming refusal, E4M3->bf16 widening, writer conversion. Anything in the source that assumes a BF16 group has 2*HD bytes per token/head with K then V halves (prefix cache, host tier, checkpoints, KV copy/defrag, rollback, any kernel or core code reading a bf16 cache directly, e.g. the QSA indexer, kv_store_bf16 rows that would wrongly match, ref_attn_paged host fallback, the libref 'ref_kv_store' resolution) -- i.e. what would silently break or resolve to a wrong row.
2. Kernel feasibility: where in the bf16 (KVP=1) bodies V is staged; whether O_VF8 fits (VGPR pressure at the 256 bound, LDS layout, O_V8-like code staging), and which libr3 native rewrites must change.
3. Missing steps, files, tests or risks; anything in the plan that is wrong against the source.
4. Whether alternative (b) (declare as F8 with head_dim 3HD/2 like fp8_v4) would actually be simpler/safer.
Give findings as: blocking / should-fix / nit, each with file:line evidence. Be concrete and brief.


# PLAN
---
id: VTP43
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-15T21:38:40.989488+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# bf16_fp8 KV cache: bf16 keys, E4M3 values (more -long capacity than bf16, closer to bf16 than fp8)

## Second pass of the same review (additional findings)
## Effort & Risk

L. Risks: the h256 bf16 kernels sit near the 256-VGPR bound (O_VF8 conversion may spill: measure per instantiation, fall back to staging V as codes in LDS like O_V8); libr3 native rewrites must keep matching the patched templates; V outliers above 448 saturate (the fp8 cache has the same limit; KLD will show it).

## Standards



## Acceptance Criteria

- [ ] GPT design review done and its findings resolved before implementation.
- [ ] `--kv-cache-dtype bf16_fp8` serves qwen27b at -long with about 3/4 of bf16's bytes per token (capacity >= 1.3x bf16), refused clearly elsewhere.
- [ ] Selftest rows for the new attention and store kernels pass on all cards; no spill in the new instantiations.
- [ ] KLD against bf16 well below fp8's on the long corpus (target <= half of 0.0054 mean, no p99.9 outliers like fp8's), or the result is recorded and the owner decides.
- [ ] Decode and prefill speed within fp8's and bf16's range (no worse than fp8).
- [ ] GPT code review done and findings resolved; PR merged with evidence.

## Notes



## Change Log

- 2026-11-15T21:38:40.989488+00:00 (created-by): Created by agent
- 2026-11-15T21:39:40.921837+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
# SOURCE EXCERPTS (paths relative to engine source root unless libr3)

===== abi/rad_builder.h lines 55-82
