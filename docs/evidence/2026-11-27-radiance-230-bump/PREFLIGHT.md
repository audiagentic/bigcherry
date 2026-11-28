# 2.3.0 executable prework — ABI and first dependency

2026-11-27; BigCherry 2b7e5e6d5, upstream old b6b8751cccb0e834a13ae9e5afca824ee9a487aa, target 3b13fd32663ef52252a7f0be255523480aa22ffe. Work performed in isolated temporary executable directories, reading exact upstream Git objects; no edits to vendor, no GPU or engine build. Reproducer [abi-preflight.py](abi-preflight.py) uses existing patch objects and patcher, not a new production harness. Run from this evidence revision with `python3 docs/evidence/2026-11-27-radiance-230-bump/abi-preflight.py --mirror <upstream-git> --scratch <existing-executable-scratch>`. Initial /tmp execution refused (no-exec); rerun in configured executable scratch succeeded. The short shell limit interrupted one intermediate run; completed job 2 exit 0, elapsed 23.72 s. Raw stdout retained in [abi-preflight.txt](abi-preflight.txt).

## ABI result

| Headers | Version | Fingerprint | C/C++ rows |
|---|---:|---|---:|
| old pristine | 43 | 1260f5fc8445d3a0 | 951 |
| old + local | 44 | 5cc71b9c5bb7ab40 | 959 |
| target pristine | 44 | 9e15786273f47304 | 960 |
| target + proposed local extensions | 45 | 6c431f97e8b670d4 | 968 |

The old local fingerprint reproduces the historical rad0048 receipt. Four header patch subsets apply and reapply idempotently in both compositions. Candidate local field order: append after native dspark_walk in RadBuildCtx and vl_mask in RadBatch, not before them; align fingerprint row order identically. Native target offsets all unchanged, on-disk records unchanged relative to target. Both KV byte-contract fixtures pass under NDEBUG. Target pristine C/C++ suite 30 checks / 8 cases passes; scratch combined suite 30 checks / 8 cases passes with table regenerated from measured rows and revised version/size expectations. The regenerated table is not independent numerical or loader evidence; the independent target-offset comparison and suite's row coverage check are meaningful. No actual runtime ABI transition or patch modification yet.

| Field | Proposed offset/size (bytes) |
|---|---|
| RadBuildCtx (whole) | 256 |
| draft_head_off / expert_split / dspark_walk | 204/4, 208/8, 216/4 (native) |
| tp_ownership / pipelined_prefill / pipelined_prefill_min_rows | 224/8, 232/4, 240/8 |
| kv_layout / attn_prefill_fp8 | 248/4, 252/4 |
| RadBatch (whole) | 512 |
| vl_mask / cu_seqlens_h | 496/8, 504/8 |
| RadKVGroupDecl / kv_layout | 64 whole, 56/4 |
| RadArchProbe / kv_layout_ok | 136 whole, 128/4 |

Implementation: rad0018 reanchor n2-ctx-field and n2-layout-tp-ownership at dspark_walk, preserving the native fields; rad0020 trailing local fields then apply as written; rad0025 msp-abi append after vl_mask; rad0048 abi44-layout-batch append fingerprint after vl_mask; one version transition 44 -> 45 with guard matching 45 (not 44); remaining ABI header edits apply unchanged. Adapt rad0025 msp-key-size from native 504 -> 512, preserve native vl_mask not-recordable guard and host cu_seqlens key. Update rad0048 tests/abi_layout_test.cpp assertions for native 224 -> 256, 504 -> 512, 128 -> 136; regenerate table from compiled layout or derive exact affected table records, retain full on-disk checks; update C fixture version and loader migration tests (old local44 and native44 both reject; no-layout45 rejects); existing Python test must not assume the pristine source is ABI43. Note native table permits appended fields and is not stale merely because its RadBuildCtx baseline size is 208. Audit positional aggregate initialization, runtime pass-key semantics and all seven plugin coupling manifests separately. Do not accept the header proof as integrated gate.

## First dependency: rad0016 -> rad0017

A second actual isolated patcher test staged only exact target files touched by rad0016 and rad0017, including created source files via original patch objects. rad0016 had two failed anchors because upstream added fetches_rows. For edit n1-rowmask replace anchor/replacement prefix `    bool                 chain_fenced = false;` with `    bool                 fetches_rows = false;` (same trailing struct close). For n1-row-mask replace `        row.chain_fenced = chains == 2;` with `        row.fetches_rows = fetches && fetches(sr.second) != 0;` (same next push_back); leave native chain and fetch semantics intact. Preserve original guards. Both full packages then apply and reapply:

```
rad0016 apply edits 43 bad []
rad0016 reapply edits 43 bad []
rad0017 apply edits 12 bad []
rad0017 reapply edits 12 bad []
```

This resolves the observed two anchor failures and establishes that the formerly dependency-blocked rad0017 can compose on top in isolation. It does not prove full production recipe, actual rank eligibility or shared rendezvous behavior. Port these two edits first, run existing test_vtp14_n1_rank_code_selection.py, test_vtp14_n3_shared_rendezvous.py and the actual bc_rank_devices_test.cpp when the relevant target is built; no whole-engine run for the anchor repair. Preserve fetches_rows. Upstream repeats row.chain assignment; avoid gratuitous upstream cleanup.

## Mechanical rename checklist

Verified Git rename detection on the exact target (not just name guesses): arch/common/rad_block_{attn_gated,attn,gdn,mlp,moe,mtp}_fp8.h -> same prefix _q.h; rad_fp8.h -> rad_linear.h. libr4d/r4d_fused_quant_fp8.hip -> r4d_fused_quant.hip; r4d_gemm_fp8a8.hip -> r4d_gemm_i8a8.hip; r4d_gram_accum_fp8.hip -> r4d_gram_accum.hip; r4d_fp8_lm.h -> r4d_lm_row.h; r4d_gemm_fp8a16_nt_m1.hip -> r4d_logits_gemm_i8.hip; r4d_fp8_args.h -> r4d_quant_args.h. Thirteen verified renames; similarity ranges 56–95%, so do not global-replace identifiers or assume unchanged semantics. Reconcile TP/pipeline patches on generic headers first, then MoE/KV and ownership; update actual build/test inventory for renamed units. Re-run composition after dependency repair. All active features and recipes remain protected under parity guide; no retirement claimed.

No full source build, GPU, engine startup or actual production patch changes in this prework. Exact reproducible header test and first dependency anchor proof remove guesswork; integrated candidate remains blocked until implementation and qualification.
