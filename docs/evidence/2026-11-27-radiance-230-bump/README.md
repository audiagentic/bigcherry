# Radiance 2.3.0: migration and upstream adoption review

## Scope and observed gates

Baseline b6b8751cccb0e834a13ae9e5afca824ee9a487aa (2.2.0); candidate 3b13fd32663ef52252a7f0be255523480aa22ffe (v2.3.0); BigCherry base 5d1e64fcf, transition 570b5b0a9, pulled record 6a3fe02d6, draft PR #1713. 123 upstream commits, 863 files, +136070/-13203. Baseline self-triage clean; footprint 40 composed patches, 649 edits, 82 files. Target triage 66 exposed patch targets, 11 rewrite anchors, 18 reported drift failures, four changed owned origins. Orchestrator stopped at drift, COMPONENT_COUPLING_DRIFT: all seven components expect pristine ABI43 and see ABI44. No gates skipped.

Executed observational patch-rebase-check for each source and --all against exact candidate.

| Source | Clean | Failed | Blocked | Total |
|---|---:|---:|---:|---:|
| radiance-native | 0 | 0 | 0 | 0 |
| radiance | 8 | 22 | 4 | 34 |
| radiance-diag | 8 | 23 | 4 | 35 |
| radiance-routing-evidence | 9 | 22 | 4 | 35 |
| radiance-vtp44-evidence | 10 | 24 | 4 | 38 |
| all | 12 | 27 | 4 | 43 |

Zero upstream-absorbed verdicts. Lexical failures are not evidence of supersession; clean patches are not runtime-qualified. Re-run after fixing dependencies to reveal masked failures. This is source-level assessment, not compile/GPU/model qualification.

Executable prework: [ABI and dependency preflight](PREFLIGHT.md) establishes measured combined ABI45 headers and the rad0016/17 anchor repair in isolated scratch. Production implementation remains pending.

## 1. ABI — first critical gate

Upstream abi/rad_abi.h changes 43 to 44. abi/rad_abi_layout_rows.h adds RadBatch.vl_mask, RadBuildCtx.draft_head_off/expert_split/dspark_walk, RadChatFormat.object_args_key/message_open, RadEncoderDecl.span_rows, RadGenBatch.ref_log_bias/mode_sigma. Optional RadKernelFetchesRowsFn also added. BigCherry's old patched ABI44 is a different layout (historical fingerprint 5cc71b9c5bb7ab40, rad0048 README). rad0018/20 introduce TP ownership/pipeline fields, rad0025 adds runtime fields, rad0048 owns version transition and local KV/attention fields. The old version guard can match upstream's number despite divergent layouts. Actual failures: abi44-build-context, abi44-test-context-size.

Propose pristine ABI44 and composed ABI45 if retained extensions alter layout. One central explicit version transition only; append/fingerprint combined fields in actual struct order, regenerate canonical layout table, C/C++ size/offset fixtures, reject both previous local44 and pristine44 plugins in composed45. Distinguish pristine coupling from composed ABI; review bc-gfx12, bc-iqr, libr2, libr3, moe-remote, ngram-helper and serving individually. Do not just change manifest numbers, guess sizes or advance origin manifests. Inspect positional aggregate initializers and optional exports. Rebuild every plugin; compare exported ABI and layout fingerprint with matched engine. Serving contract edition 3 remains separate.

## 2. Upstream adoption decisions

| Upstream | Local disposition | Remaining work |
|---|---|---|
| 5e0cd1e2 per-card headroom (vram_budget_headroom_mib, rank_room, vram_budget_rank_spare_mib) | Adopt upstream budget foundation; rad0029 partially overlaps | Native keeps common KV split, extra named-reserve room goes to expert_tiered. Local dense rad0029 reclaims uneven static/free space into per-rank KV and computes common minimum tokens. Preserve that delta using post-reserve rank room, not raw free-min. Respect broker, explicit memory, unified and atomic publish. RSC16-18 use native rank facts, no sessions*max-context assumption. |
| 818e9133 scheduler admission errors and idle local exhaustion | Adopt native scheduler behavior; rad0036 still needed for prefill governor | Reattach policy without undoing error propagation and idle exhaustion handling; test mixed steps, retries, cancellation, starvation. |
| 4865bd6b queue reserve, 4fbce0c9 close parked queues, 0158f244 failed handoffs, 896ab80f quiescent serialization | Adopt native lifecycle; review rad0070 managed entry and rad0011 | No duplicate broker or queue reserve; feed actual queue memory into planner; test startup, park/unload, failed handoff. |
| 7244cef0 copy completion before staging recycle; 1c3584a8 reconstruct parked CPU weights; 8a1e2ac4 immutable base LoRA | Adopt native loader fixes; RUM14 re-inspect production materialization seam | Not evidence of prebuilt Program test loader. Preserve refill, staging lifetime, cancel, no stale publication. LTX02 owns model issue. |
| Native --expert-split, RadBuildCtx.expert_split, rad_expert_split.h, 8b1c108a GLM IQR split | Reuse native expert split where supported | Not general asymmetric dense head/FFN geometry, remote MoE handoff or collective. Preserve rad0018/19 and rad0002 deltas. |
| aac92d7a dense IQR, 88b5a04a builder proof, 3ade1e25 expert gate/up, 8c5b601a gfx1100 guard, be9bcf82 reversible selftests | RCS15/16/23/25 native compute candidates | Not a global switch; see section 4. |
| 475e2fb3, ae134f8d, 5da0007e native int8, 6bb1aed7 had quant | Compare native on supported group128 forms | Not IQ8R group256; retain RUM15. |
| 5d4431b8 h128/gqa3 attention; e35fdf75/b42b7129 native split-KV | Adopt supported native shapes | Does not cover local h256/gqa6, V4/K6V4/B16F8/B16I8, prefill and page constraints. |
| abi/rad_kv_fp8.h block-scaled E4M3 entries | Use upstream sparse latent/index formats | RAD_KVF_E4M3_E4M3_16 / E4M3_UE8M0_32 are not asymmetric K/V cache. |
| f1113ee9 per-M head row logging, e22a4668 scalar head, fb39c877 opt-in F32 WMMA M17-64, fa760413 BF16 columns | Adopt observability; evaluate head variants | rad0063/64 weighted vocabulary and packed IQR head not replaced; compare exact F32 logits and rank fold. |
| 63ffd212 BF16 Qwen DFlash2; 05e8af9e/e7388f9b lossless BF16x12 | Optional format/model experiment | Not replacement for IQ6R/IQ8R; measure memory/quality/speed before changing configs. |
| 7e85e03e GLM chained decode, 7b4fbf56 fused quant, 65ced7c9 i8 latent cache | Adopt native GLM paths | Not VTP multisequence pipeline or mixed GPU transport. |
| aa84ec2a ngram pread fallback and optional fetches_rows | Adopt native row-prefetch contract; review ngram-helper | Declare fetches_rows only when helper really reads file; avoid double fetch, verify mmap/pool ownership. |
| 088be423 converter workers/reuse-only, 8c5ca11f/db594d25 validation | Prefer native converter facilities if compatible | Preserve IQ6R-V4 source/weight identity and conversion evidence; compare outputs and worker isolation. |

No whole package proven superseded. Archive original packages unchanged only after equivalence/coverage proof; reason, successor and evidence in archive ledger. Keep old installed 2.2 runtime for rollback.

## 3. Retired model encodings — release blocker

**Follow-up:** actual metadata inspection and owner-approved retirement policy are in [MODEL-COMPATIBILITY.md](MODEL-COMPATIBILITY.md). All four configured files exist; no retired weight plane set was observed, and the existing Flash-Next codebook has a plausible native int8-activation migration. The initial filename-based risk below is superseded by these observations; actual candidate graph and runtime proof remain required.

Upstream README lines 87-96 explicitly refuses block FP8, MXFP4, M2XFP4, ParoQuant weights and E4M3 expert activations. Affected drafter-only models may serve with num-speculative-tokens: 0, not production-equivalent speculation. Eight Flash-Next fp8-iq4r-moe named YAMLs are at obvious risk; actual model header inspection is still pending (filenames not proof). Inspect every shipped container plane, exact startup unsupported-weight inventory, rank-local K/N group divisibility, drafter and actual row selection. For qwen27b IQ8R/IQ6R and local HCBF16 Flash-Next inspect every block including drafter. Qualify approved replacement model via KL/top-k, memory, TTFT/decode, speculation and context. Do not restore removed kernels merely to make an old model load; retain 2.2 until replacement proven.

Native core/config.cpp still rejects fp8_v4/k6_v4 and accepts bf16/i8/fp8 only. qwen27b-tp3-long-bf16i8 and flashnext-4gpu-bf16i8 require local 16/8 or approved migration, not silent symmetric i8. Port rad0048-54 preserving paired cache descriptor, scale offsets, page16, h256, QK/PV capability, default BF16/E4M3/I8 controls; unknown format must refuse, never fall through. Native block-scaled sparse cache is a different contract.

## 4. Native RDNA3 and IQR overlay

libr3d now a useful candidate, not drop-in. Its README explicitly says no collective rows and no chained forms; dot2/wave-matrix arms round differently. CMake enables dense/expert IQR only when R3D_GPU_TARGETS exactly gfx1100.

* Dense gemm_nt_q_iqr: r3d_i8_gemm_rows.h/r3d_iqr_layout.cpp/r3d_gemm_iqr.hip, raw packed r3d.iqr.mad.raw.v1, MAD U3/U4/U5 L16 fwht128 group128, N/K multiples 128 (prior exact source review). No evidence of U6/U7, L15, bias/chained/ragged forms; scalar correctness path not measured WMMA throughput. Retain optimized local IQ6R.
* Routed IQR gate/up: r3dh_moe_iqr_layout.cpp config and r3d_rows.cpp IQR_GATE_CONSTRAINTS restrict token i8a8h, K2560, 512 experts, top_k10, group/a_group128, parts2, N/N_odd1280/1280 or 512/768 or 768/512, M1..2048. Five-plane U4 codes, 65536-entry I8 table, two U32 MAD coefficients, F32 row and column scales. Narrow model geometry, not generic Flash-Next experts/down projection or remote handoff. Reversible tests present, not run.
* Native int8 group128 does not cover IQ8R group256 two-level U8×BF16. Target libr4d/r4d_fp8_layout.cpp i8_enc_layout still uses RAD_BF16 r4d.iq8r.s.t16 (line 212), relayout stores product BF16 (320-328). RUM15 precision issue remains. Port diagnostic rad0061 to int8-only r4d_gemm_i8a8.hip and shared prefill header, with layout/relayout/unrelayout; keep diagnostic until proven.
* bc-iqr/bc_iqr_plugin.cpp resolves libr4d.so/libr3.so r4d_* layout/shape/launch/fusion and split-counter/narrow geometry hooks. libr3d uses r3d_* and distinct raw packed layout. A YAML name change or removing libr3 breaks fallback and risks layout mismatch. Resolve exact device/provider/layout per row; never reinterpret raw packed as fragment order. Retain current base/transport until a proven native alternative exists; local collective/chained overlay remains necessary.
* Owned origins: r4d_iqr_embed.hip upstream comment-only; likely semantic no-op, prove byte/row smoke. r4d_logits_vocab.hip and r4d_gemm_iqr.hip use mma_i8/acc_i32 instead of mma8<true>/acc_val<true>, remove I8 template bool; integrate equivalent helpers preserving gfx11 accumulator, trellis and fused tile. Shared r4d_gemm_8a8_prefill.h adds ROWM/w_ld, mapped row-major vision gather and scale addressing, removes FP8 specialization. Structurally rebase extracted owned header without dropping measured fused/half-height/native operand geometry or F32-scale variants. Advance last-integrated only after proof, even no-op decisions. RCS25 overlay-build/measure and matched generated/native/ref: layout roundtrip, supported/declined forms, selected row, resource/spill, chained/unchained M1..64, M65/96/128, tails, supported N5376, cold/warm replay. Byte identity for local optimized paths; native rounding differences require separately approved reference budget.
* libr2 gfx1030 still depends on generated shared source; native gfx11 is no replacement. Retain head/tail routes, prove all build dependencies before removal.

## 5. Port order and code-level guidance

1. Establish ABI/ownership contract above. rad0016 KernelRow and loader selection: retain rank masks around new fetches_rows field and chaining. Reverse device order and reject wrong architecture. rad0017 shared rendezvous and rad0031 timeline unblock after prerequisites, not new transport.
2. Upstream renames rad_fp8.h -> rad_linear.h, rad_block_attn_gated_fp8.h -> rad_block_attn_gated_q.h, rad_block_mlp_fp8.h -> rad_block_mlp_q.h, rad_block_gdn_fp8.h -> rad_block_gdn_q.h, rad_block_moe_fp8.h -> rad_block_moe_q.h (also attention/MTP q headers). Port rad0018/19/20/25/26/27/28/30 as coherent ownership/pipeline series to new generic declarations, not old filename aliases. Re-evaluate native spans, LoRA, stage offsets, local scale groups and draft_head_off. Prove mixed one/multiple prefill sequences, unequal TP2/3, rank-consistent verify and group256.
3. rad0021/24/32/33 collide in r4d_ar_entry.hip. Reconcile one weighted/duplex/once dispatch; retain shared lifecycle, AQL/host exactness, both gfx1100/1201 rank permutations, chain flags, timeout, abort and fresh restart. Re-enable rad0031 after dependencies.
4. rad0029 and rad0036 integrate native budget/scheduler as section 2. rad0011 plan-memory lexical clean; verify native measured per-rank layout and physical capacity; existing RSC16-18 own shared planner and per-client caps, no duplicate.
5. rad0002/04 and rad0045 port to rad_block_moe_q.h current routing/expert_split; preserve ordering, protected experts, route record/replay and remote handoff. rad0043/44 clean but jointly test. rad0046 retained rows clean text is not permission to resurrect retired encoding: establish required surviving rows, archive only obsolete arms with proof. rad0008 HC/rad0012 helper clean but test new native contracts.
6. rad0048-54 and rad0042 KV: use existing bc_kv_rows.h registration, current parser/descriptor/op keys, ref refusal and h256/page16 constraints; no wholesale old FP8 block restore. rad0060 LTX include port remains diagnostic; RUM14/LTX02 own loader/issue; no TP3 activation. rad0061 RUM15 diagnostic. rad0063/64 rank-weighted head after TP dependencies; rad0055/0070 and tune/timeline clean still need runtime proof.
7. Native rewrites: r4d_gemm_8a8_prefill.h block1 old i2v bf[kPTA] anchor; new int8 body/ROWM changes structural assumptions. r4d_gemm_fp8a16_nt_m1.hip -> r4d_logits_gemm_i8.hip, r4d_gemm_fp8a8.hip -> r4d_gemm_i8a8.hip, r4d_fp8_args.h -> r4d_quant_args.h, r4d_fp8_lm.h -> r4d_lm_row.h, fused_quant/gram_accum also renamed. Update inventory, CMake and compatibility references coherently; do not create dummy old files. MXFP4 GEMV blocks 9-22 clash with TN-templated body; check remaining supported model use before port/retirement. Selftest help anchors 7/12 require current flags and real fixtures, not wording-only fix. Check patch-created bc_ar_* and bc_b16_* on actual composed source. Run compose anchor/syntax on transitive affected host/device closure; unknown widens rather than skips.

## 6. Proof boundaries and next work

Use existing MEN24/PA49/RCS19/RCS23/RCS25, RUM14/15, RSC16-18 and VTP/LTX owners. One candidate-specific ledger, no duplicate programme. After each family rerun exact patch replay/idempotence, targeted regression and footprint. All five compositions and catalog must pass. Resume existing pin-bump at drift once real consumers reconciled, no skip flags. Verify actual transitive build, installed ABI/layout, component CTests, CPU smoke, edition 3 contract and all YAML with matched native engine/model root. Discrete GPU qualification under short queued leases before installed Gate E, actual long-context admission, TTFT/decode/quality and affected power qualification. No global native or format promotion on source checks. Preserve prior artifacts and rollback. Draft PR must not merge while unqualified.

Open: exact candidate fingerprint (not measured here), full model-header and availability checks, independent semantic equivalence for every patch, implementation and all compile/device gates. No GPU or production service touched by analysis. Rebase probes used detached worktrees, did not edit shared vendor. All changes to this ledger and future ports stay in bump branch.
