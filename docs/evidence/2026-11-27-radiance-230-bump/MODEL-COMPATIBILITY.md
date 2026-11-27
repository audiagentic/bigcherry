# Current-model compatibility and encoding retirement gate

Owner decision 2026-11-27: dropping encodings is acceptable provided current models remain working. This is a release gate, not a requirement to preserve all historical encodings. Prefer existing containers and native paths. Where conversion is unavoidable, qualify the replacement before changing a managed config; do not silently disable speculation, reduce context, lose a hardware topology or degrade quality. Existing 2.2 runtime and model files remain immutable rollback. No retirement by filename, dtype label, patch failure or source-level matching alone.

## Observed current roster (bump branch, 18 YAMLs)

Read file sizes, RAD1 v2 header, string/encoding/directory tables from each actual configured container, without reading weight payloads, loading models, using GPU or hashing whole files. All four files exist. Source struct layouts from exact candidate abi/rad_format.h, rad_encoding.h, rad_types.h: little endian header 248 bytes, RadFileEntry 136 bytes (RAD_MAX_RANK=6), RadEncoding 568 bytes, RadEncPlane 64 bytes. Checked magic, version, declared file_bytes == stat size, encoding index, rank and n_planes. Counts below are directory entries, not parameter counts; draft prefix grouping is observational, actual builder is authoritative.

| Container basename | Bytes | Config count | Arch | Entries | Encodings |
|---|---:|---:|---|---:|---:|
| qwen3.8-27b-iq8r-dflash.rad | 30876340224 | 6 | qwen35 | 1101 | 5 |
| qwen3.8-27b-iq6r-v4-df2-w3.rad | 24476004352 | 2 | qwen35 | 1109 | 12 |
| qwen3.8-next-flash-fp8-iq4r-moe.rad | 121969901568 | 8 | qwen4exp | 51489 | 8 |
| qwen3.8-next-flash-hcbf16.rad | 122598637568 | 2 | qwen4exp | 51489 | 7 |

All 18 configured YAMLs request num-speculative-tokens: 3. Preserve speculative functionality; num-speculative-tokens: 0 is a diagnostic, not equivalent. Retain existing fp8, bf16 and bf16_i8 KV variants, TP2/TP3 and TP1 with up to three remote helpers, including RDNA2 head/tail and combine modes. Actual deployed processes were not queried, so this is the tracked config roster, not proof of every ad hoc model used.

### IQ8R

288 base affine I8 FWHT128 group256 U8 scale with one BF16 row factor; 735 plain BF16 and one group128 INT8/BF16 head. DFlash: one U2 asymmetric group128 head (F16 scale/U8 zero), 31 U4 FWHT128 group128 BF16-scale I8-table weights and 45 BF16 entries. No observed retired block-FP8/MXFP4/M2XFP4/ParoQuant plane set. Preserve group256 and RUM15 numerical contract; native libr3d group128 is not sufficient. Builder must prove asymmetric head and drafter layout too.

### IQ6R-V4

734 BF16 entries, five I32 head-id tables, one U6 asymmetric embedding (BF16 scale/U6 zero), one U4 and one U5 factored F32 trellis vocabulary class, 81 U6 and 113 U7 factored trellis entries (run256), 32 U6 and 32 U7 factored entries (run512), 32 U6 trellis entries with BF16 row-group128 scales (run256). All trellises have FWHT128, L16 65536-entry I8 table and U32 MAD coefficients; factored entries have F32 row and column group128 scale. DFlash entries have same encoding families/counts as IQ8R. Preserve exact U6/U7 run256/512, weighted vocabulary partition, embedding and gated/chained paths. Upstream libr3d dense U3/U4/U5 cannot replace these. No observed retired plane set.

### Flash-Next — important correction to filename-based risk

FP8-IQ4R-MoE: 50,156 base expert entries are affine U4, FWHT128, E4M3 scale [1,64] under tensor F32 scale, F32 16-entry codebook (w4nl64 family); 223 I8 group128 BF16-scale trunk/head entries, 97 E4M3 row group128 and 97 E4M3 row group80 HC entries, one E4M3 tensor-scale ngram table, 896 BF16 and 3 I64. MTP-prefixed entries: 5 E4M3 group128, 3 E4M3 group80, 7 BF16 and one U2/F16/U8 asymmetric draft head. No observed 128x128 block-FP8 weight plane, MXFP4, M2XFP4 or ParoQuant. E4M3 codes in HC/MTP rows and scale bytes in experts are NOT retired block-FP8 weights.

HCBF16: same 50,156 U4 expert entries, 223 I8 trunk/head, one E4M3 tensor-scale ngram table, 1090 BF16 and 3 I64 base. MTP 13 BF16, 2 E4M3 row group128 FC entries and one U2 asymmetric head. Again no observed retired plane set. The difference in HC mixing precision is intentional; do not replace one configuration with the other without comparing.

Candidate 3b13fd32 arch/common/rad_linear.h linear_retired checks E4M3 codes with scale block 128x128 (not row-scaled or tensor-scaled E4M3), MXFP4, M2XFP4 and Givens/ParoQuant. arch/common/rad_block_moe_q.h MoeQ::format_of recognizes exact U4/E4M3 scale [1,64]/F32 tensor-scale/F32 16-entry codebook as format 5; kServed8[5] = w4nl64a8i8h. arch/qwen4exp_fp8/qwen4exp_fp8.cpp near lines 872-890 chooses exp8 for rotated four-bit codebook g64 and codes_i8 when trunk_i8; arm8 feeds grouped int8-activation experts. This is a plausible existing-container migration (switch activation path, not weight storage). Must prove actual probe/declared graph, gate/up and down rows, protected BF16 experts, HC/MTP, remote handoff and selected device kernels. libr4d/r4d_rows.cpp includes w4nl64a8i8h; no match in native libr3d for that dtype, so retain generated/owned gfx11 provider unless proven. Do not delete rad0046 or local MoE rows based only on old format names. MTP FC explicitly supports BF16 or E4M3 row codes (rad_block_mtp_hc.h 205-225). Keep E4M3 row codecs and scale support even when dropping block-FP8 compute. FP8 KV is separate again.

Local alternate qwen3.8-next-flash-i4x-i8-hc8m.rad exists, 51489 entries, same 8 encoding families except 50,156 expert entries are rotated I4 group128 BF16 scales rather than U4 codebook. Not currently referenced by shipped YAML; candidate only if primary path fails, requires quality and capacity comparison, no automatic substitution.

## Retirement decisions

* Candidate to drop: block-FP8 weight linear compute, MXFP4/M2XFP4 trunks, ParoQuant, E4M3-activation expert paths if no retained model/graph needs them and actual replacement path passes. Preserve historical patch packages/code/results unchanged in archive with successor and evidence. Do not remove general FP8 codecs/row scales, FP8 KV, quantizer conversion readers or model file format solely because serving compute is retired.
* Must keep for current roster: IQ8R group256; U6/U7 IQR including run512, factored and BF16 scale forms; weighted mixed-bit vocabulary; U6 embedding; U2 asymmetric draft head; DFlash U4 I8-table path; Flash-Next U4 F32-codebook E4M3 scales and INT8 activation rows, protected BF16 experts, row-E4M3 HC/MTP, ngram tensor E4M3; all current TP, collective, remote MoE and cache capabilities.
* No new conversion unless native/retained path fails. Prefer same file and same model identity, then exact or quality-bounded representation change. If replacement required, record source checkpoint, conversion recipe, output hash, accepted quality/capacity/perf and all configs mapped to it. Preserve old runtime and files for rollback, not indefinite support for unused formats.

## Reviewed qualification coverage

- IQ8R: two XTX engine ranks, and XTX/XTX/R9700 unequal TP3. Include capacity and governor variants, BF16 and BF16-I8 caches.
- IQ6R: same mixed TP3, short and long settings, actual mixed head and speculative verification.
- Flash-Next FP8-IQ4R-MoE: single R9700 with host tier, XTX helpers (combine on/off), RDNA2 head and tail, four-card helper combine and 16/8 cache. Four cards here mean TP1 + three helpers, NOT TP4.
- Flash-Next HCBF16: two XTX engine ranks with R9700 head helper, and R9700 head plus RDNA2 tail with combine.

Use existing native profile allowlist, graph importer and acceptance runner. Map each of 18 YAMLs to matched candidate evidence; flashnext-2xtx-rdna2-tail-combine declares an empty differs list against flashnext-4gpu, so share execution only after native resolved equality (port/provenance exceptions). Do not infer equivalence from filenames. All other variant differences require explicit affected-path coverage. The planned matrix is capability coverage, not all possible cross-products.

## Acceptance before release

1. Existing bc-serve explain for every exact YAML with matched candidate binary; native startup unsupported weight list empty for all actually used weights and drafters; no disabled speculation or fallback to unsupported/unqualified providers. Prove rank geometry, row choice, layout and memory admission, including RDNA2.
2. Existing discrete harness tests every required encoding and actual shapes on relevant gfx1100/gfx1201/gfx1030: relayout/unrelayout, integer sums, byte identity where contract unchanged, supported/declined rows and actual graph selection. Use the same source pin for native/provider comparisons.
3. Candidate installed runtime: real generation and accepted speculative tokens; cold/warm startup, prefill, decode, concurrent/mixed batches, offload and long context, FP8/BF16/16-8 KV, mixed-device rank permutations, RDNA2 head/tail, remote MoE. Use existing RSC/RCS/VTP and release Gate E evidence, not a new harness. Compare against matched 2.2 baseline (KL/top-k and actual serving metrics); require explicit acceptance of intentional numerical changes or regressions, no guessed thresholds. Cover all managed configurations offline and every distinct affected runtime topology with real qualification; do not mark an untested topology supported.
4. Only then prune retired encodings or switch provider/config; archive originals and retain rollback. Draft PR #1713 cannot merge until this gate and the existing bump gates pass.

Status: metadata inspection complete for tracked roster, no explicit retired weight plane set observed. Not a proof of startup or generation; actual graph/row selection, installed builds, performance and GPU acceptance remain blocked by earlier ABI/patch drift. No production changes or model file changes.
