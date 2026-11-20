# rad0049_qwen4exp_e4m3_kv

## Summary

The patch restores the deployed H256/GQA12 and GQA6/page4 E4M3 rows, store and attention wiring without aliasing the format to I8.

## Outcome

Validated on the promotion evidence below.

## Detail

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-Next-Flash HCBF16 on two RX 7900 XTX at TP2

Mechanics: The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically. RUM07's Qwen4exp E4M3 architecture fixture reported 170 checks on the preceding ABI44 composition, covering inherited store/reader declarations. The v2.2 attachment retains the H256/GQA12/page4 and GQA6/page4 row constraints.
Activation: The historical Flash-Next stock and patched TP2 runs selected ordinary FP8 E4M3. The v1.17 ancestor 7cf4c005e18aedee67f2667899f264e1be803b8c identifies the GQA6/page4 unfused TP4 rows. The deployed baseline in IDENTITY.md exercises the TP2 FP8 path.
Identity: In the historical Flash-Next TP2 FP8 comparison, mean KL was 3.3e-11 with 100% top-1 over 4,161 positions. Both arms used FP8 cache.
A/B: VTP44 IDENTITY.md compares the stock and packed-layout-patched binaries on the same old Flash-Next model, context and FP8 reference. It establishes preservation of this cache path. RUM09 owns v2.2 deployment throughput.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)


### RUM11: Qwen 27B TP3 page-16 E4M3 store

A separate qk_norm_rope_gate_kv_store_h256_page16_fp8 row selects
only BF16-Q, H256, page16 and native fp8_e4m3. Head counts and rotary
width are not pinned by the dispatch row: Qwen 27B TP3 uses 6/1 on the
XTX ranks and 12/2 on the R9700 rank; other layouts may differ.
The existing page4 cQkNormRopeGateKvQ4Fp8 constraint, both page4
attention paths, and the streamed row are unchanged.

The resident fused kernel (r4d_qk_norm_rope_gate.hip) uses kvs.block_size
in the block and offset computation, sizes each row as nkv * block_size *
(2 * HD) bytes and clamps each value to the E4M3 finite range before
packing. r4d_shape_qk_norm_rope_gate_kv_store derives its cache shape from
the caller's block size; r4d_fuse_qk_norm_rope_gate_kv_store mirrors it
to the reference kv_store. Neither mandates page4. The gfx1100 libr3
rewrite rules do not replace the fused-store body or its E4M3 row.
The launch checks in r4d_qk_norm_rope_gate.hip:450-462 (qk_body:
n_head >= 0, n_head_kv >= 1, head_dim in {64,128,256}, and
2 <= rot <= head_dim with even rot) provide runtime validation; they do not require a fixed
head count or a fixed rotary width.

The two dedicated cases of qk_kv_fold_case use H256, local heads
6/1 and 12/2, rotary 64 and page16. Both compare Q, rotated K and
every cache byte against qk_norm_rope_gate followed by kv_store_fp8, including negative and
out-of-range slots, under both gfx1100 and gfx1201 selftest runners.
This is a runnable test, not a claim of a GPU pass. RUM11's actual card
selftest, TP3 cold start and KLD/top-1 gate remain for the release owner.
