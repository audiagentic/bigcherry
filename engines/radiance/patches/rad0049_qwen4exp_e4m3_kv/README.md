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
