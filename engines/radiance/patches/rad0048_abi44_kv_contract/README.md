# rad0048_abi44_kv_contract

## Summary

The patch appends and fingerprints the local KV layout/capability fields and defines one descriptor table for V4, K6V4, B16F8 and B16I8.

## Outcome

Validated on the promotion evidence below.

## Detail

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B W8A8 on TP3 (two RX 7900 XTX and an R9700)
Model: Qwen3.8-Next-Flash HCBF16 on two RX 7900 XTX at TP2

Mechanics: The v2.2 Release ABI44 layout recheck passed 23 checks over six cases, including C/C++ agreement and a reference-row fingerprint. The matched SDK exposes ABI44 fingerprint 5cc71b9c5bb7ab40. The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically.
Activation: The historical rad0040 descriptor selected the B16 formats and drove their observed page sizes. V4 and K6 selection and capacity are recorded in the v1.17 ancestor 1823ccd5e9b5e77ebfaf20a37c8329e76a50f725. The ABI44 port describes those formats centrally.
Identity: The historical VTP44 default-cache controls agreed at the KL scorer float floor with 100% top-1 on Flash-Next and 27B. See IDENTITY.md and EQUALITY-27B.md. RUM07 checks the new descriptor's distinct E4M3, I8, V4, K6V4, B16F8 and B16I8 row extents.
A/B: The historical VTP43/VTP44 27B long-corpus BF16-reference comparison measured mean KL 0.002685 for B16I8, 0.003228 for B16F8 and 0.004947 for FP8. Flash-Next TP2 serving and capacity are recorded in FLASHNEXT-AB.md. The Flash-Next KL in that A/B was inside the arithmetic-noise band.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)
