# rad0050_b16_kv_runtime_contract promotion evidence

Model: Qwen3.8-27B W8A8 on TP3 (two RX 7900 XTX and an R9700)
Model: Qwen3.8-Next-Flash HCBF16 on two RX 7900 XTX at TP2

Mechanics: The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically. The RUM07 Release runtime fixture checked CLI and YAML selection, per-layer resident allocation and binding, unsupported tuples, and warning-plus-resident fallback. At H256/page4 and two local heads it measured 6,144 and 6,272 bytes for B16F8 and B16I8.
Activation: Historical rad0037/38/40 promotion records report working B16 CLI selection, page size, and capacity on 27B and Flash-Next. Flash-Next attention pages were 36.00 and 36.75 KiB for B16F8 and B16I8, versus 48.00 KiB for BF16.
Identity: The historical VTP44 default-cache controls agreed at the KL scorer float floor with 100% top-1 on Flash-Next and 27B. See IDENTITY.md and EQUALITY-27B.md.
A/B: The historical VTP43/VTP44 27B long-corpus BF16-reference comparison measured mean KL 0.002685 for B16I8, 0.003228 for B16F8 and 0.004947 for FP8. Flash-Next TP2 serving and capacity are recorded in FLASHNEXT-AB.md. The Flash-Next KL in that A/B was inside the arithmetic-noise band.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)
