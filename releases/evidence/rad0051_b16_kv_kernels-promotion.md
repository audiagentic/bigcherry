# rad0051_b16_kv_kernels promotion evidence

Model: Qwen3.8-27B W8A8 on TP3 (two RX 7900 XTX and an R9700)
Model: Qwen3.8-Next-Flash HCBF16 on two RX 7900 XTX at TP2

Mechanics: The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically. RUM07 checked exact H256/GQA12/page4 registry linkage, 768/784 bytes per head, separate BF16 QSA storage and one trunk/MTP kv_attn group. Selected gfx1201 store, attention, registry and shape units compiled during RUM07.
Activation: VTP44 DEVICE-AB.md records 274 additional passing gfx1201 rows and no failed rows, including the B16 fused-versus-standalone GQA12 byte check. The historical gfx1100 diagnostic ran all 21 focused B16 selftest cases successfully after the inventory fix.
Identity: Historical rad0042 selftest and DEVICE-AB.md supply device row and store reference agreement. B16F8 and B16I8 deliberately use different V codecs. The historical VTP44 default-cache controls agreed at the KL scorer float floor with 100% top-1 on Flash-Next and 27B. See IDENTITY.md and EQUALITY-27B.md.
A/B: The historical VTP43/VTP44 27B long-corpus BF16-reference comparison measured mean KL 0.002685 for B16I8, 0.003228 for B16F8 and 0.004947 for FP8. Flash-Next TP2 serving and capacity are recorded in FLASHNEXT-AB.md. The Flash-Next KL in that A/B was inside the arithmetic-noise band.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)
