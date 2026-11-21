# rad0052_b16_kv_geometries promotion evidence

Model: Qwen3.8-27B W8A8 on TP3 (two RX 7900 XTX and an R9700)
Model: Qwen3.8-Next-Flash HCBF16 on two RX 7900 XTX at TP2

Mechanics: The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically. RUM07 checked Qwen4exp GQA6/page4 and GQA12/page4 declarations and the Qwen35 GQA6/page16 trunk fixture with the independent native E4M3 DFlash2 cache. The latter fixture reported 100 checks.
Activation: Historical rad0041 validation identifies the Flash-Next packed GQA12/GQA6 cases. The 27B B16 run and the old rad0042 tests exercise the other retained geometry and check B16 rows separately from ordinary E4M3.
Identity: The historical GQA6 and GQA12 B16 device checks passed after the rad0042 inventory fix. RUM07's architecture fixtures keep the Qwen35 H128/GQA4/page16 drafter on native E4M3 and validate the trunk separately.
A/B: The historical VTP43/VTP44 27B long-corpus BF16-reference comparison measured mean KL 0.002685 for B16I8, 0.003228 for B16F8 and 0.004947 for FP8. Flash-Next TP2 serving and capacity are recorded in FLASHNEXT-AB.md. The Flash-Next KL in that A/B was inside the arithmetic-noise band.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)
