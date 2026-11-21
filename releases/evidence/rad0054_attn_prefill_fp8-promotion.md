# rad0054_attn_prefill_fp8 promotion evidence

Model: Qwen3.8-27B MXFP4 on two R9700s at TP2, H256/GQA6/page16

Mechanics: The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically. The v2.2 Release config recheck passed 3,409 checks in 78 cases including the independent off/QK/PV/both option. RUM07 checks nine supported row combinations, rejects lossy page4 and B16, and keeps decode and DFlash2 exact. Selected gfx1201 and native gfx1100 attachment compiles are recorded in RUM07.
Activation: The shipped v1.17 lineage records historical E4M3 QK/PV device activation in cdc0ae26df0e87b69bfbae3ae7bdff31be93c001 and V4/K6V4 support in 51db508f91f08d6f7bf6959a76126e0304b34e71. The old gfx1100/gfx1201 catalogs retain all nine row names. Historical combined-mode KL is recorded for E4M3 and V4.
Identity: Historical E4M3 off mode had KL 0 over 10,198 positions versus the unmodified build. The V4 off mode reproduced the shipped libr4d's mean KL 0.006713. Historical combined mode is lossy by design, with unchanged decode and no claim of model-level identity across the modes.
A/B: Historical off-versus-both on E4M3 reported 2,621 vs 3,082–3,127 prompt tokens/s at 200K and combined mean KL 0.0076. V4 reported 2,609–2,611 vs 3,073–3,091 and combined mean KL 0.0073. Those measurements do not cover every single-leg and K6V4 combination.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)
