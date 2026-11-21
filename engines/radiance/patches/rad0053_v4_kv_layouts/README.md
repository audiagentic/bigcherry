# rad0053_v4_kv_layouts

## Summary

The patch ports the historical V4/K6 stores, readers and geometry rows using descriptor-owned offsets and preserves the tested standalone and fused scale arithmetic.

## Outcome

Validated on the promotion evidence below.

## Detail

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B on two R9700s, H256/GQA6 with V4 and K6V4 cache

Mechanics: The exact v2.2 production rebase at b6b8751cccb0e834a13ae9e5afca824ee9a487aa passed all 32 ordered packages. The seven-package replay fixture reapplied the chain byte-identically. RUM07's retained codec comparison checked old and ported writer/reader scale rounding, clamps, nonfinite handling and K6 bit positions. The host vector reconstructs all 64 K6 codes and the runtime fixture checks 400/352 bytes per H256 head.
Activation: The historical v1.17 ancestors c00a650bce641073000068f5b66e594806677712 and 1823ccd5e9b5e77ebfaf20a37c8329e76a50f725 record V4 and K6V4 store, fused store, attention and 27B execution. The latter reports 664,400 and 755,184 cache tokens respectively.
Identity: Historical c00a650b records folded and standalone store byte agreement and the selftest's representable-set and attention-oracle checks. The tracked RAD53 gfx1100 run reports zero failed prefill and split rows across the four historical cache layouts. RUM07's host reference checks cover the changed codec attachment.
A/B: The historical 17-document 27B comparison reported mean KL against BF16 of 0.039963 for FP8, 0.040076 for V4 and 0.039965 for K6. The tracked RAD53 card benchmark includes all three cache types and the BF16 control.

Focused Radiance 2.2 attachment checks and historical behavior evidence. Integrated deployment KL and parity are tracked separately in RUM09.

[Historical sources and port checks](../../../../docs/evidence/radiance-latest-bump/RETAINED-KV-G4.md)
