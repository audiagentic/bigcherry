---
id: RV4225
review-of: RAD46
plan: patching-radiance-rdna3
state: closed
reviewed-by: codex
reviewed-at: ''
---

# Rows precision fix improves but does not pass tail quality

## Notes



## Findings

A captured RDNA2 layer-6 request showed split-rank relative L2 of 0.00679 and 0.00899 against the corrected CPU reference. Float scaling reduced this to 0.00160 and 0.000648; XTX reference errors were 0.000810 and 0.000463. This request was captured on another prompt and does not identify the KLD outliers. The attested production-layout capture retained the same helper-off and two-XTX p99.9 values as the prior run, but tail p99.9 fell from 4.36970 to 3.71049. Its increase over helper-off is 0.91820, exceeding the 0.5 limit.

## Conclusion

Keep RAD46 pending; investigate the remaining error and repeat the full quality and throughput gates.
