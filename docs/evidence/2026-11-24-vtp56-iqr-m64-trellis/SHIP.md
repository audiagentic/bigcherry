# IQR wide-tile native accumulation — review set

**Status:** qualified for review, not published or pushed.

## Included changes

The only native code delta is the IQR wide-tile managed rewrite: three narrowly matched replacement blocks preserving the upstream 16-row native-fragment optimization. No vendor files or unrelated kernels are edited. This folder supplies the path-free validation narrative and the complete paired GPU replay data in `replay-wide-median-min-max.csv`.

## Mandatory gates

| Gate | Observed outcome |
|---|---|
| Complete plugin + selftest build | PASS, successfully compiled and linked |
| Full BF16 byte identity | PASS, 18 leased GPU arms, 9 forms, chained and unchained, 7 shapes, all M=17–64, 6,048 comparisons, 0 differences |
| IQR regression selftests | PASS, 4 actual GPU cases, 2,398 successful assertions, 0 failures |
| Same-toolchain register control | PASS, 544/544 wide-tile specializations with no increased spills/private allocation or lost register-file waves, and no increased SGPRs |
| All other records in matching control | PASS, 2,625/2,625 identical resources including SGPRs |
| Cold/prefetch-warm replay | PASS, 16 short GPU arms, 560 comparisons at M=17/24/32/48/64 over 6 trunk shapes plus the ragged shape, every candidate median improved, 0 regressions beyond measured spread |
| Residual spill review | 438 spill-free, 106 residual spillers, maximum 136 private bytes. Guarded multi-tile B-fragment lifetime trial rejected before GPU: no improvements and one resource regression |

## Replay median gain over 70 groups per form and mode

| Form | Cold | Prefetch-warm |
|---|---:|---:|
| Computed MAD6 | 81.9% | 81.9% |
| Table-trellis B6 | 79.2% | 79.3% |
| Table4 | 31.7% | 32.6% |
| Grid8 | 84.8% | 84.6% |

The attached CSV records every requested form, shape, token height, chain mode and replay mode as baseline/candidate median with min–max and the observed spread. No numbers are inferred from a serving run.

## Limits and delivery

The optimized wide tile is qualified at M=17–64; M greater than 64 routes to separate prefill code and is outside this step's acceptance scope. A scratch replay initially included an unsupported M=65 ragged prefill case on *both* baseline and candidate, and was corrected to the required token heights. The optimized plugin was not changed by that harness correction.

No serving, A/B, KLD or production run was performed. The surviving code contains no added model-specific constants; the tested and rejected lifetime experiment was not committed. The review set is committed but has not been pushed, tagged, deployed or shipped.
