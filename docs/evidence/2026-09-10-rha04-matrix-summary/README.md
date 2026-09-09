# RHA04 current-source matrix summary

The four single-GPU 9B matrices are required-attestation captures: 72/72
cells completed with physical identity verified and clean teardown. The dual
XTX 27B matrix completed 18/18 cells in explicitly exploratory `observe` mode;
its required-attestation attempt failed closed before timing because tensor
split logs had no ordered physical PCI locators.

## 9B medians (tokens/s)

| topology | arm | pp512 | pp2048 | tg128 | tg512 |
| --- | --- | ---: | ---: | ---: | ---: |
| GPU0 gfx1100 | stock | 1290.355 | 2512.050 | 84.300 | 84.985 |
| GPU0 gfx1100 | native | 1279.175 | 2520.325 | 84.265 | 85.030 |
| GPU0 gfx1100 | replay | 1268.820 | 2523.355 | 84.260 | 84.725 |
| GPU1 gfx1100 | stock | 1303.420 | 2519.355 | 84.385 | 85.080 |
| GPU1 gfx1100 | native | 1296.765 | 2526.685 | 84.490 | 85.185 |
| GPU1 gfx1100 | replay | 1299.100 | 2529.420 | 83.835 | 84.755 |
| GPU2 gfx1201 | stock | 1240.345 | 2246.085 | 67.745 | 68.255 |
| GPU2 gfx1201 | native | 1388.285 | 2793.680 | 67.815 | 68.095 |
| GPU2 gfx1201 | replay | 1388.955 | 2791.335 | 67.825 | 68.310 |
| GPU3 gfx1030 | stock | 905.130 | 1281.160 | 55.260 | 55.815 |
| GPU3 gfx1030 | native | 892.860 | 1279.880 | 55.290 | 55.700 |
| GPU3 gfx1030 | replay | 888.725 | 1281.450 | 55.250 | 55.785 |

## Direct replay/native effects

Effects are paired geometric effects with 10,000 bootstrap resamples; intervals
are 95% bootstrap intervals.

| topology | pp512 | pp2048 | tg512 |
| --- | ---: | ---: | ---: |
| GPU0 | -2.420% (-5.316,+0.201) | +0.026% (-0.249,+0.214) | -0.360% (-0.935,+0.385) |
| GPU1 | +0.205% (-0.841,+1.195) | +0.063% (-0.165,+0.242) | -0.464% (-0.587,-0.303) |
| GPU2 | +0.451% (-0.438,+1.496) | -0.717% (-2.500,+0.322) | +0.409% (+0.169,+0.684) |
| GPU3 | -0.800% (-2.299,+0.482) | +0.018% (-0.210,+0.177) | +0.033% (-0.251,+0.318) |

## Dual-XTX 27B observe-only medians (tokens/s)

| arm | pp512 | pp2048 | tg128 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| stock | 929.250 | 1287.315 | 34.000 | 34.190 |
| native | 929.815 | 1286.975 | 33.870 | 34.100 |
| replay | 929.065 | 1286.185 | 34.175 | 34.265 |

Direct replay/native effects are pp512 -0.056% (-0.377,+0.298), pp2048
-0.024% (-0.185,+0.131), and tg512 +0.657% (+0.411,+0.990). These numbers
are not decision-grade because all 18 cells lack dual-device physical
attestation.

Per-topology raw manifests and configurations are retained in the sibling
`rha04-gpu*-full` and `rha04-27b-dual-observe` directories. No replay
promotion or parity verdict is inferred from this summary.
