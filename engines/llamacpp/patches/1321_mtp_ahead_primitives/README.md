# 1321_mtp_ahead_primitives

## Promotion record

Promotion record (FMTP03 lightweight promotion tier, pin b11474, Brutus 2026-10-08). 1321/1322 are promoted together because the overlap patch requires the primitives patch. The runtime profile enables `BIGCHERRY_MTP_AHEAD=1`; the build remains opt-in outside the Flash-Next profile because evidence covers one model/topology.

build deploy-v6-plus-ahead compiled clean, smoke 0 error lines; ABBA A = production, B = BIGCHERRY_MTP_AHEAD=1, decode t/s (512 tokens):
 depth 8192: A 82.6, 83.8 | B 87.0, 86.5 (+4.3%, complete separation), accepted A 348/487 B 340/511, 339/514
 depth 24576: A 71.7, 74.8 | B 75.8, 74.5 (+2.6%, ranges overlap), accepted A 338/514, 343/499 B 330/543, 327/552
 depth 98304: A 64.8, 65.3 | B 69.5, 68.9 (+6.4%, complete separation), accepted A 349/485 B 343/503, 342/506
 prefill unchanged (1172-1177 t/s at 8K/24K; 1087 at 98K). Greedy text md5 identical A vs B at all three depths (e3833264.., 0d8c726f.., 3ee76e81..).

## Coverage limit

Verified on Flash-Next IQ4_XS with the production Brutus layout only. No claim is made for other models or topologies.
