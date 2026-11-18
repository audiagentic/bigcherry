# rad0042 selftest overlay: inventory and gqa6/block-4 cases, on a gfx1100 device (VTP44)

The GPT review of the KV patches (rad0037, 0038, 0040, 0041, 0042) found no key/value offset, scale-placement, quantiser or ordering defect and
three P2 findings about validation and test coverage. Two were checked on the device; the third (the geometry table is keyed by layout, head_dim
and gqa, not block size and operation) is a diagnostic-contract gap that the row constraints already refuse later, and is left as a follow-up.

- Before (`before-summary.txt`, libr3's selftest built from the production tree, one 7900 XTX, one case at a time): the gqa6 and gqa12 decode rows,
  the gqa6 prefill rows and both store kernels pass for both layouts; `bf16_i8_rows` FAILS (`before-bf16_i8_rows.txt`): its list of four rows rejects
  the six rows rad0041 added (gqa 12, and gqa 6 at four ranks, over 4-token pages). The cases reported `rc=2` there are "skip -- no smoke case"
  (the runner then counted them as failures).
- After (`after-summary.txt`, the overlay with the ten-row inventory and the gqa6/block-4 fold flag): all 21 cases pass, `bf16_i8_rows` among them
  (10 checks), and the two gqa6/block-4 fused rows (`attn_decode_gq_h256_gqa6_bs4_<layout>`) now run (8 checks each) against their unfused pair.
  The gqa12 prefill and fused rows run too (117 and 8 checks).
- The libr3 rewrite step's exact-count checks passed with the new selftest text.
- The scripts: `before-run_r3_selftest.sh`, `run_selftest2.sh`. Absolute paths are replaced by placeholders.
