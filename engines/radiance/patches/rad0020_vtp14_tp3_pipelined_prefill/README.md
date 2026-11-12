# rad0020_vtp14_tp3_pipelined_prefill

## Summary

`qwen35_fp8` (and the builds that include it: `qwen35_w8a8`) pipelines a wide prefill step (`step_pipelined`). A step
of 1024 rows or more runs as two pieces, and each block's all-reduce for one piece runs on the second lane
(`rad_lane`) beside the other piece's GEMMs. The pipelined step computes the serial step's bytes.

Upstream enables it at two ranks only, because two is the only rank count where it was timed. The native TP3
ownership (rad0018, rad0019) therefore prefilled serially, while the TP2 baseline it is compared with pipelined. Its
time to first token was 17–18% higher, and the R9700's x4 link makes its all-reduces the longest of the three ranks.

This patch enables the pipelined prefill at three ranks when `--tp-ownership` is in force (`Geom::tp_owned`). An even
three-rank split without a profile is unchanged.

## Outcome

Untested (not promoted). To be measured: time to first token on prompts long enough for a 1024-row step, TP3 with and
without this patch against TP2.

## Detail

- `pipe_on` becomes `(world == 2 || (world == 3 && tp_owned)) && max_tok >= kPipeMinRows`.
- The note that an int8 trunk prefills serially (its boundary norms write the E4M3 pair only) also prints at three
  ranks.
- Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0019.
