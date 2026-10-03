# 1254_nro05_gdn_mtp_prefix_tail

Plan `NRO05`; state `untested`; requires `1253_nro04_gfx1100_bf16_chunked_gdn`.

For MTP (`K > 1`) prefill of one long sequence, this patch runs 1253's gfx1100 BF16 chunked GDN on the first `n_tokens-K` tokens, then the stock sequential kernel on the last K tokens so snapshot slots retain stock recurrence semantics. The runtime route is implemented; set `GGML_CUDA_GDN_CHUNKED=0` to opt out.

Activation evidence is `BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16` under patch tracing. Eligibility remains fail-closed for unsupported shapes/sequences.

Measured fact from `tools/lab/native-vs-patched/runs/nro05-ab2/RESULT.md`: on Qwen3.8-27B-Q8_0, dual gfx1100, `-sm tensor`, MTP `n_max=4`, 8 order-balanced pairs measured pp4096 +2.12%; decode was flat; MTP acceptance was identical in all 16 cells. This is benchmark evidence only, not contract qualification or sign-off. Contract-grade full-vocabulary MTP logprob parity remains required.

See `TESTING.md` for the remaining activation, correctness, work-equivalence, and order-balanced A/B plan.
