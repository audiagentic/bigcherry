# Kernel A/B report

Weighting: equal calls per case. B/A < 1 means B is faster.
Timing-only verdicts are unverified. Negative correctness margins or FAIL/vacuous are not wins.
Noise is the per-case absolute log ratio of A-again/A, not a fixed threshold.

| op | kernel family | geometry | operand bytes | A us | B us | B/A | A margin | B margin | A verdict | B verdict | noise log spread | status |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| attn_paged | attn_decode_h128_gqa1_bf16kv | block_size=16 causal=1 gqa=1 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 204.01 KiB | 6.630 | 6.610 | 0.997 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_decode_h128_gqa1_fp8kv | block_size=16 causal=1 gqa=1 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 108.01 KiB | 6.700 | 6.680 | 0.997 | - | - | timed | timed | 0.001494 | unverified |
| attn_paged | attn_decode_h128_gqa2_bf16kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 108.01 KiB | 6.610 | 6.650 | 1.006 | - | - | timed | timed | 0.004528 | unverified |
| attn_paged | attn_decode_h128_gqa2_fp8kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 60.01 KiB | 6.750 | 6.790 | 1.006 | - | - | timed | timed | 0.002959 | unverified |
| attn_paged | attn_decode_h128_gqa2w_bf16kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 108.01 KiB | 6.620 | 6.600 | 0.997 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_decode_h128_gqa2w_fp8kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 60.01 KiB | 6.590 | 6.580 | 0.998 | - | - | timed | timed | 0.001516 | unverified |
| attn_paged | attn_decode_h128_gqa4_bf16kv | block_size=16 causal=1 gqa=4 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 60.01 KiB | 6.790 | 6.770 | 0.997 | - | - | timed | timed | 0.002941 | unverified |
| attn_paged | attn_decode_h128_gqa4_fp8kv | block_size=16 causal=1 gqa=4 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 36.01 KiB | 6.790 | 6.760 | 0.996 | - | - | timed | timed | 0.001474 | unverified |
| attn_paged | attn_decode_h128_gqa8_bf16kv | block_size=16 causal=1 gqa=8 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 36.01 KiB | 6.620 | 6.650 | 1.005 | - | - | timed | timed | 0.004521 | indistinguishable |
| attn_paged | attn_decode_h128_gqa8_fp8kv | block_size=16 causal=1 gqa=8 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 24.01 KiB | 6.730 | 6.880 | 1.022 | - | - | timed | timed | 0.002967 | unverified |
| attn_paged | attn_decode_h256_gqa12_bf16kv | block_size=4 causal=1 gqa=12 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 32.01 KiB | 8.700 | 8.710 | 1.001 | - | - | timed | timed | 0.002301 | indistinguishable |
| attn_paged | attn_decode_h256_gqa12_fp8kv | block_size=4 causal=1 gqa=12 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 28.01 KiB | 7.970 | 7.970 | 1.000 | - | - | timed | timed | 0.003771 | indistinguishable |
| attn_paged | attn_decode_h256_gqa2_bf16kv | block_size=16 causal=1 gqa=2 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 216.01 KiB | 8.700 | 8.670 | 0.997 | - | - | timed | timed | 0.003442 | unverified |
| attn_paged | attn_decode_h256_gqa2_fp8kv | block_size=16 causal=1 gqa=2 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 120.01 KiB | 8.790 | 8.570 | 0.975 | - | - | timed | timed | 0.023016 | unverified |
| attn_paged | attn_decode_h256_gqa4_bf16kv | block_size=16 causal=1 gqa=4 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 120.01 KiB | 8.220 | 8.180 | 0.995 | - | - | timed | timed | 0.004878 | indistinguishable |
| attn_paged | attn_decode_h256_gqa4_fp8kv | block_size=16 causal=1 gqa=4 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 72.01 KiB | 7.860 | 7.840 | 0.997 | - | - | timed | timed | 0.002548 | indistinguishable |
| attn_paged | attn_decode_h256_gqa6_bf16kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 88.01 KiB | 8.850 | 8.790 | 0.993 | - | - | timed | timed | 0.001129 | unverified |
| attn_paged | attn_decode_h256_gqa6_bs4_bf16kv | block_size=4 causal=1 gqa=6 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 40.01 KiB | 8.920 | 8.980 | 1.007 | - | - | timed | timed | 0.007817 | indistinguishable |
| attn_paged | attn_decode_h256_gqa6_bs4_fp8kv | block_size=4 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 32.01 KiB | 8.140 | 8.210 | 1.009 | - | - | timed | timed | 0.007344 | unverified |
| attn_paged | attn_decode_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=1 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 8.940 | 8.800 | 0.984 | - | - | timed | timed | 0.011249 | unverified |
| attn_paged | attn_decode_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=2 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 8.900 | 8.860 | 0.996 | - | - | timed | timed | 0.006764 | indistinguishable |
| attn_paged | attn_decode_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=3 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 8.900 | 8.850 | 0.994 | - | - | timed | timed | 0.007896 | indistinguishable |
| attn_paged | attn_decode_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 8.930 | 8.820 | 0.988 | - | - | timed | timed | 0.007870 | unverified |
| attn_paged | attn_decode_h256_gqa8_bf16kv | block_size=16 causal=1 gqa=8 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 72.01 KiB | 8.300 | 8.290 | 0.999 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_decode_h256_gqa8_fp8kv | block_size=16 causal=1 gqa=8 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 48.01 KiB | 7.820 | 7.790 | 0.996 | - | - | timed | timed | 0.001280 | unverified |
| attn_paged | attn_prefill_h128_gqa1_bf16kv | block_size=16 causal=1 gqa=1 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 204.01 KiB | 5.390 | 5.310 | 0.985 | - | - | timed | timed | 0.020619 | indistinguishable |
| attn_paged | attn_prefill_h128_gqa1_fp8kv | block_size=16 causal=1 gqa=1 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 108.01 KiB | 5.180 | 5.180 | 1.000 | - | - | timed | timed | 0.007752 | indistinguishable |
| attn_paged | attn_prefill_h128_gqa2_bf16kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 108.01 KiB | 5.100 | 5.120 | 1.004 | - | - | timed | timed | 0.003914 | indistinguishable |
| attn_paged | attn_prefill_h128_gqa2_fp8kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 60.01 KiB | 4.970 | 5.010 | 1.008 | - | - | timed | timed | 0.006018 | unverified |
| attn_paged | attn_prefill_h128_gqa2w_bf16kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 108.01 KiB | 5.240 | 5.240 | 1.000 | - | - | timed | timed | 0.001907 | indistinguishable |
| attn_paged | attn_prefill_h128_gqa2w_fp8kv | block_size=16 causal=1 gqa=2 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 60.01 KiB | 5.200 | 5.190 | 0.998 | - | - | timed | timed | 0.001921 | unverified |
| attn_paged | attn_prefill_h128_gqa8_bf16kv | block_size=16 causal=1 gqa=8 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 36.01 KiB | 5.090 | 5.100 | 1.002 | - | - | timed | timed | 0.005877 | indistinguishable |
| attn_paged | attn_prefill_h128_gqa8_fp8kv | block_size=16 causal=1 gqa=8 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 24.01 KiB | 5.010 | 5.020 | 1.002 | - | - | timed | timed | 0.001994 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa12_bf16kv | block_size=4 causal=1 gqa=12 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 32.01 KiB | 11.110 | 11.130 | 1.002 | - | - | timed | timed | 0.003594 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa12_fp8kv | block_size=4 causal=1 gqa=12 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 28.01 KiB | 11.460 | 11.450 | 0.999 | - | - | timed | timed | 0.001744 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa2_bf16kv | block_size=16 causal=1 gqa=2 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 216.01 KiB | 11.130 | 11.070 | 0.995 | - | - | timed | timed | 0.001799 | unverified |
| attn_paged | attn_prefill_h256_gqa2_fp8kv | block_size=16 causal=1 gqa=2 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 120.01 KiB | 11.210 | 11.180 | 0.997 | - | - | timed | timed | 0.000892 | unverified |
| attn_paged | attn_prefill_h256_gqa4_bf16kv | block_size=16 causal=1 gqa=4 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 120.01 KiB | 10.710 | 10.690 | 0.998 | - | - | timed | timed | 0.000934 | unverified |
| attn_paged | attn_prefill_h256_gqa4_fp8kv | block_size=16 causal=1 gqa=4 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 72.01 KiB | 11.030 | 10.990 | 0.996 | - | - | timed | timed | 0.001815 | unverified |
| attn_paged | attn_prefill_h256_gqa6_bf16kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 88.01 KiB | 11.010 | 10.900 | 0.990 | - | - | timed | timed | 0.024829 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa6_bs4_bf16kv | block_size=4 causal=1 gqa=6 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 40.01 KiB | 11.070 | 11.200 | 1.012 | - | - | timed | timed | 0.009074 | unverified |
| attn_paged | attn_prefill_h256_gqa6_bs4_fp8kv | block_size=4 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 32.01 KiB | 11.120 | 11.420 | 1.027 | - | - | timed | timed | 0.011623 | unverified |
| attn_paged | attn_prefill_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=1 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 12.730 | 12.690 | 0.997 | - | - | timed | timed | 0.007886 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=2 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 12.760 | 12.870 | 1.009 | - | - | timed | timed | 0.007078 | unverified |
| attn_paged | attn_prefill_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=3 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 12.730 | 12.650 | 0.994 | - | - | timed | timed | 0.003935 | unverified |
| attn_paged | attn_prefill_h256_gqa6_fp8kv | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 12.800 | 12.680 | 0.991 | - | - | timed | timed | 0.009419 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa6_fp8kv_pv8 | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=2 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 8.990 | 8.960 | 0.997 | - | - | timed | timed | 0.004459 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa6_fp8kv_qk8 | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=1 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 6.920 | 6.880 | 0.994 | - | - | timed | timed | 0.005797 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa6_fp8kv_qk8pv8 | block_size=16 causal=1 gqa=6 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 prefill_fp8=3 q_dtype=bf16 q_len=1 splits=0 window=0 | 56.01 KiB | 6.110 | 6.090 | 0.997 | - | - | timed | timed | 0.001635 | unverified |
| attn_paged | attn_prefill_h256_gqa8_bf16kv | block_size=16 causal=1 gqa=8 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 72.01 KiB | 11.010 | 11.010 | 1.000 | - | - | timed | timed | 0.000000 | indistinguishable |
| attn_paged | attn_prefill_h256_gqa8_fp8kv | block_size=16 causal=1 gqa=8 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 48.01 KiB | 11.410 | 11.400 | 0.999 | - | - | timed | timed | 0.000876 | unverified |
| attn_paged | attn_prefill_h64_gqa4_bf16kv | block_size=16 causal=1 gqa=4 head_dim=64 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 30.01 KiB | 3.980 | 3.970 | 0.997 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_prefill_h64_gqa4_fp8kv | block_size=16 causal=1 gqa=4 head_dim=64 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 18.01 KiB | 3.920 | 3.910 | 0.997 | - | - | timed | timed | 0.002554 | indistinguishable |
| attn_paged | attn_window_h128_gqa1_bf16kv | block_size=16 causal=1 gqa=1 head_dim=128 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 204.01 KiB | 6.760 | 6.720 | 0.994 | - | - | timed | timed | 0.004448 | unverified |
| attn_paged | attn_window_h128_gqa1_fp8kv | block_size=16 causal=1 gqa=1 head_dim=128 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 108.01 KiB | 6.580 | 6.570 | 0.998 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h256_gqa1_bf16kv | block_size=16 causal=1 gqa=1 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 408.01 KiB | 7.970 | 7.950 | 0.997 | - | - | timed | timed | 0.002513 | indistinguishable |
| attn_paged | attn_window_h256_gqa1_fp8kv | block_size=16 causal=1 gqa=1 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 216.01 KiB | 7.630 | 7.610 | 0.997 | - | - | timed | timed | 0.002625 | indistinguishable |
| attn_paged | attn_window_h256_gqa2_bf16kv | block_size=16 causal=1 gqa=2 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 216.01 KiB | 8.630 | 8.600 | 0.997 | - | - | timed | timed | 0.001159 | unverified |
| attn_paged | attn_window_h256_gqa2_fp8kv | block_size=16 causal=1 gqa=2 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 120.01 KiB | 8.860 | 8.830 | 0.997 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h256_gqa4_bf16kv | block_size=16 causal=1 gqa=4 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 120.01 KiB | 11.240 | 11.160 | 0.993 | - | - | timed | timed | 0.007143 | indistinguishable |
| attn_paged | attn_window_h256_gqa4_fp8kv | block_size=16 causal=1 gqa=4 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 72.01 KiB | 9.700 | 9.630 | 0.993 | - | - | timed | timed | 0.006205 | unverified |
| attn_paged | attn_window_h256_gqa8_bf16kv | block_size=16 causal=1 gqa=8 head_dim=256 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 72.01 KiB | 12.020 | 11.960 | 0.995 | - | - | timed | timed | 0.007516 | indistinguishable |
| attn_paged | attn_window_h256_gqa8_fp8kv | block_size=16 causal=1 gqa=8 head_dim=256 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=1 | 48.01 KiB | 9.980 | 9.930 | 0.995 | - | - | timed | timed | 0.004016 | unverified |
| attn_paged | attn_window_h512_gqa16_bf16kv | block_size=16 causal=0 gqa=16 head_dim=512 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=32 q_dtype=bf16 q_len=1 splits=0 window=2048 | 128.01 KiB | 10.190 | 10.170 | 0.998 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h512_gqa16_fp8kv | block_size=16 causal=0 gqa=16 head_dim=512 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=32 q_dtype=bf16 q_len=1 splits=0 window=2048 | 96.01 KiB | 10.120 | 10.090 | 0.997 | - | - | timed | timed | 0.001978 | unverified |
| attn_paged | attn_window_h512_gqa1_bf16kv | block_size=16 causal=1 gqa=1 head_dim=512 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 816.01 KiB | 7.630 | 7.600 | 0.996 | - | - | timed | timed | 0.001311 | unverified |
| attn_paged | attn_window_h512_gqa1_fp8kv | block_size=16 causal=1 gqa=1 head_dim=512 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 432.01 KiB | 7.200 | 7.180 | 0.997 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h512_gqa2_bf16kv | block_size=16 causal=1 gqa=2 head_dim=512 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 432.01 KiB | 8.710 | 8.680 | 0.997 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h512_gqa2_fp8kv | block_size=16 causal=1 gqa=2 head_dim=512 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 240.01 KiB | 8.500 | 8.470 | 0.996 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h512_gqa4_bf16kv | block_size=16 causal=1 gqa=4 head_dim=512 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 240.01 KiB | 8.680 | 8.660 | 0.998 | - | - | timed | timed | 0.000000 | unverified |
| attn_paged | attn_window_h512_gqa4_fp8kv | block_size=16 causal=1 gqa=4 head_dim=512 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 144.01 KiB | 8.510 | 8.480 | 0.996 | - | - | timed | timed | 0.001176 | unverified |
| attn_paged | attn_window_h512_gqa8_bf16kv | block_size=16 causal=1 gqa=8 head_dim=512 kv_dtype=bf16 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 144.01 KiB | 8.830 | 8.800 | 0.997 | - | - | timed | timed | 0.001132 | unverified |
| attn_paged | attn_window_h512_gqa8_fp8kv | block_size=16 causal=1 gqa=8 head_dim=512 kv_dtype=fp8_e4m3 max_ctx=262144 max_seqs=8 n_head=24 q_dtype=bf16 q_len=1 splits=0 window=0 | 96.01 KiB | 8.730 | 8.690 | 0.995 | - | - | timed | timed | 0.000000 | unverified |

## Op-family summary

| family | weighted B/A | cases | calls/step | unchecked cases | invalid cases | missing noise |
|---|---:|---:|---:|---:|---:|---:|
| attn_paged | 0.998 | 73 | 73 | 73 | 0 | 0 |

## Only A (0)

None.

## Only B (0)

None.

## Only A vs A-again (0)

None.

## Only A-again vs A (0)

None.

## Missing ops on A (0)

None.

## Missing ops on B (0)

None.
