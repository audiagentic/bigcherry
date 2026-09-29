# rd33 (1241) vs control, Qwen3.8-27B-Q8_0, dual 7900 XTX, MTP n_max=4

Single variable: 1241_rd33_mmvq_q8_0_f32_decode (experiment rd33-only). 4 order-balanced pairs, sigint shutdown.

VOID for throughput attribution: MTP draft acceptance differs (rd33 0.9558 vs control 0.9010), i.e. the arms
generated different work. tg512 +1.85% and tg2048 -5.51% are therefore unattributable. This reproduces the
combo (combo-ab1) divergence, so 1241 is the divergent component: it changes greedy output.

Required next: fixed-work plain `llama-bench -sm tensor` decode comparison (no MTP) and a logprob-divergence check.
Not attributed: any throughput effect of 1241. Not established: whether the divergence is benign rounding or a defect.
