# rad0028_rank_consistent_block_drafts promotion evidence

Model: Qwen3.8-27B MXFP4 and int8 (.rad), TP3 over two RX 7900 XTX and an R9700, DFlash2 drafter for rad0028
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: rad0028 all-gathers the drafted tokens and every rank proposes from rank 0's (the cross-rank disagreement diagnostic found 166 differing rows on rank 2 before; VTP31.md experiment 13); rad0029 reports the per-rank KV pool above the common pool (VTP33.md).
Identity: rad0029 greedy answers identical and TP2 on two alike cards unchanged (VTP33.md); rad0028 is a fault fix: int8 TP3 stress runs faulted 11 of 22 before, 0 of 12 with rad0028 (VTP31.md).
A/B: rad0028 2x ABBA 1/4/6/8 streams 83.4/189.6/179.9/213.4 before vs 83.8/188.5/179.8/212.8 tok/s after (within 0.6%), TTFT +0.2%; rad0029 holds 5.5 full-length 262k sessions against TP2's 3.3 (+67%) at unchanged 4/6/8-stream throughput (VTP33.md, logs cap-1812 and cap-1839).
