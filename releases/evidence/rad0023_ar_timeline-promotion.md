# rad0023_ar_timeline promotion evidence

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700 (host-staged transport, no peer access)
Mechanics: tools/tests/patch/test_vtp31_ar_timeline.py passes and patch-lint is clean on the promotion PR; the lane builds carrying it build and serve.
Activation: with R4D_AR_TIMELINE=DIR the engine wrote per-rank timeline rings (ar-tl-r<rank>-ws3-p<pid>-i<n>.bin) and tools/lab/radiance/ar-timeline.py matched them by order and seq; VTP31.md experiment 2 (build 9e3cf8c7, logs tl-1011) split the one-shot into push, wait and fold (R9700 fold the long phase, about 120 us for 640 KiB; it arrives first and waits about 95 us for the XTX pushes).
Identity: diagnostic only; unset or empty, the launcher passes a null ring: no stamp, no store, the same work and the same bits; the TP3 greedy hashes and rad0024 results in VTP31.md were taken on builds that carry it.
A/B: no speed claim (off by default); the stamps are the result: per-call push/wait/fold totals in VTP31.md experiment 2 decided rad0024 (+5.0/+3.5/+6.1% at 4/6/8 streams).
