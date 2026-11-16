# rad0016_vtp14_rank_code_selection promotion evidence

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), tensor parallel over two RX 7900 XTX and an R9700 (TP3)
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the TP3 world over (XTX, XTX, R9700) starts and serves only with this chain (rad0016 per-rank code selection, rad0017 shared rendezvous, rad0018/rad0019 uneven ownership); without rad0017 any libr3+libr4d world hangs in all-reduce init (reproduced on cards, VTP14.md), the engine reports the rendezvous as shared by libr3.so and libr4d.so and the ownership split in its startup report.
Identity: TP3 6/6/12 against an exact TP2 reference over 42,761 positions: KLD mean 0.00307, p99.9 0.084, top-1 96.80%, PPL x0.9993; the production TP2 wht6 wire scores 0.00391 / 0.111 / 96.57% on the same corpus (VTP14.md, N7, build a0979ca2). Launchers qwen27b-tp3, -int8 and -capacity start from their layouts with greedy output identical to hand-flag runs.
A/B: control cannot run (mixed-library world hangs without the chain); TP3 6/6/12 serves 64-65 tok/s at about 0.05 s TTFT, correct (VTP14.md milestone table); w8a8 TP3 KLD against BF16 within TP2's median/p99/p99.9 (VTP14.md N6).
