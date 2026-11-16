# rad0020_vtp14_tp3_pipelined_prefill promotion evidence

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the scheduler's "runs serially" note is gone in every arm that carries rad0025; rad0026 and rad0027 are the mixed-step shapes (decode beside two or more prefills, decode beside one cut prefill), each exercised by the workload that makes that shape (VTP31.md experiments 8, 9, 11).
Identity: pipelined steps compute the serial step's bytes; greedy texts differ only where the base build's own runs differ (VTP31.md: rad0025-0027 diverge only where the base build's runs diverge); rad0020 output unchanged.
A/B: 2x ABBA, one binary per arm, same settings. rad0020 4096-token TTFT -5% (VTP14.md); rad0025 four 4096-token prompts -7.0% slowest TTFT, 4 streams +4.0%, 1/8 streams 0; rad0026 4/6/8 streams +7.5/+10.0/+10.6%, four-prompt TTFT -6.7%; rad0027 8192-token prompt beside 4 streams TTFT -19.2%, standard bench unchanged. Logs e8-1223, e9-1328, e11-1421 (VTP31.md).
