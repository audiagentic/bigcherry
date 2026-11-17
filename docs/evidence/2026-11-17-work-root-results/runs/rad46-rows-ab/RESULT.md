# RAD46: tail large lane on the rows kernel vs grouped GEMM (R3_MOE_WORKER_ROWS=0)
Main build 78704e2fa (radiance 1.16), production flashnext-4gpu layout, ABBA x2, same build both arms (HELPER_WRAP sets ROWS=0 on the tail only; tail log confirms "few-rows prefill kernel" only in rows arms).
- decode 1 stream (4 prompts x 4 arms): rows 24.66 ms/step, 129.1 tok/s | GEMM 24.61 ms, 129.4 tok/s
- decode 6 streams: rows 47.89 ms, 328.8 tok/s | GEMM 47.19 ms, 332.1 tok/s
- prefill TTFT 512 tokens: rows 389 ms | GEMM 707 ms (+82%)
- prefill TTFT 6 x 2048 tokens: rows 5.64 s | GEMM 9.13 s (+62%)
- (8k single-prompt probe hit the prefix cache on repeats; not used)
Conclusion: grouped GEMM passes KLD (codex/rad46 notes) but costs 62-82% prefill on the 6900 XT (no matrix instructions; emulated tile). Not an interim fix. RAD46 needs the accurate rows kernel (codex/rad46-outputlane: passes KLD, -31.5% prefill) brought to the original kernel's speed.
