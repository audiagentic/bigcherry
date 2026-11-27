# First actual runtime tests

Controller e6598253c, Radiance source 3b13fd32663ef52252a7f0be255523480aa22ffe. This candidate linked the base engine, libbcgfx12, libbciqr and libr2; libr3 device codegen failed on r4d_ds41_ggml.hip's gfx12-only s_prefetch_data. The RDNA3 native rewrite removes only the optional cache-prefetch lambda, leaving the demand loads, arithmetic, gfx12 implementation and rows intact. Exact production composition gate1 passes. Targeted codegen and complete linked build are separate checks; DS41 performance is unqualified.

## CPU smoke repair and real pass

First attempt: old flag-only invocation refused (2.3 requires YAML). Shared ServerRunner now accepts an optional argument builder, invoked with its selected free port; Radiance adapter uses the checked-in native CPU YAML fixture, filling only the selected port and pinned model path (no legacy flag translation). Model hash, GPU hiding, completion checks and clean shutdown unchanged. New receipt records actual launch args. All 32 existing/new smoke tests pass.

Second attempt: tiny.rad is labelled quant=bf16 but its fused QKV is actually retired block FP8. Engine refused it with a clear unsupported-encoding diagnostic. Preserve original tiny.rad and all production containers. New smoke fixture at smoke/tiny-iq8r-230.rad, 15 weights (9 quantized, 6 preserved), from the same original llama-68m checkpoint. The local model store contains the new fixture; other hosts must copy the exact artifact or reproduce and verify its digest before use. No large binary added to Git.

Provenance:
- source model.safetensors SHA256 a0bf4d8d47e9d27cd68543726e6e6192fbc7a2ed061f9d1c3a6cc2a5c452ffbc
- config.json SHA256 872a7781fce21a579b3dd9ba9b8f5e305db4fd2e031a7efbed3213d7e16b71f2
- tokenizer.json SHA256 bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e
- pinned output SHA256 f6221c8484f74e0942307b6d2035260200d801ff79610fa8c64618aa1bc0c6a7
- recipe tiny-iq8r.recipe, candidate rad-convert --home CANDIDATE/radiance_home --kernels libavx:libref --cpu --max-ctx 512 --max-tok 64 --max-spec 0 --recipe tiny-iq8r.recipe SMOKE_SOURCE/llama-68m -o tiny-iq8r-230.rad. Hide all GPU visibility variables, OMP_NUM_THREADS=4.
- four fused trunk rules are the upstream IQ8R group256/FWHT128 form with U8+BF16 scale; unrotated int8 group128 output. Converter declared against CPU kernels and reported success. No production conversion performed.

Actual CPU pass on the native fixture (third successful run): six completions, three matching deterministic prompt pairs, zero GPU log hits, clean SIGINT exit 0. Final receipt cpu-smoke-iq8r.json (local path/port fields normalized; binary and model hashes retained). This is evidence of real CPU inference on the base runtime only; repeat against final canonical bundle.

## Leased gfx1201 base numerical tests

Used short gpu leases, ROCr device 2/HIP ordinal 0, gfx1201 verified by host inventory, idle VRAM before each lease; second lease records before every launch. All five invocations exit 0. IQR tiled (129 ok lines), int8 M16 (13), int8 logits (28), IQ8R group256 (35 including both decode and tiled numeric plus byte-identical quantizer fusion checks). Both IQ8R selectors produced identical logs, so do not count their 35 checks twice (205 distinct ok lines across the four suites). See gfx1201-base-selftest.json and corresponding text logs, exact binary/plugin hashes retained. These exercise base libr4d only, not libbciqr or libr3 parity. No performance claim.

A separate leased GPU tiny inference startup correctly identified gfx1201 but refused head_dim=64/gqa=1: no device attn_paged row supports that geometry; 16 device bands unresolved. The tiny fixture is CPU-only and is not a GPU-serving model. No GPU completions claimed. Do not widen unsupported device row constraints to make a test pass. Next GPU model tests should use the real existing supported geometry and final linked runtime. No production serving process, power settings or existing model altered.

Remaining: finish the actual RDNA3 codegen/link, full candidate ABI/offline tests, all native config/model graph and lifecycle checks, overlay/native byte differential and KLD/performance against independent references. No release or whole plan closure claimed.
