# IQ6R-V4 single-tensor converter validation — step 2

- PASS: a representative calibrated attention projection and FFN-down projection converted on the gfx1201 device, each under its own queued short lease.
- Attention: blk.23.attn_v.weight (input K=5120); 5.15 s total, 4 s reported quantisation/write; 3.82 MiB weight payload, 11.82 MiB RAD2 container, relative Frobenius error 1.974e-02.
- FFN-down: blk.23.ffn_down.weight (input K=17408); 33.32 s total, 32 s reported quantisation/write; 63.83 MiB weight payload, 71.83 MiB RAD2 container, relative Frobenius error 1.883e-02.
- Both converter commands returned 0; the container inspector parsed both RAD2 files (exit 0).
- The quantiser binary's GPU offload bundle includes gfx1201. Device isolation was enforced.
- Both diagnostic containers deliberately defer 1,108 other weights and cannot be served until completed. No full-model conversion, serving or quality benchmark was run.
