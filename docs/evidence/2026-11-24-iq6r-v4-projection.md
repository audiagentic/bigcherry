# IQ6R-V4 one-pass projection and resource gate — step 4

## Time model and qualification

- An intentionally provisional two-point fit uses the measured 5.15 s attention and 33.32 s FFN-down conversions: T (seconds/tensor) = 3.389375 + 0.33581257 * millions of source-weight elements.
- The planned encodings and quantised payload sizes imply 292 GPTQ tensors. Estimated GPTQ-only duration: 9,608 s (2 h 40 min).
- Class contributions: SSM 96 tensors, 2,194 s (36.6 min); attention 64, 784 s (13.1 min); FFN gate/up 64, 4,054 s (67.6 min); FFN down 64, 2,135 s (35.6 min); four vocabulary head classes, 441 s (7.4 min).
- The further 33 RTN weights (including the draft and embedding) and 784 untouched weights, setup, reading and full-file writing are not covered by measured timings. A full-run duration must therefore not be treated as qualified; allow substantial margin over the 2 h 40 min extrapolation.
- Additional class measurements were prevented by an existing shared-device queue. No full run was initiated.

## Capacity

- Plan-only predicted a 22.53 GiB container. At measurement time approximately 151 GB of disk space remained on the work volume; the existing 94 GB calibration set and target/draft input files are read-only and do not require duplication.
- The largest measured calibration dimension is K=17,408, so one dense float32 Hessian is about 1.13 GiB. Multiple factor buffers and tensor data may coexist; peak full-run RAM/VRAM demand has not been measured.
- The machine reported 188 GiB total RAM and about 173 GiB available when checked, with approximately 31.9 GiB VRAM on the selected gfx1201 device. This is capacity context, not a peak-allocation guarantee.
- The full conversion must be separately approved and run as one pass under an appropriate lease window. No serving, A/B or KLD was performed.
