# RWQ01 offline mock evidence (2026-10-15)

Input: `tools/lab/radiance/wire-quality/codec_mock.py` defaults; Python 3 stdlib; seed 7107, eight groups of 64 values for each of two ranks per pattern. Command `python3 tools/lab/radiance/wire-quality/codec_mock.py --json`; recorded structured output: [mock-results.json](mock-results.json). Verification: `python3 -m unittest discover -s tools/tests/radiance -p test_wire_quality_mock.py -v`, **7 tests passed** locally.

| Pattern | WHT6 rel-L2 | WHT8 rel-L2 | Interpretation |
|---|---:|---:|---|
| Gaussian | 0.022557 | 0.005892 | WHT8 reduces numerical error on this synthetic sample |
| Heavy-tail | 0.012318 | 0.003187 | WHT8 reduces numerical error on this synthetic sample |

Payload per mock rank for 512 elements: exact BF16 = 1,024 bytes, WHT6 = 400 bytes, WHT8 = 528 bytes (scale included). Do not equate byte-size projections with effective PCIe throughput or end-to-end prefill benefit.

**Proof scope:** local proxy correctness and test invariants only. **Not measured:** engine kernels, compiler/device parity, model KLD, logits agreement, top-1, throughput, GPU stream sequencing or actual memory layout. In particular, per-op sensitivity of Flash-Next is still unknown and all production switches remain unchanged. No production configuration may be promoted on this evidence.