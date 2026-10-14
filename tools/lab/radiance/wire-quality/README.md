# Radiance rotated-wire quality: offline numerical proxy (RWQ01)

**Question:** Does the proposed 8-bit Hadamard wire reduce local quantisation error versus the pinned six-bit method on controlled synthetic inputs, and do our pack/unpack and two-rank symmetry invariants hold?

**Owner / disposition:** `RWQ01`; exploratory laboratory code, not the engine runtime, and not production quality evidence. Keep under `tools/lab/radiance/wire-quality/` until reviewed. A later GPU kernel checker will own device parity.

**Inputs:** built-in seeded (`7107`) deterministic patterns: gaussian, heavy-tail, sparse, mixed-group-scale. Two mock ranks, eight 64-value groups per rank. No external model, network, ROCm, or GPU. Standard-library Python 3 only.

**Command:** from the repo root, `python3 tools/lab/radiance/wire-quality/codec_mock.py --json`. Tests: `python3 -m unittest discover -s tools/tests/radiance -p test_wire_quality_mock.py -v`.

**Outputs:** stdout JSON (`schema=1`, explicit `gpu_tested=false` and `model_kl_tested=false`), numerical error versus a simulated BF16 exact sum, logical per-rank payload sizes. A snapshot from the initial local run is `docs/evidence/RWQ01/mock-results.json` (not hardware evidence). No files are written by the tool; capture command outputs in the run-specific artifact directory if desired.

**Mocked arithmetic:** 64-value orthonormal WHT in fp32, BF16 input, BF16 RNE block scale `amax / (2^(bits-1)-1)`, signed symmetric fixed-width integer quantisation with RNE and saturation, two-rank sum in the rotated domain, inverse WHT, BF16 output. Both ranks send an identically quantised local contribution and fold in the opposite rank's packed contribution. Bits 4, 6, 8 share this numerical path.

**Critical limitations:** the packed stream uses logical contiguous little-endian codes, *not* libr4d's per-wave/chunk-interleaved `uint4` layout. Python rounds intermediate fp32 operations but is not a bit-exact HIP oracle (different math-instruction ordering and compiler transformations). This does not exercise HIP BF16 conversion edge cases, shared-memory addressing, protocol flags, graph replay, row-stride rules, numerical tails in actual activations, or model/logits KL divergence. Synthetic relative L2 improvements do not imply the same model-quality improvements.

**Safety/mutation:** stdout-only; deterministic CPU work; no downloads, credentials, GPU allocations, filesystem writes, model conversions, or source modifications. This tool is intended for early falsification only. GPU-checker and model-KL gates in RWQ02/RWQ05 are mandatory.

**Reference source:** `Quackster/radiance` at `89cee7ce414486c717aa2085c99e1f522e4c0ea6`, `libr4d/r4d_ar_oneshot_2rank_wht6.h` and fused paths in `libr4d/r4d_ar_oneshot_2rank_wht6.hip`; rewrite path `engines/radiance/components/libr3/native/r4d_ar_entry.hip.rw`.