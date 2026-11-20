# LTX-2.5 TP3 card-free reference (LT3 foundation)

Owner: future native LTX architecture / upstream migration track, captured in docs/design/ltx25-iq8r-tp3.md. This is a **lab-only**, self-contained C++20 reference, **not** a production engine component, runtime plugin or model implementation.

## Question

Before changing upstream Radiance's LTX TP1 refusal, can three uneven ranks preserve all 32 video/audio attention heads, reconstruct fused QKV bank slices, reject unaligned IQ8R K cuts, normalize Q/K with one global RMS denominator, and enforce a disjoint four-device role/memory sketch? Can we bound the shape of the Gemma-to-connector transfer correctly?

## Inputs, outputs and requirements

Inputs are synthetic fixed-seed values and the *source-declared* LTX-2.5 geometries (32 x 128 video heads, 32 x 64 audio heads; Gemma 3840 channels and 48 layers; maximum two 1024-row prompts), not checkpoint weights. The reference includes IQ8R alignment constraints (128-K FWHT, 256-K scales) but **does not parse or execute real IQ8R files**.

The separate ltx_tp3_protocol.h and ltx_tp3_protocol_test.cpp also test identical collective schedules for conditional, STG and isolated guidance, leader-only non-DiT writes, bias-once reductions, and an in-memory conditioning mailbox that refuses stale epochs, mismatched shapes, overwrites and consumed/cancelled requests. The mailbox is intentionally not a GPU transfer or production wire format.

The ltx_tp3_numerical_test.cpp independently compares multi-row dense self-attention and cross-attention with three uneven rank slices across six cases: fused projection banks, global Q/K RMS, per-head softmax and 2*sigmoid gate, and bias-once output reduction. Uses f64 tiny data and omits RoPE, IQ8R encoding and BF16 rounding; it does not establish real GPU parity. Shape/transfer mock arithmetic now refuses integer overflow.

Outputs are process exit status and the CTest result. No generated evidence receipts or recordings are claimed. Requires a C++20 compiler and CMake 3.21+; no ROCm/GPU/model/network access. The reusable header deliberately has no imported production entrypoint, and this lab must not appear in an engine release.

From the repository root, for a direct native CTest run:

    cmake -S tools/lab/radiance/ltx-tp3 -B "$BC_WORK_ROOT/ltx-tp3-reference" -DBUILD_TESTING=ON
    cmake --build "$BC_WORK_ROOT/ltx-tp3-reference"
    cmake --build "$BC_WORK_ROOT/ltx-tp3-reference" --target test

For the standard tooling test suite (Python test harness only, never engine runtime):

    PYTHONPATH=tools python3 -m pytest tools/tests/lab/test_ltx_tp3_reference.py -q

The model-aware numeric tool also audits all 48 DiT blocks: 960 required IQ8R projection entries share expected shapes; audio 10/10/12 would break scale-group K cuts for four output projections per block (192 invalid). Audio FFN-down remains aligned because its K width is four times the audio head-inner width. This is an admission oracle only, not kernel or TP3 execution coverage.

## Mutation/safety

Writes only the explicitly selected CMake build directory or pytest temporary directory. Does not touch GPUs, allocate model weights, mutate engine source, weaken the TP1 guard, or hold a hardware lease. No background processes. A mock 16-GiB gfx1030 memory budget is not a residency proof: the real Gemma workspace, allocator and connector placement must be measured separately.

## Disposition

Experimental lab proof for LT3-03/04/05/07/11/12. Do not promote to shared tooling or production until its ownership/ABI is designed against the RUM-migrated upstream LTX builder and a real device implementation passes numerical parity. Delete only through the tooling disposition/retirement process after an accepted replacement; preserve this reference while cited by the design.


## Optional real-model validation (separate test tooling)

The corresponding Python test tooling can inspect the actual local IQ8R container with bounded disk reads, without running Radiance or modifying production: `tools/tests/lab/test_ltx_iq8r_planes.py` checks every declared IQ8R plane extent and `tools/tests/lab/test_ltx_iq8r_numeric.py` performs sampled row-level BF16/U8/I8/FWHT128 dense-versus-TP3 arithmetic. Both accept an explicit model-file positional argument and have CPU-only synthetic pytest fixtures. Evidence and limitations: `docs/design/ltx25-iq8r-model-plane-evidence.md`. The former's pure header/plane validation and latter's f64 oracle do not qualify the native HIP kernel. A 10/10/12 audio **K** split is forbidden at group 256 even though it owns whole attention heads.

## Per-component harnesses and promotion gates

The lab now owns **nine independently selectable native CTests**: `ltx_tp3_reference_test`, `ltx_tp3_protocol_test`, `ltx_tp3_numerical_test`, `ltx_tp3_linear_test`, `ltx_tp3_rms_test`, `ltx_tp3_handoff_test`, `ltx_tp3_guidance_test`, `ltx_tp3_output_test` and `ltx_tp3_micro_pipeline_test`. The two additional real-container Python oracles inspect model planes, sampled quantized rows, all 48 DiT block layouts and both eight-layer connector stacks. For the exact matrix of component pass/fail gates, device promotion steps and end-to-end acceptance see `docs/design/ltx25-tp3-component-test-ladder.md`. All native tests are synthetic and make no production engine calls; optional real-file tests are read-only and CPU-only.
