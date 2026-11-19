# LTX-2.5 TP3 card-free reference (LT3 foundation)

Owner: future native LTX architecture / upstream migration track, captured in docs/design/ltx25-iq8r-tp3.md. This is a **lab-only**, self-contained C++20 reference, **not** a production engine component, runtime plugin or model implementation.

## Question

Before changing upstream Radiance's LTX TP1 refusal, can three uneven ranks preserve all 32 video/audio attention heads, reconstruct fused QKV bank slices, reject unaligned IQ8R K cuts, normalize Q/K with one global RMS denominator, and enforce a disjoint four-device role/memory sketch? Can we bound the shape of the Gemma-to-connector transfer correctly?

## Inputs, outputs and requirements

Inputs are synthetic fixed-seed values and the *source-declared* LTX-2.5 geometries (32 x 128 video heads, 32 x 64 audio heads; Gemma 3840 channels and 48 layers; maximum two 1024-row prompts), not checkpoint weights. The reference includes IQ8R alignment constraints (128-K FWHT, 256-K scales) but **does not parse or execute real IQ8R files**.

The separate ltx_tp3_protocol.h and ltx_tp3_protocol_test.cpp also test identical collective schedules for conditional, STG and isolated guidance, leader-only non-DiT writes, bias-once reductions, and an in-memory conditioning mailbox that refuses stale epochs, mismatched shapes, overwrites and consumed/cancelled requests. The mailbox is intentionally not a GPU transfer or production wire format.

Outputs are process exit status and the CTest result. No generated evidence receipts or recordings are claimed. Requires a C++20 compiler and CMake 3.21+; no ROCm/GPU/model/network access. The reusable header deliberately has no imported production entrypoint, and this lab must not appear in an engine release.

From the repository root, for a direct native CTest run:

    cmake -S tools/lab/radiance/ltx-tp3 -B "$BC_WORK_ROOT/ltx-tp3-reference" -DBUILD_TESTING=ON
    cmake --build "$BC_WORK_ROOT/ltx-tp3-reference"
    cmake --build "$BC_WORK_ROOT/ltx-tp3-reference" --target test

For the standard tooling test suite (Python test harness only, never engine runtime):

    PYTHONPATH=tools python3 -m pytest tools/tests/lab/test_ltx_tp3_reference.py -q

## Mutation/safety

Writes only the explicitly selected CMake build directory or pytest temporary directory. Does not touch GPUs, allocate model weights, mutate engine source, weaken the TP1 guard, or hold a hardware lease. No background processes. A mock 16-GiB gfx1030 memory budget is not a residency proof: the real Gemma workspace, allocator and connector placement must be measured separately.

## Disposition

Experimental lab proof for LT3-03/04/05/07/11/12. Do not promote to shared tooling or production until its ownership/ABI is designed against the RUM-migrated upstream LTX builder and a real device implementation passes numerical parity. Delete only through the tooling disposition/retirement process after an accepted replacement; preserve this reference while cited by the design.
