# HI84 D=3 synthetic-input corpus: local recovery (13 November 2026)

This is a compact **historical input/reconstruction record**, not a hardware-success receipt or a new META/RCCL qualification. It recovers the inputs found only in the old BigCherry bisect clone's ignored `_scratch_hi84_d3_corpus` directory. The actual D=3/D=4 qualification and outstanding gates are owned by [HI84](../../planning/completed/patching-hip-collectives/HI84.md) and the [HI18 generator/evaluator](../../../tools/bigcherry/tuning/reduction.py).

## Original inputs

- 18 synthetic cases: six patterns (`ordinary_signed`, `all_positive`, `alternating_signs`, `cancellation`, `mixed_magnitude`, `sparse_zeros`) times seeds 1, 2, 3.
- Each case has an original `case.json` (schema v2) and three little-endian F32 input arrays (`rank-0.f32` through `rank-2.f32`), each with 8,192 elements. Generator identity is `hi18-f32-v1`.
- The frozen case metadata declares `slice_shape=[4096,2,1,1]`, `topology_key=n3:peer100010001`, `peer_access=partial`, and the per-rank input digests. These are *historical case inputs*, not a measurement of present hardware.
- `case-manifests.tar.gz` preserves **all 18 original case.json files byte-for-byte** (including their original line endings). `source-files.sha256` records the SHA-256 of all 72 original files, including the 54 rank buffers.
- Original file timestamps were 2 October 2026; the recovery and checks were performed on 13 November 2026. The precise execution context that created the scratch directory is unknown.

## Independent card-free regeneration check

The pinned source `tools/bigcherry/tuning/reduction.py`, at BigCherry revision `ad6e8e3201112d33df085ef17aec1e81f7b4e3ab` (source-file SHA-256 `525c074b06719e96456f847a05adea0ed8da112b4eca49595f5cd8772999a2fa`), exposes `generate_case(seed, pattern, element_count, device_count)` and `to_f32_bytes(values)`.

The archive recovery reproduced **54 of 54 rank files byte-for-byte** using those functions with each manifest's `seed`, `pattern`, `element_count`, and `device_count`. All 54 hashes also agreed with each case's `input_digests` values, with zero mismatches. The current generator's byte content was verified identical to the code used for that check.

To reproduce later, extract `case-manifests.tar.gz`, parse each `case.json`, call the tracked generator with its four recorded parameters, serialize each rank with `to_f32_bytes`, and compare against the corresponding `input_digests` and `source-files.sha256` entries. This reconstructs the original rank inputs exactly; there is no need to retain separate binary F32 copies in Git.

## Limits and retention

This proves **synthetic input identity only**. It does not show that any GPU probe completed, that META or RCCL was correct, or that the collected topology is current. No hardware or GPU lease was used during recovery.

The source checkouts and GPU run artifacts are separately managed under the project's work root. This compact source-plus-digest record closes the unique raw-input dependency before the obsolete bisect clone is considered for cleanup; it does not authorize unrelated run-data removal.
