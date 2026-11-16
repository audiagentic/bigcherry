# IQ6R-V4 kickoff — card-free evidence (2026-11-16)

Status: **offline preflight passed; representative calibration and RAD conversion NOT done**.

## Provenance and work

- The precision-retention decision in XDP03 and XDP10 was merged to main as PR #996 (merge commit `6966213e3`) before creating the `feat/iq6r-v4-qualification` experiment worktree.
- Local recipe and engine source examined are the v1.17.0 materialized Radiance source, referenced read-only; the worktree remains the authoritative place for BigCherry-owned changes. `engines/radiance/recipes.toml` pins `v1.17.0`. Source/materialization identity must be recorded by the device-aware converter gate, not inferred from this card-free survey.
- Original BF16 Qwen3.8-27B: 64 layers, 18 indexed safetensors shards, tokenizer and chat template present at `${BC_MODEL_ROOT}/qwen3.8-27b/Qwen3.8-27B`. The original BF16 full-shard SHA-256 receipt is documented in VTP06; this run only checked index/file presence, **not** hashes of all 18 payload files.
- Candidate: `data/recipes/q38-27b-iq6r-v4.recipe` SHA-256 `0c41614302d448ba61c39f31eb6da1ee4970fcd4fa1a6011ec479d51b64e597f`, 293 unique expected quantizer rules, 146 u6, 145 u7, 1 u5, 1 u4 (width counts refer to logical weights, not bytes). Missing/unexpected rule names: zero. All trunk rules use GPTQ with FWHT128 and `$CALIB`.
- V4 vocabulary-head classes: `data/vhead/q38-27b-vhead-5.0.cls` SHA-256 `cdc09e6da341e5cbd924a54d6a0b7c6d649f4ca4d39797ebd6707fc87d9c773b`.
- Previous VTP06 two-document pilot has 25 safetensors files spanning layers **0,1,2,3,62,63** only. Not a complete or representative calibration, even if every present file is otherwise valid.
- The local Radiance inventory has no IQ6R `.rad` artifact. No assumption of IQ6R model-serving availability or quality is made.

## Repeatable offline check

Tool: `tools/lab/radiance/iq6r_v4_offline.py`. It checks 64-layer geometry, the 18-indexed-shard inventory, first-pass Q6 recipe name/encoding coverage, vocabulary class presence, and observes layer numbers in existing calibration safetensors. It does **not** validate Hessian shape, numerical fidelity, source full hashes, RAD layout hooks, GPU selection or actual model quality.

Run the script from a pinned-source context with explicit `--source`, `--recipe`, `--vhead`, `--calibration`, `--output` paths. Its output is stored in the task-specific work root: `${BC_WORK_ROOT}/iq6r-v4-qualification/offline-inventory.json`. Result: `offline_inventory_passed`, with `observed_all_64_layer_indices=false`. Unit tests: `python3 -m unittest discover -s tools/tests/lab -p test_iq6r_v4_offline.py -v`: 5 passing, zero failures.

## Next blockers / stop gates

1. Revalidate pinned original-BF16 calibration numerical correctness and source identity with the managed VTP11 GDN fix. Existing historical layer-0 smoke alone does not establish v1.17.0 full-model calibration.
2. Predeclare a representative corpus and obtain full 64-layer RADGRAM2 Hessians, with full per-file payload and calibration digests. Reject absent, partial or uncalibrated fallback.
3. Produce and hash the V4 vocabulary-head staging checkpoint from BF16 using `tools/vhead.py`; it may require the isolated Python NumPy dependency. Never edit the original source checkpoint.
4. Under exclusive GPU lease, perform the actual `rad-convert --plan-only` with device-accurate `libr3/libr4d`, then convert and compare BF16-relative quality. A converter plan cannot substitute for a kernel acceptance gate.
5. Full TP3 service benchmarks and production decisions remain blocked until those steps pass.

No GPU measurement, full conversion, production change, or performance claim has occurred in this kickoff.

## Vocabulary-head staging completed

The pinned `tools/vhead.py` completed CPU-only preparation using isolated NumPy 2.5.5 (in the task work root, not system Python). Its output `${BC_WORK_ROOT}/iq6r-v4-qualification/vhead-5.0/vhead.safetensors` is approximately 2.4 GiB; SHA-256 `1dd39da8c0a5205fd1f53e2052807bafe0a1c2fb291a0a88908df427423e5f90`. The tool reported classes k4:130560, k5:33280, k6:38400, k7:46080 (mean 5.000 bits). The staging checkpoint has a rewritten tensor index and links to unchanged original BF16 shards. This **does not** mean a quantized model exists: IQ6R trellis codes and Hessian-derived weights have not been generated yet.

## Full BF16 source identity and public calibration (2026-11-16)

All 18 original BF16 weight shards were reread end to end. Each SHA-256 matches the prior VTP06 full-payload audit. Full receipt (not committed due to local-path metadata): `${BC_WORK_ROOT}/iq6r-v4-qualification/source-sha256-recheck.json` (18/18 matching; 87.69 s). The output-head staging remains SHA-256 verified.

Radiance dataset: `https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge`, immutable revision `2dcf0061cb683152742b904e2965ce7cd92b2da6`. The dataset publisher releases **Hessians under CC BY 4.0**; its corpus has mixed upstream source licences and is retained locally rather than copied into this repository. `calib`: 4,789 documents, 10,259,331 tokens, 66 source identities, 15 categories. `public` (disjoint): 596 documents, 1,694,621 tokens, 57 source identities, 15 categories. Exact prompt overlap is zero.

SHA-256 of pinned corpus files: calibration `ea0569b94cb8a86dc33fc7920dcdc199e91ee39f4fa4fe7283839a83ad3e8e53`; public `144243ceba1e6bde89f2a94fda39991a6e027847c797590fc1da5aa6d0ecb93e`. Manifest is preserved as `${BC_WORK_ROOT}/iq6r-v4-qualification/upstream-corpus-manifest.json`. The 257 Hessians total 46.928 GiB and are being downloaded independently against LFS SHA-256; do not treat a partial set as complete.

Two **exclusive-leased** converter `--plan-only` probes were run using an installed candidate build: gfx1100/libr3 and gfx1201/libr4d both exit zero, each 1,032 planned weights (293 quantized, 739 retained), estimated 21.02 GiB. The earlier candidate build failed on IQR weight-plane joins; full error log is preserved separately. This is not a proof of quantizer execution or TP3 admission. All experiments use scratch paths under `${BC_WORK_ROOT}`; production and vendor source trees remain unmodified.

## 2026-11-16 managed-release correction

The first GPU conversion attempt used a historical experimental build and returned SIGSEGV (-11). Its stderr also reported missing GPTQ inputs (`no calibration for ...; rounding it to nearest`), because its `CALIB` directory contained packed safetensors rather than `gram.r<rank>.<stem>.bin` RADGRAM2 matrices. It produced an incomplete 8,404,795,392-byte RAD file, now **deleted**; the original stderr and receipt remain under the isolated work root for diagnosis. The failed artifact was never loaded or promoted.

Managed release `bc-radiance-1170.26.0` is selected from its installed `install.json`, not a historical build path. Release manifest pins CI source, source slice, plan/build identities and a runtime-binary SHA-256. The experiment runner verifies the runtime against that SHA before using the adjacent converter, libr3 and libr4d plugins. The corrected calibrated input adapter expands each pinned packed F32 upper-triangle Hessian to a domain-0 RADGRAM2 full row-major matrix and aliases shared projection input sites using hardlinks. It checks name coverage before conversion and rejects any logged missing-calibration fallback. This changes neither the original BF16 checkpoint nor production deployment. Retain only the complete, independently fingerprinted experimental RAD output.

## Managed release IQ6R conversion — active checkpoint

The offline adapter completed 257/257 source sites and generated 289 RADGRAM2 calibration stems (shared projections hard-linked), approximately 94 GiB of isolated conversion inputs. All 13 offline tests passed with the isolated NumPy dependency; Ruff passed. The obsolete historical-build converter launcher was discarded. Production release manifest `bc-radiance-1170.26.0` identifies build plan `8d2c6983c713aeb3cdafab55038cd200`; the runner validates its runtime SHA-256 and uses its colocated `rad-convert`, `libr3`, `libr4d`. The GPU lease is acquired exclusively via `agent-lease` and is released when the command finishes. The experiment receipt is `${BC_WORK_ROOT}/iq6r-v4-qualification/conversion-status-managed.json`; all model writes use the `*-unverified.rad` path before a complete SHA-256 and atomic rename. The managed build passed both architecture preflights and began the actual GPTQ conversion; no missing-calibration fallback appeared in the log at this checkpoint. The model is **not qualified** until the full conversion exits zero, hashes, passes `rad-info`, quality checks, and TP3 serving tests.
