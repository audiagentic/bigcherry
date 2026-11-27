---
id: XDP11
order: 10
plan: run-xtx27-prefill
state: in_progress
created-at: '2026-11-16T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# IQ6R-V4 calibrated RAD2 conversion and TP3 qualification

## Scope and lineage

The owner requested a real Qwen3.8-27B IQ6R-V4 RAD model and assessment, **after** the precision policy merged to main (PR #996). This branch was created from the merged main revision. XDP03 owns model quantisation/per-layer candidate decisions; XDP10 owns optional variant-loader admission. This item owns the **single immutable IQ6R-V4 artifact and its calibration, converter, quality, performance and acceptance evidence**. VTP06/VTP11 supply the original-BF16 calibration prerequisites; VTP09 supplies gfx1100 IQR kernel evidence; VTP14 owns the TP3 runtime contract.

Do not modify a pinned engine source, production model, running server or other worktree. Do not run on GPUs or launch device-aware `rad-convert --plan-only` without an exclusive GPU lease. Retain every unqualified result as an experiment, not a production default.

## Candidate and baseline

- Primary: `data/recipes/q38-27b-iq6r-v4.recipe` on Radiance pinned v1.17.0, `quantization_config.quant_method=w8a8` and `quantization_config.fmt=fwht128`, `qwen35_w8a8` activation ABI.
- Source: the original, independently fingerprinted 18-shard BF16 27B checkpoint (VTP06 receipt); never re-quantize the existing MXFP4/W8A8/GGUF files.
- The V4 recipe uses GPTQ Hessians, heterogeneous 6/7-bit IQR trellis weights, and `tools/vhead.py` vocabulary classes; preserve unmatched BF16/FP32 exceptions from the source.
- Controls: the SHA-verified existing W8A8+DFlash2 RAD, MXFP4 RAD and BF16 reference. Do not compare DFlash2-on results to IQ6R draft-off results.
- Secondary variants IQ6R-S-V4 and IQ6R-L-V4 are **not** prerequisites to qualifying the primary model.

## Gates

1. **Offline preflight:** inventory BF16 config/tokenizer/18 shards, pinned recipe, vhead class file and expected 64 layers. Record source checksums from VTP06, current source revision and an explicit absence/presence receipt for representative RADGRAM2 Hessians. Do not mistake a first-layer pilot for 64-layer calibration.
2. **Calibration plumbing:** review VTP11 GDN ownership and its layer-0 TP1/TP2 smoke. Under a separately acquired exclusive GPU lease, revalidate correct, numerical BF16 GDN and bounded Hessian generation on the current pin. Abort on a device/host kernel mismatch or incomplete corpus.
3. **Representative calibration:** predeclare corpus, tokenizer, seed, layer coverage, source hashes and storage budget. Collect all 64 layer Hessians in bounded chunks; validate format, shape and full coverage, and record their hashes. Uncalibrated quantiser fallback is a hard failure.
4. **Converter gate:** derive vocabulary classes using pinned `tools/vhead.py`; run `rad-convert --plan-only -v` with the pinned `libr3/libr4d` and recipe only under lease; inspect **every** tensor encoding, selected kernel/layout, no missing projections and no unplanned BF16/F16 substitutions. A preflight failure blocks actual conversion. Convert once to a new immutable experimental artifact, record full SHA-256, `rad-info -v`, per-tensor quant errors, converter binary/build, recipe hash and source/calib/vhead identities.
5. **Math and model quality:** qualify numerical IQR kernel rows (gfx1100 and gfx1201) at representative M and full layer shapes. Compare IQ6R against BF16, W8A8 and MXFP4 with held-out KL/top-1/PPL and reasoning/coding/long-context prompts. Test draft-off first; DFlash2 is a separate compatibility/quality gate.
6. **TP3 / memory / speed:** verify XTX+XTX+R9700 identical encoding/activation semantics, startup and graph selection, no silent kernel fallback, physical VRAM weight/KV/GDN/scratch, 1/4/6/8 streams, prefill and decode at short and long contexts with paired repeated runs. Fail closed for admission/collective and quality defects.
7. **Decision:** initial proposed gates (not a guarantee): mean held-out KL <= 0.001, top-1 >= 98.8%, no more than 10% end-to-end prefill/decode regression against same-configuration W8A8, and a meaningful physical KV-capacity increase. Only propose targeted BF16/F16 promotion after an otherwise qualified IQ6R baseline and measured per-projection sensitivity. Record the exact tradeoff and keep production W8A8 unchanged pending owner approval.

## Initial findings (2026-11-16, card-free)

- Pinned `engines/radiance/recipes.toml` is v1.17.0; the pinned engine includes `q38-27b-iq6r-v4.recipe`, `docs/IQR.md`, `r4d_iqr_layout.cpp`, and `qwen35_w8a8`.
- Local BF16 source has 18 model safetensors shards, `config.json` states 64 transformer layers; the DFlash2 checkpoint and current W8A8+DFlash2 model also exist.
- The inspected Radiance model inventory contains no ready IQ6R `.rad`; the recorded VTP11 smoke covers layer 0, not a representative complete calibration.
- There is no claim of any GPU preflight, new converted artifact, or model-quality result from this kickoff. See `docs/evidence/2026-11-16-iq6r-v4-kickoff/README.md`.

## Progress and stop rules

- [x] Prior precision-policy plans merged to main (#996)
- [x] Independent experiment worktree created from merged main
- [x] Card-free source/recipe inventory started
- [x] Full BF16 18-shard SHA-256 recheck versus VTP06 (18/18 match)
- [x] All 257 upstream Hessians SHA-256-verified, with all 64 layer sites and RADGRAM2 headers validated
- [ ] Exclusive-lease BF16 calibration qualification
- [x] Representative upstream 10,259,331-token BF16 Hessians imported (not locally recollected)
- [x] BF16-derived vocabulary-head staging checkpoint produced and SHA-256 pinned
- [x] Both gfx1100/libr3 and gfx1201/libr4d candidate converter layout preflights pass
- [ ] Immutable IQ6R-V4 model artifact
- [ ] BF16-relative quality acceptance
- [ ] TP3 admission, VRAM, throughput and long-context acceptance

Never claim completion from plan-only, a kernel microbenchmark, a smoke corpus or upstream published KL numbers.

## 2026-11-16 upstream Hessian option and qualification

The public `StillDeadcode/qwen3.8-27b-quant-challenge` dataset includes 257 BF16 Hessians for the `calib` split (4,789 documents, 10,259,331 tokens; 15 categories), plus a disjoint `public` scoring split (596 documents, 1,694,621 tokens). Pin dataset revision `2dcf0061cb683152742b904e2965ce7cd92b2da6`, not moving `main`. The Hessians and tools are **CC BY 4.0**; the corpus has separate mixed-source licences. Attribute the publisher in retained evidence.

Using these Hessians avoids duplicate GPU calibration collection but does not waive provenance or independent quality qualification. Require all 257 LFS SHA-256 checks, exact 64-layer input sites, consistent shapes and 10,259,331 rows, original BF16 shard hashes, and disjoint held-out prompts. Missing/altered inputs stop conversion. Keep the Q6 result experimental until numeric kernel and TP3 validation.

A historical converter build failed to join IQR weight planes. A more complete installed build passed both gfx1100 `libr3,libref` and gfx1201 `libr4d,libref` layout preflights (1,032 weights; 293 quantized in total (292 GPTQ trellis plus one RTN embedding), 739 retained; ~21.02 GiB predicted). Pilot calibration paths were supplied for plan-only name resolution; no model data was written. Current managed-pin identity and complete calibration remain separate gates.

## Managed-release conversion checkpoint (2026-11-16)

Current CI-managed release bc-radiance-1170.26.0 was selected using its install manifest; the runner verifies the recorded runtime SHA-256. All tracked executable paths are provided via BC_WORK_ROOT, BC_MODEL_ROOT, BC_RADIANCE_SOURCE and BC_RELEASE_MANIFEST rather than embedded host locations. The failed partial RAD is removed; its logs remain as diagnostic evidence. All 257 packed Hessian sites were expanded to 289 RADGRAM2 input stems. Offline tests and current-release preflights passed; the actual GPTQ conversion is under an exclusive lease. The output is not qualified until exit-zero, final hash, model inspection, BF16-relative quality and TP3 serving checks pass. No production promotion is authorized by preflight alone.

## 2026-11-16 crash triage gate

The full managed-build conversion of calibrated IQ6R-V4 stopped with SIGSEGV (-11) after 1,329.55 seconds, with no qualified RAD artifact. The last converter message described successful GPTQ Hessian loading for the layer-23 attention-value projection, but that does not locate the fault. Input checksums, all 64 layers, 289 RADGRAM2 stems, independent row/column sample comparisons (12,593 cells), and the managed release identity pass. No missing-calibration fallback appeared. The incomplete output was deleted and failure receipts retained. GPU memory corruption or host-RAM failure is unproven. Full-quality and TP3 benchmark gates must remain blocked.

The focused diagnostic isolates the single IQ6R-V4 layer-23 attention-value projection while reusing an existing read-only W8A8 model for all other weights. Use an exclusive lease and record a genuine SIGSEGV backtrace, comparing single-GPU, dual-GPU and CPU-only outcomes if needed. A manually interrupted debugger snapshot shows a host worker in GPTQ factorization, not a crashing instruction. See the dedicated CRASH-DIAGNOSIS evidence note. Keep any focused RAD unqualified and never overwrite production model data.

## 2026-11-16 confirmed crash cause and remediation

The matched single-projection reproduction located SIGSEGV at the host HIP hipLaunchKernel call from libquant gptq_tcq_gpu/run_rows. Managed libquant contains only gfx1201 offload code, while both failed full-model launches selected gfx1100 XTX cards. The same calibrated IQ6R-V4 projection crashed on one XTX and converted successfully on the R9700 with the same managed release. The diagnostic RAD2 was validated by rad-info and then discarded; it was a mixed-W8A8 probe, NOT a production IQ6R model. See the dedicated CRASH-DIAGNOSIS evidence note.

The experimental runner now checks the libquant offload bundle before launch and selects the R9700 (HIP_VISIBLE_DEVICES=2, local --gpus 0), with libr4d/portable reference layout. Keep gfx1100/libr3 and gfx1201/libr4d independent plan-only compatibility gates. Serving architecture and quantizer device architecture are separate contracts. Managed libquant also lacks OpenBLAS, which causes slow host Hessian factorization but was not the immediate SIGSEGV cause. The complete IQ6R model, BF16-relative quality and TP3 acceptance remain pending.

## Three-card crash confirmation and HIP source (2026-11-16)

The same isolated layer-23 calibrated IQ6R projection crashes at hipLaunchKernel on **both** gfx1100 RX 7900 XTX cards (physical GPUs 0 and 1), while it converts successfully with the same release and input on the gfx1201 R9700 (GPU 2). GDB traces, crash receipts, the ROCm 7.2.4 source investigation, and the independent three-card comparison are retained in CRASH-DIAGNOSIS.md. The HIP runtime source contains a host-function-handle fallback when the requested GPU image cannot be resolved. The libquant binary has only gfx1201 offload code. Do not treat this as evidence of faulty XTX memory or a broken serving kernel; keep the full-model conversion pinned to the R9700 with an offload-image admission guard. The full artifact, BF16-relative quality and TP3 throughput assessment remain outstanding.

## ROCm SDK wrong-architecture regression gate (2026-11-16)

The BigCherry vendor directory includes complete ROCm 10.1.0rc2. A controlled six-case kernel A/B built gfx1201-only binaries against installed ROCm 7.2.4 and vendored ROCm 10.1.0rc2. With correct device selection, both gfx1100 XTX cards segfault on ROCm 7.2.4 but return clean HIP error 200 on vendor 10.1.0rc2; the gfx1201 R9700 executes successfully on both SDKs. This verifies safer error handling, not production inference compatibility or gfx1100 quantizer capability. Retain the offload image guard, keep production ROCm 7.2.4 unchanged, and continue full IQ6R conversion on the R9700. See ROCM-VENDOR-ABI.md in the evidence folder for test source, complete results and reproduction instructions.

## Full conversion and test programme checkpoint (2026-11-16)

A fresh, bounded, exclusive-lease full-model conversion was started with the managed bc-radiance-1170.26.0 converter on physical GPU 2 (gfx1201 R9700), using local HIP ordinal 0 and the verified 64-layer calibration. The prior gfx1100 SIGSEGV receipt was preserved separately. This is an **active experiment**, not an assertion that a valid RAD exists; the final artifact, complete SHA and model inspection remain outstanding.

The independently owned Q0-Q9 test matrix and explicit stop/go conditions are in the evidence directory at QUALIFICATION-TEST-PLAN.md. It separates conversion integrity, gfx1100/gfx1201 kernel numerics, isolated service, BF16-aligned held-out quality, exact TP3 admission, physical memory/capacity, paired ABBA throughput and optional draft-on validation. The card-free iq6r_v4_audit_artifact.py helper checks the immutable artifact/recipe/encoding contract after conversion exits successfully. Thresholds are proposed until the repeat-BF16 floor is frozen.

## 2026-11-16 checkpointed conversion adaptation

A full-recipe R9700 run converted 32 of 292 calibrated GPTQ tensors without HIP or calibration errors before it was deliberately stopped to avoid the hard four-hour wrapper timeout. The incomplete model was discarded; diagnostics were preserved. The new staged conversion uses exact --reuse matching of encoding, quantizer options and tensor shapes to carry completed tensors forward between sealed RAD2 checkpoints.

The 16-stage strategy converts four vocabulary-head classes first, while maintaining the pinned rotated-INT8 W8A8 trunk fallback. Stages 2-16 add up to 20 explicit GPTQ trunk rules each, which precede wildcard fallback rules. Stage 16 is the exact original V4 recipe. Two initial plan-only checks correctly rejected a BF16 vocabulary head and a BF16 trunk, respectively. After correction, stage-1 gfx1201 plan-only passed, estimating 25.57 GiB, and the R9700 stage-1 converter was launched under a fresh exclusive lease. Until full RAD2 checkpoints and stage-2 reuse are verified, this remains an experimental, unqualified route. See QUALIFICATION-TEST-PLAN.md and scratch receipts.

## RDNA3 card-free compatibility preflight (2026-11-16)

The exact managed-release gfx1100 libr3.so passes a repeatable offline IQR contract check: the pinned rewrite set applies cleanly, gfx1100 HIP images are present, twenty required IQR hooks are non-stubbed exports, and seven upstream IQR rows are present with no IQR GEMM unit exclusion. See docs/evidence/2026-11-16-iq6r-v4-kickoff/RDNA3-IQR-PREFLIGHT.md and its SHA-256 pinned receipt. These are necessary compatibility checks only; the full IQ6R RAD remains incomplete, and per-XTX device numerics, live row selection, model loading and mixed TP3 quality are still outstanding. No GPU tests were run during the conversion lease.
