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
- [ ] Full source/calibration manifest and independent file-hash checks (recipe/vhead hashes done; BF16 18-payload recheck pending)
- [ ] Exclusive-lease BF16 calibration qualification
- [ ] Representative 64-layer Hessians
- [x] BF16-derived vocabulary-head staging checkpoint produced and SHA-256 pinned
- [ ] Converter plan and immutable IQ6R-V4 artifact
- [ ] BF16-relative quality acceptance
- [ ] TP3 admission, VRAM, throughput and long-context acceptance

Never claim completion from plan-only, a kernel microbenchmark, a smoke corpus or upstream published KL numbers.
