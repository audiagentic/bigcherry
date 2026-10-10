# 1267_rd07_q6k_mmq_scale_fold

**Status:** untested / default-off
**Plan item:** PRBE110 (technical owner); PRBE04 (PEF01/HI71 safety)

Hoists and folds Q6_K MMA scales; no DP4A change. Requires 0300; composes with validated 1006's Q6_K explicit F32 cast. Host `BIGCHERRY_PATCH_HIT` reports Q6_K dispatch only, **not actual modified MMA-kernel execution**.

Historical b11126 gfx1201 +2.4675% pp512 (CI95 [2.2776%,2.6508%]) is measured without production-selected 1006 in its stored baseline. No current-pin incremental gain exists. Upstream llama.cpp [#29953](https://github.com/ggml-org/llama.cpp/pull/29953) fixes MMQ padding/launch mismatch; [#30168](https://github.com/ggml-org/llama.cpp/pull/30168) aligns host precision config. b11474 predates both. **Block promotion pending baseline safety and actual-kernel route proof**; use the existing producer, not a second qualification path. PRBE110 contains terminal gates.
