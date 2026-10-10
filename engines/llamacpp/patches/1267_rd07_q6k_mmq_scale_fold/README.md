# 1267 RD07 Q6_K MMQ scale fold

**State:** `untested`, default-off. **Authoritative owner:** PRBE110; PRBE04 owns downstream PEF01/HI71 safety gates. This is the RD07-only extraction from rejected 1203, not a new Q6_K MMQ implementation.

The three ordered `mmq-vec-dot.cuh` edits hoist the Q6_K row base scale, fold each k01 subscale before the j0 loop, and replace the AMD MFMA/WMMA accumulation. DP4A remains unchanged. `patch.toml` requires `0300_mmq_forced_j`; the Q6_K host-dispatch trace marker does **not** prove the changed MMA kernel ran. The existing J_MAX override is a qualification knob, not a new production tuner.

**Provenance:** [stew675 source commit](https://github.com/stew675/llama.cpp/commit/1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff). `1006_rdna4_mmq_q6k_codegen_fix` changes the same accumulation's explicit F32 cast and is now validated/production-selected; 1267 accepts either sum-line anchor, but its historical b11126 +2.4675% gfx1201 pp512 receipt did **not** contain 1006 in the baseline. Neither this receipt nor rejected 1203's results qualify the incremental production gain.

**2026-10-10 correctness prerequisite:** pinned b11474 has the pre-[#29953](https://github.com/ggml-org/llama.cpp/pull/29953) MMQ staging-padding calculation; source-derived RDNA4 Q6_K `ne11=17` may select J=32 after reserving only J=16 padding. [#30168](https://github.com/ggml-org/llama.cpp/pull/30168) also changes host precision-config propagation. These are upstream baseline gates, not proof of a 1267 runtime failure. Do not time or promote 1267 until the existing upstream-fix/pin owner proves safe current-pin allocation and composed 0300+1006+1267 behavior.

Use `validation/producer.py` and the existing mechanics tests; require actual MMA-kernel activation, PEF01/HI71 safety, full-vocab/greedy/MTP parity, graph replay and multi-request checks. Profile the incremental Q6_K critical path first; retire without a hardware campaign if theoretical end-to-end gain <3%. Only surviving gfx1201 candidates get four independent >=10-pair ABBA sessions with CI95-low >=3% end-to-end and <=1% controls. No gfx1100/gfx1030 promotion claim.
