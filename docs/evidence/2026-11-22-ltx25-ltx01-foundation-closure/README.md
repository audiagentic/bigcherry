# LTX01 — source-backed IQ8R TP3 foundation qualification

**2026-11-22. Verdict:** engineering acceptance evidenced. Formal Planning MCP registration/close remains unavailable.

**Baseline:** merged main at `b522ac88b` (LTX02 C05 graph proof merged). This closure changes no production code, no recipe and no enabled LTX TP3 serving.

## Frozen source, patch and ownership

- Frozen Radiance **2.2/ABI44** source `arch/ltx2/ltx2.cpp` SHA-256: `543490eb70829693f945233d1b66e2d6d87b5e6a2817a72233fa0649218e6b53`.
- Source-managed diagnostic `rad0060_ltx2_tp3_linear_declare/patch.py` SHA-256: `964d6dea9001b052092aca3124f188dc4766f89cfbed8c9ae4ac6e5aadf87ba2`.
- Real-source test applied **22 anchored edits** into a temporary copy of the frozen source, checked all expected code anchors, and proved second-application idempotence. Its wrong-source test refuses changed upstream anchors.
- Real ABI44 native test uses the frozen patched translation unit, actual `RadFile`, `RadBuilder`, `LinearFP8::span_parts`, `rad_weight_shard_span_parts` and original reference kernel rows, with a read-only IQ8R checkpoint.
- One actual model-specific geometry authority: `C01Slice` in `src/bc_ltx_tp3_c01.h`. Its consumers are `preflight.inc`, `linear.inc`, `bias.inc`, `attn_geometry.inc` and `attn_output.inc`. **No duplicate LTX geometry authority** was found for deletion. Runtime source added/deleted in this closure: **0/0 lines**. The patch remains unselected and TP1 production behavior is not changed.

## Real checkpoint coverage

| Model audit | Result |
| --- | ---: |
| Validated IQ8R entries (all code, U8 scales and BF16 row scales) | 1,306 |
| Full DiT block inventory | 48 |
| DiT IQ8R projection entries | 960/960 |
| Fused QKV/KV sources | 288 |
| Output IQ8R sources | 768 |
| Valid raw-plane rank roundtrips | 1,920 |
| Misaligned rank-specific K cuts refused | 384 |
| Fused bank source boundary rows verified | 756 |
| Checkpoint bias sources | 336 |
| Biasless video FFN-down sources | 48 |
| Connector IQ8R sources | 64 |
| Sampled connector/cross-modal rows | 104 |

Supported video shape: 32 heads × 128 = 4096 width; audio: 32 heads × 64 = 2048 width. Both **8/8/16** and **12/12/8** are full-head, legal IQ8R group-256 arrangements. The **10/10/12** audio split must reject four output K-shard types in each of 48 blocks (192 distinct source/type cases, or 384 rank-specific invalid cuts). Video 10/10/12 and other source-aligned K projections are not categorically banned.

The extra `ltx_tp3_builder_shard_test.cpp` source-backed-header fixture checks unique whole-head coverage with reversed rank enumeration, refusal for duplicated head ownership, missing head span, zero-width row and zero-width column. It does not implement any production group admission or rank lifetime: those belong to RUM06.

## Native declaration and test results

| Native Radiance model-backed receipt | Count |
| --- | ---: |
| Fused-row and biasless-column Lin declarations | 3,024 |
| Deferred post-reduction bias declarations | 2,640 |
| Invalid audio K cuts refused before weight allocation | 384 |
| Exact collective + once-only bias graph declarations | 2,016 |
| Global Q/K RMS fused projection graph declarations | 1,728 |
| Native Q/K graph operations | 17,280 |
| Full video/audio self-attention declaration graphs | 576 |
| C02 preflight negative cases | 4 |

C01's separate RadBuilder test returned 2,592 fused-row views, 432 biasless-column views and 336 biased-output views refused without deferral. C04 also checks duplicate-finalizer refusal. All are **source-backed graph declaration receipts**, not complete C06 execution.

**Revalidation:** **11/11** synthetic CTests; **6/6** native actual-source CTests (including prior merged C05 host issue); **12/12** focused Python tests with the actual model and frozen source explicitly supplied. No optional real-model/source tests were skipped. The standalone real-model numeric oracle verified the 960 DiT tensors and 64 connector sources.

## Already merged dependency versus remaining work

RUM13's generic native mixed-card AQL regression proof and LTX02's model-specific issued C05 Q/K RMS are independently merged. The actual LTX02 C05 native three-GPU graph passes **2,088,960/2,088,960** exact BF16 on seeded inputs and real checkpoint Q/K gains, including reversed physical GPU order. Neither changes the LTX01 geometry authority.

**Still open:** LTX02 full IQ8R C06 self-attention issued with checkpoint TP1 intermediate parity; LTX03 full 48-block six-kind multimodal attention and FFN; LTX04 serving/conditioning; LTX05 quality/performance; optional LTX06–07 worker/handoff; LTX08 qualified release. Production TP3 remains refused.

**Planning authority:** `LTX01` is only a registration candidate, not a Planning MCP item. This establishes technical completion and supplies the evidence for future registration and closure; it does **not** claim a generated `docs/planning/` edit or Ledger MCP event.
