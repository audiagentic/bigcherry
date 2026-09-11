# 1224 (HI18/GP04): standalone SPLIT_REDUCE correctness probe, test-hip-reduce

## Scope

Wires a standalone host executable (source in the overlay) into the build
that drives the real production META backend through
`GGML_HIP_REDUCE_PLAN=auto|rccl|meta` with a minimal 2-node graph, reporting
machine-readable execution facts to `tools/bigcherry/tuning/reduction.py`.
This is a native-half correctness-comparison gate that exercises the real
META split-state/reduce-plan machinery directly -- not a Python
reimplementation that could silently disagree with it. **D=2 device-count
slice only** (see HI84 for the planned D=3/D=4 extension).

## Why

HI18 needed a native-half correctness-comparison gate that exercises the
real production reduce-plan machinery; this is the tool HI15/HI16's review
update assigned as the missing piece.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI18).

## Real hardware evidence (2026-09-02, Brutus, {0,1} pair)

1. **Mechanism validation**: 63/63 synthetic RCCL/META/AUTO executions
   across 6 numerical patterns, 3 seeds, and swapped device order.
2. **Real production-signature corpus (GP04, "CLOSED 2026-09-02")**: all 9
   real reduction signatures captured from actual Qwen3.8-27B MTP dual-XTX
   traffic (`artifacts/hardware/20260820-qwen38-mtp-dual-xtx/reduction.jsonl`,
   element_count 5120 through 2621440) plus one additional real signature
   from an earlier Qwen3.5-9B capture (RV59, element_count=8192) -- **10
   signatures x 3 providers x 18 cases (6 patterns x 3 seeds) = 540 real
   hardware executions, 540/540 PASS, zero failures.**
3. **BF16-regime bound fix independently verified non-blanket**: the two
   largest real stress signatures (element_count 2621440/2606080) --
   exactly the cases that originally exposed a correctness-harness bound
   gap (production's real BF16 activation compression for large reductions
   vs the harness's pure-FP32 analytical bound assumption) -- now pass
   cleanly on RCCL (worst_nmse ~3.31, within the corrected BF16-aware
   bound) while META stays exact-F32 tight (worst_nmse ~7.8e-16). This
   confirms the bound fix widens tolerance only where the real BF16-wire-
   compression regime actually applies, not a blanket loosening that would
   mask genuine errors.
4. Two earlier corpus attempts in the same investigation were invalidated
   and explicitly discarded (one ran stale pre-fix code, one ran an
   incomplete first-draft bound caught by GPT review before being
   trusted) -- kept on disk for the record, not counted as evidence.

Full evidence: `artifacts/gp04-full-corpus/` on Brutus (per-signature logs
+ `reduce-correctness.jsonl` for all 10 signatures).

## Lifecycle: promoted to `validated` (2026-09-11, D=2 scope)

The patch's own inline `STATE` comment named the exact remaining gate:
"the required tracked validation record exists AND the real recorded
production signatures (not this synthetic stand-in shape) have been run."
GP04 closed exactly that gate on 2026-09-02, but the state fields were
never updated to match. Requested and obtained an explicit GPT
solution-approval decision (dev-gpt-agent, session `ses_f46829ab914d4ef7`,
`req_09004a8da1524f8f`): **APPROVE** -- "The 540/540 real-hardware pass
corpus covers all known production signatures and all three providers,
with the BF16 tolerance correction independently justified and
non-blanket; heterogeneous/D>2 qualification remains separately out of
scope and does not block D=2 validation." `patch.py`'s `STATE`,
`patch.toml`'s `state`, and `SUMMARY.md`'s `Status` corrected together.

## Known limitations

- **D=2 only.** Heterogeneous-topology (`{0,2}`/`{1,2}`/`{0,1,2}`) and
  D=3/D=4 promotion are explicitly out of this patch's scope, tracked
  separately under GP01/GP06 and HI84 respectively.
- No `validation.toml` adapter exists (`kind = "diagnostic"`, not eligible
  for the local-framework adapter path).
