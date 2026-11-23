---
id: VTP50
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T21:05:38.143523+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Replace the gfx1100 WMMA shim at every remaining site with grouped native-layout kernel families (accumulate in the native layout, convert once)

## Description

**Problem.** On gfx1100 every gfx12-style WMMA call in the shared kernel sources goes through a compatibility shim: both operand fragments are re-laid out across lanes (12 -> 11 lane exchanges), the gfx11 instruction runs on a zero accumulator, the product is re-laid out back (11 -> 12) and then added to the running sum. The exchanges cost far more than the matrix instruction. Measured so far (VTP37): the K-parallel int8 tile at two or more token tiles ran 27-37% faster in isolation once the accumulator stayed in the gfx11 layout and was converted once per scale group (-9.1% / -6.7% decode ms/step at 6 / 8 streams, 2020.8.2); the drafter 4-bit row, logits and decode attention are the next decode sites (slices drafter-two-m16, logits-acc-once, attn-decode-acc-once).

**Goal.** No kernel on a shipped path calls the per-instruction shim. Instead of converting 26 sites one by one, convert them as a few grouped families that share one native-layout fragment layer, so each family is a mechanical change and a single qualification.

**Families (to be confirmed by the reach audit in step 1).**
1. *Integer-activation GEMMs* (int8 and int4 activations with int8 / int4 / 2-bit weights, tiled and narrow forms, prefill and decode rows). Integer group sums are exact, so accumulating natively is bit-identical: first, cheapest, safest.
2. *16-bit-float-activation GEMMs* (4 / 5 / 8-bit and bf16 weights, the LoRA kernel, and the shared fragment helper they all include). These funnel through one small wrapper, so one native wrapper converts the family. Float sums need the order-preserving form (add the zero-accumulator product in the native layout, same order as the shim's add) to stay bit-identical; native accumulate into the running sum is measured separately and ships only with an explicit numerical qualification.
3. *Attention* (decode, paged-window, generic, ViT, dense, selection, sparse-attention): the query-key and probability-value chains; the output accumulator is carried across KV blocks, so convert once per output tile at the end.
4. *fp8-through-bf16 sites* (the emulated fp8 matrix multiply and the fp8 tile): same float rules as family 2.
5. *Mixture-of-experts and the remaining one-off sites.*

**Shared layer.** One header in libr3 (beside the existing native wrappers) provides: native-layout fragment loads for A and B straight from LDS or registers (so the 12 -> 11 exchange disappears where the data can be loaded in the right layout in the first place), a chain type that holds the gfx11-layout accumulator, `convert_once()` to the gfx12 layout at the end of a scale group, block or segment, and an order-preserving add. Every site then differs only in where it calls `convert_once()`. This keeps the families grouped and the build target grouped (one libr3 family target per family).

## Steps

1. **Reach audit** (no hardware): list every call site of the shim in the build tree, which kernel rows and manifests reach it (shipped layouts for the 27B int8, MXFP4, drafter and the other models), its VGPR / spill record on gfx1100 and its share of the step trace where one exists. Output a table that fixes the family order by reach times shim cost; sites with no shipped reach get a NO-GO or a deferred mark.
2. **Shared layer** in libr3: native fragment loads, chain type, convert_once, order-preserving add; unit-tested on the card against the shim for random data (the shim result is the reference).
3. **Family 1** (integer): convert, byte-identity gate at M 1-64 over the shipped shapes, cold timing replay, decode / prefill A/B where the family is on a shipped path.
4. **Family 2** (16-bit float GEMMs) with the shared wrapper; bit-identity first (order-preserving), then the native-accumulate variant only if the order-preserving form leaves a gap above noise, qualified by KLD.
5. **Family 3** (attention) on top of the existing attn-decode slice; **families 4 and 5** by reach.
6. Each family: slice + PR through the usual path, GPT session briefs for the mechanical conversions, my ABBA runs on the release that carries them.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Byte identity of outputs against the shim build for every converted kernel at the shipped shapes, M 1-64 (integer families exactly; float families in the order-preserving form).
- Resource record per kernel: VGPRs and spills before / after.
- Cold timing replay per family, then decode ms/step and prefill TTFT A/B (ABBA, untraced, 3 seeds) on the release that carries the family; acceptance and temp-0 text unchanged; KLD for any non-order-preserving float change.

## Effort & Risk

Large, but split into independently shippable families. Main risk: native fragment loads change LDS layouts and the load code of kernels that feed several instructions, so a few sites may need their data path reshaped rather than just their accumulator; those are listed after the audit and scoped separately.

## Standards



## Acceptance Criteria

- The reach audit table is committed to docs/evidence and no shipped-path kernel calls the per-instruction shim, or each remaining site carries a recorded reason (no reach, no gain above noise, or bit-identity not achievable) with its measurement.
- Each family's serving effect is recorded as measured, not inferred from isolated timing; no family is chained into another's number.

## Notes

**2026-11-24 census and results so far.** Static scan of the 2020.8.4 gfx1100 code objects (exchange instructions per matrix instruction by kernel family) joined with the traced op list ranks the families by exposure x ms: drafter 4-bit m16 tile (2.39 ms rank 0, 7.2 exchanges per WMMA; slice iqr-acc-once), K-parallel int8 tile fragment exchanges (13.4 ms pass), decode attention (1.33 ms, 12 per WMMA; shipped as 2020.8.6), logits (done, within noise). Replacing the exchange primitive (ds_bpermute -> single-instruction half-wave swap) is bit-identical and changes nothing on the K-parallel tile at six rank-0 shapes (-4..+7%, noise): the exchange latency is hidden by the stream-bound tile. **Parked:** the K-parallel tile's token / weight fragments loaded directly in the gfx11 layout (slice kpar-frag-native) failed bit identity on its first sweep (layout mismatch at M 1-48); not requalified, value doubtful after the serving A/Bs; the layout mismatch is the thing to find if it is ever revived. Serving A/Bs so far: accumulate-once -9.1% / -6.7% step (6 / 8 streams), drafter two-M16 -9.8 / -11.3%, logits within noise, look-ahead removal within noise (evidence notes under docs/evidence/2026-11-24-vtp37-*).



**2026-11-24 IQ6R exposure.** In the 2020.8.5 gfx1100 code objects the four-to-seven-bit IQR families differ: the 16-row decode tile (`iqr_m16`, 404 instantiations) has 2 spilling instances (8 B private) and serves decode, including 17-64 rows as 16-row slices; the wide tile (`iqr_m64`, 544 instantiations) spills in every instantiation (240-256 VGPRs, up to 3640 B private) and serves rows above 64, i.e. prefill. If an IQ6R container ships (trellis-coded 6/7-bit trunk), its prefill and time-to-first-token rows are the exposed ones and the wide tile's spill and shim exposure becomes a real lever then; until then it stays unranked for the IQ8R layouts (which do not use it on shipped paths). Carried in the ranking, not as done.



**2026-11-24 accept/reject under the no-regression standard.** ACCEPT-WORK: accumulate-once families (drafter IQR tile in progress, attention, prefill GEMMs when their bit identity and replay pass); single-instruction half-wave swap in the compatibility header: bit-identical and cleaner, but the K-parallel replay showed +3..+7% on three M 32 cells (3072x5120, 5120x2304, 13312x5120) inside a noisy driver, so it is accepted only if a full-plugin cold and warm replay shows no regression beyond spread (re-run, not assumed). REJECT: K-parallel token and weight fragments loaded directly in the gfx11 layout (failed bit identity on its first sweep, 432 of 576 cases); wave-per-token-tile (slower); the waves-per-SIMD cap rule for the RDNA2 helper (26 of 54 cold cases slower, production never reaches that kernel). Iqr wide-tile spills (prefill) stay ranked, not done.

## Change Log

- 2026-11-23T21:05:38.143523+00:00 (created-by): Created by agent
- 2026-11-23T21:06:00.461438+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T21:45:17.328753+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:06:33.668979+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:21:36.583650+00:00 (updated-by): Updated: section:notes
