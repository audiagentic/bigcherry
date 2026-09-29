# RD33 scoped promotion follow-up

## 1. Supportable claim

PASS, scoped only. Evidence now supports: **"On dual Radeon RX 7900 XTX (gfx1100), `-sm tensor`, dense Q8_0 Qwen3.8-27B plain non-speculative decode, RD33 improves fixed-length generation throughput by about 4.4-4.5% at tg512/tg2048 in an 8-pair order-balanced A/B; this claim excludes MTP/speculative decoding."**

Do not claim generic gfx1100 decode, other models/quantizations/device counts, quality equivalence, prefill improvement, or MTP benefit. The MTP n_max=4 lane is explicitly excluded: acceptance changes 0.90101 -> 0.95580 and tg2048 is -5.65%; outputs/work differ. Prefill is also not a benefit claim (measured -0.32% pp1024, -0.13% pp4096).

## 2. Proposed amended correctness contract

Replace stock-relative `max_abs_logprob_diff <= 5e-4`; stock deliberately quantizes activations to Q8_1 and is not the numerical oracle for RD33.

### Primary gate: backend/reference accuracy

Positive scope: gfx1100, dense Q8_0, `ncols_dst==1`, representative production K/M shapes including Qwen3.8-27B decode shapes. Use the existing backend/reference test machinery (CPU/backend reference) to evaluate the intended operation: Q8_0 weights dequantized and multiplied by original F32 activations with F32 accumulation. Require all selected shapes to pass the existing backend-reference tolerance; concretely gate normalized error/NMSE <= 5e-4 and no NaN/Inf. Report stock and RD33 errors against the same reference; require RD33 error <= 5e-4 and no >2x worsening versus stock on aggregate/reference error. This is an accuracy gate, not byte identity.

### End-to-end distribution guard

Use a fixed, versioned prompt/corpus set and deterministic server settings. `llama-server` with `n_probs` can capture token probabilities, but top-N output cannot prove full-vocabulary KL unless the harness exposes complete logits; therefore KL is optional until a full-logit artifact exists. Required server diagnostics: >=32 fixed prompts, >=256 generated tokens/prompt, temp=0; report top-1 agreement and first-divergence positions versus stock. Because divergence is intentional/possible, set a guard rather than equality: top-1 agreement over common-prefix decisions >=95%, and prompt-level greedy-divergence rate <=25%. The currently observed 2/4 is insufficient to pass this proposed guard and is too small a sample to calibrate it; thresholds are prospective and must not be weakened after collection.

If full-vocabulary logits can be emitted by an existing producer, additionally require mean KL(reference || RD33) <= mean KL(reference || stock) + 1e-4 nats/token and 99th-percentile KL <= 1e-2 nats/token. Reference must be an independently computed F32-activation/backend reference, not stock Q8_1 activation quantization.

### Broader quality guard

Run `llama-perplexity` at `-b 1 -ub 1` on a fixed, checked-in/versioned corpus of at least 100k tokens spanning prose/code/multilingual text, same model/context/settings. Require RD33 PPL relative change <= +0.25% versus stock; report bootstrap/chunk CI and require its upper bound <= +0.5%. Existing 4096-token PPL (11.5989 vs 11.6132) is supportive only, not sufficient campaign evidence.

### Performance acceptance

Plain non-MTP decode only: order-balanced A/B, fixed lengths, tg512 and tg2048; require positive CI95 lower bound for both. Prefill is a no-regression control: pp1024/pp4096 regression <=1%. MTP is an excluded positive lane and must not be used to satisfy promotion.

## 3. Remaining validated-evidence blockers / minimum runs

1. **Activation evidence:** run one traced plain-decode subject/control probe with `BIGCHERRY_PATCH_TRACE=1`; require exact RD33 hit marker on subject and absence on control, bound to binaries/source identity. If patch has no permanent marker, add one before the contract campaign.
2. **Primary correctness campaign:** backend-reference exact/representative `ncols_dst==1` Q8_0 production shapes on gfx1100; capture normalized error for stock and RD33 against the same reference.
3. **Quality campaign:** fixed >=100k-token PPL corpus plus >=32-prompt deterministic server distribution/greedy diagnostics. Add full-logit KL only if current tooling can capture full logits without new invasive machinery.
4. **Contract performance campaign:** reuse/reproduce the order-balanced non-MTP tg512/tg2048 A/B under the amended contract so `patch-verify-evidence` can consume validated evidence rather than ad-hoc lab evidence; include pp controls.
5. **Architecture coverage:** contract/patch currently validates gfx1100 only. Do not block the narrowly worded gfx1100 promotion on gfx1201/gfx1030; instead keep runtime eligibility gfx1100-exact and require other architectures only before broadening scope.

Priority is 1 -> 2 -> 3 -> 4. Do not spend hardware on MTP performance qualification for this promotion scope.

## 4. MTP runtime gating

BLOCKED for production-wide enablement unless MTP is excluded at runtime/composition. The measured MTP lane has a deterministic acceptance shift and tg2048 -5.65%, so enabling RD33 indiscriminately in an MTP server contradicts the only supportable promotion scope.

Preferred solution: gate RD33 off when speculative/MTP verification is active, **if that state is available cheaply and unambiguously at dispatch/configuration time**; then validate both activation (plain decode hits) and non-activation (MTP does not hit). If the CUDA/MMVQ dispatch layer cannot cheaply know MTP mode without cross-layer plumbing, do not add brittle inference based on tensor shape/content: make patch selection/composition mutually exclusive with the production MTP profile instead. This gating implementation is a code change and therefore outside this docs-only follow-up, but it is a blocker to shipping RD33 in configurations that may enable MTP.