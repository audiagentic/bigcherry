# bc-llamacpp-11552.0.0

BigCherry on llama.cpp `b11552` (`23b0202a189c`), BigCherry commit `ebe35277bc7e`.
Previous release: `bc-llamacpp-b11474-r9`.

## Bump record

release 11552.0.0 2026-10-12

completion gate (pin-status --complete --all-remotes):
pin-status: local
  pin           b11552 -> 23b0202a189c
  vendor HEAD   23b0202a189c (tag b11552)  [56 modified, 48 untracked]
  releases      0504396140d1 audited, 0adcc3bb5710 broken, 22dc605c4ead patched, 23b0202a189c patched, 2578138397d7 generated, 28ff0958291c patched, 370cb12e8bb7 patched, 434ddbbc0e30 patched, 50182a53fa2c generated, 6ea215d171fd patched, 91f6a6cf3613 patched, abc123 validated, b1ff4ca23630 patched, b9acf138a1e2 patched, c061df19838f broken, c841aeeb8bb2 patched, d77dd0806dc2 patched, d7bd3bfcad3e patched, d89651a7b205 patched
  descriptors   0adcc3bb5710, 23b0202a189c, 2578138397d7, 28ff0958291c, 370cb12e8bb7, 4801e3c567d5, b9acf138a1e2, c061df19838f, c841aeeb8bb2, d59d455fd8ea, d77dd0806dc2, d7bd3bfcad3e, d89651a7b205
  bigcherry     5b65e5fcb49b
  marker        b9acf138a1e2 -> 23b0202a189c (b11552) [committed-clean]
  VERDICT       consistent  (content-unverified: 56 tracked file(s) modified (expected after patch application; verified by audit, not here))

pin-status: brutus (remote)
  vendor HEAD   23b0202a189c
  pin           b11552 -> 23b0202a189c
  bigcherry     5b65e5fcb49b
  VERDICT       consistent

AGGREGATE  converged

COMPLETION  PASS

hardware evidence:
Pin b11474 -> b11552 (23b0202a189c): hardware evidence, Brutus (2x RX 7900 XTX gfx1100, R9700 gfx1201, RX 6900 XT gfx1030 drafter), 2026-10-11/12.

Build: b-pq11552 (bigcherry:stock:linux-multi, gfx1100,gfx1201,gfx1030) compiled first time from the reconciled production set (61/61 patches apply; strict audit 34/34). Base for every comparison: b-p11474, same recipe on the previous pin.

Flash-Next (UD-IQ4_XS, production profile, f16 KV, MTP drafter on the 6900 XT)
- Prefill: 8K flat; 24K over four prompts 1447-1466 t/s vs 1411-1435 t/s (+2.3%); 98K +1.9%.
- Fidelity against a CPU reference at 24K (38.7K tokens filled, 24 probes, flash-fidelity.sh; runs fidO / fidN):
  old pin top-1 21/24, TV mean 0.1085; new pin top-1 21/24, TV mean 0.1087. On each pin the three GPU arms are
  bit-identical to each other (TV 0.0000). The new pin is not numerically further from the reference.
- MTP decode at 24K over four prompts, two runs each (dacc-oldpin / dacc-newpin): pooled 73.1 -> 69.0 t/s with draft
  acceptance 59.1% -> 55.6%. The greedy text differs between the pins. Per prompt the sign is mixed (two prompts
  slower, two faster: 85.0->67.6, 71.4->77.2, 78.0->68.4, 58.0->62.8 t/s) and decode t/s x drafted tokens is the
  same on both pins (39.65k vs 39.40k, -0.6%), i.e. the cost of a step is unchanged and the pooled difference is
  which continuation the four prompts happened to take. Not treated as a regression; no patch.
- Reconciled patches 1348 / 1346 / 1359 checked: identical text with catch-up or the pipeline off.

Qwen3.8-27B Q8 (dual XTX, production flags, --allreduce cpu-root; prod27b-ab.sh ABBA, run nr27-pin)
- 10K: prefill 1305.8 (old, warm run) vs 1292.4 / 1307.7 (new); decode 74.4 vs 74.3-74.6 t/s; accepted 177/310 vs 177/311.
  (First old-pin run 1219.9 t/s was the cold start of the session.)
- 32K: prefill 1244.9 / 1253.8 (old) vs 1271.5 / 1275.8 (new), +1.9%; decode 73.1 / 73.4 vs 74.7 / 69.8 t/s (one low
  run on the new pin, 49.6 vs 46.3 ms/step, not repeated); accepted 181/293 vs 182/289.
- Greedy text identical across pins at 32K; at 10K each pin is self-consistent and the two differ.

Not carried over: six non-production patches do not apply on b11552 and have revision-bound known_broken
dispositions (1250, 1256, 1257, 1266, 1267, 1338). 1337 is superseded: the pin carries upstream #29887 itself. Tracked-sources check: 63 findings, triage in PA50.

## Patches

### State changes

- `1337_moe_expert_caching`: untested -> superseded

### Implementation changed

- `0300_mmq_forced_j`
- `1006_rdna4_mmq_q6k_codegen_fix`
- `1237_rd30_moe_mmq_compact_grid`
- `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2`
- `1294_topk_deterministic_ties`
- `1307_q81_activation_cache_mmvq`
- `1321_mtp_ahead_primitives`
- `1333_mixed_batch_on_demand`
- `1346_mtp_prompt_overlap`
- `1348_mtp_deferred_catchup`
- `1350_mmq_few_tile_streamk`
- `1359_prefill_pipeline`
