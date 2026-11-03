# bc-radiance-1160.0.0

BigCherry on radiance `v1.16.0` (`769b4c64a671`), BigCherry commit `f1b676e679eb`.
Previous release: `bc-radiance-1.14.0-r0`.

## Bump record

pin-bump lane build: radiance:stock:linux-radiance build_plan_id=634453dc1cd981f83b71ac9a8ee72a0f binary_sha256=8e3a19db795407ceaf3dc22a3f09837215f78f9695579079583cd6a3f447c58b
pin-bump lane build: radiance:stock:linux-radiance build_plan_id=6d933f0c5d28de8dc1274f959b047470 binary_sha256=8e3a19db795407ceaf3dc22a3f09837215f78f9695579079583cd6a3f447c58b

release 1160.0.0 2026-11-03
upstream: radiance v1.16.0 (revision 769b4c64a671e4b52c43507e50fba1cf591bd139)

completion gate (pin-status --complete --all-remotes):
pin-status: local
  pin           v1.16.0 -> 769b4c64a671
  vendor HEAD   769b4c64a671 (tag v1.16.0)  [10 modified, 11 untracked]
  releases      0504396140d1 audited, 0adcc3bb5710 broken, 1e968f6e1d86 patched, 22dc605c4ead patched, 23b0202a189c patched, 2578138397d7 generated, 28ff0958291c patched, 370cb12e8bb7 patched, 434ddbbc0e30 patched, 4a61d7296bc5 patched, 50182a53fa2c generated, 6645337f695a audited, 685ec5bd858a patched, 6ea215d171fd patched, 769b4c64a671 patched, 7a7707a95559 patched, 91f6a6cf3613 patched, abc123 validated, b1ff4ca23630 patched, b9acf138a1e2 patched, c061df19838f broken, c841aeeb8bb2 patched, d77dd0806dc2 patched, d7bd3bfcad3e patched, d89651a7b205 patched
  bigcherry     61610f89fd36
  marker        769b4c64a671 -> 769b4c64a671 (v1.16.0) [committed-clean]
  VERDICT       consistent  (content-unverified: 10 tracked file(s) modified (expected after patch application; verified by audit, not here))

COMPLETION  PASS

hardware evidence:
radiance v1.16.0 release evidence (bigcherry main after #555).
Bump: pin-bump drift, audit, patch rebase (9/9 clean), compile gate, lane build radiance:stock:linux-radiance and CPU smoke all pass. Plugins export the 1.16 ABI layout fingerprint (libbcgfx12, libr3moe, libr3ngram added; the remote-MoE helper checks its kernel plugin's).
Speed, interleaved v1.16 bump vs v1.14 main on Flash-Next (R9700 engine + two 7900 XTX + 6900 XT tail), 15 rounds each (runs/rad116-ab1): decode step 1 stream 24.58 vs 24.56 ms (+0.07%), 6 streams 48.15 vs 48.56 ms (-0.83%); prefill TTFT 386 vs 385 ms (1 stream), 5.68 vs 5.66 s (6 streams). Text differs from v1.14 (new engine numerics). 1-stream engine tok/s 140.2 vs 144.5 from one prompt's MTP draft acceptance on its new text (0.850 -> 0.813); 6-stream acceptance unchanged (~0.585).
Quality (runs/rad116-kld, KLD vs a fresh v1.16 engine-local reference, helper-kld limits): two-XTX helper offload passes every limit (vs helper-off: mean -0.005, p99 -0.056, p99.9 -0.538, top-1 drop 0.30 pt, ppl -0.2%). The three-helper 6900 XT tail layout fails p99.9 only (+0.78 vs 0.5; RAD46, +1.58 on v1.14).
Known open items: RAD46 (RDNA2 tail-helper tail quality); rad0008 current-pin revalidation (MEN26).

## Patches

No patch changes.

## New and changed

- BigCherry now builds on radiance 1.16.0 (server hardening, the place caching layer, the RDNA3 libr3d plugin), with the same decode and prefill speed on Flash-Next and passing quality on the two-XTX helper layout.
