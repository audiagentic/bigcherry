# bc-radiance-170.0.0

BigCherry on radiance `v1.7.0` (`1e968f6e1d86`), BigCherry commit `d7ad32cfa5c5`.
First tagged release.

## Bump record

release 170.0.0 2026-10-17
upstream: radiance v1.7.0 (revision 1e968f6e1d86e2544455b00e9e57c09690d9ea10)

completion gate (pin-status --complete --all-remotes):
pin-status: local
  pin           v1.7.0 -> 1e968f6e1d86
  vendor HEAD   1e968f6e1d86 (tag v1.7.0)  [0 modified, 0 untracked]
  releases      0504396140d1 audited, 0adcc3bb5710 broken, 1e968f6e1d86 patched, 22dc605c4ead patched, 23b0202a189c patched, 2578138397d7 generated, 28ff0958291c patched, 370cb12e8bb7 patched, 434ddbbc0e30 patched, 50182a53fa2c generated, 6ea215d171fd patched, 91f6a6cf3613 patched, abc123 validated, b1ff4ca23630 patched, b9acf138a1e2 patched, c061df19838f broken, c841aeeb8bb2 patched, d77dd0806dc2 patched, d7bd3bfcad3e patched, d89651a7b205 patched
  bigcherry     6dc17887235a
  marker        48f7b8eb6d70 -> 1e968f6e1d86 (v1.7.0) [committed-clean]
  VERDICT       consistent

COMPLETION  PASS

hardware evidence:
radiance v1.7.0 (1e968f6), BigCherry port MEN12 (#261, #263, #264); Brutus, 2026-10-17.
Build: lane radiance-lab:stock:linux-radiance (+moe-remote-handoff) ok; pin-bump compile gate ok on platform linux-radiance (GCC 14) - every kernel plugin reports RAD_ABI_VERSION 17, r3_moe_shm_test passed; diagnostic build radiance-diag:moe-diag ok.
Smoke: CPU smoke PASS (2.0-2.5 s, 146 MiB, no GPU) on stock, +rad0002 and native builds; strict re-audit 4/4; pin-status --complete PASS.
A/B (ab-benchmark helper-ab.json, Flash-Next, R9700 engine + two XTX helpers, natives on, 2 balanced rounds, R3_MOE_HELPER_ROWS 0 vs 1): ROWS=0 pp4096 2,049-2,050 tok/s, tg512 122-126, tg2048 140-141 (1.3.0: ~1,990 / 128 / 141-143); ROWS=1 slower as expected (pp4096 480-487).
Quality: helper-kld (vs 1.3.0 two-XTX exact reference) gate PASS - helper-on KLD mean 0.0642, P99 0.684, P99.9 1.93, top-1 88.7%, identical to 1.3.0; 27B MXFP4 two-XTX attention native KLD 0.0040, top-1 96.6%, PPL 7.358 vs 7.368 (1.3.0: 0.0035 / 96.8%).

## New and changed

- Each engine now declares how its server is run, so the lab server runner can launch llama.cpp or radiance from the same code. (MEN03)
- Radiance is now built, patched, configured and A/B-measured through the same BigCherry patch sets, recipes, profiles and campaign harness as llama.cpp. (RAD01)
- The R9700 expert helper for Flash-Next on two XTX was shown to be numerically sound and raises decode speed by about 78%. (RAD03)
- Flash-Next can now run with the R9700 as the main engine and both RX 7900 XTX cards computing experts; prefill reaches ~1,880 tok/s, the fastest Flash-Next prefill so far on Brutus. (RAD02, RAD03)
- Flash-Next prefill on Brutus reaches about 2,640 tok/s with the R9700 running the model and both XTX cards computing the cold experts, more than double the original two-XTX setup. (RAD02)
- Flash-Next decode on Brutus reaches about 108 tok/s with the R9700 running the model and both XTX cards computing experts, four times the original two-XTX setup. (RAD05, RAD06)
- Added an experimental prefill kernel for the radiance expert helper; it measured slower and stays off. (RAD06)
