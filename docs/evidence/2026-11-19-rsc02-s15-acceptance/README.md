# RSC02 S15 hardware acceptance (launcher against the native owner)

Leased runs on the production host, 2026-11-18 to 2026-11-19. Arm A is the bash launcher (from the archived pre-S16
checkout), arm B is `bc-serve serve`, both on one build, one A/B pair per cohort. The comparison projects the observed
engine and helper argv, env, role set and the stored manifest, and compares the greedy completion hash and the clean stop.

| Cohort | Layout | Result |
|---|---|---|
| Dense | `qwen27b-tp3-int8` | PASS: equal projection, equal greedy hash, clean stops (`dense-comparison.json`) |
| Helper | `flashnext-tp2xtx-r9700`, disk n-gram placement in both arms | Greedy hash equal (`eb2265d3...`), clean stops. The driver's comparison flagged 7 paths, all live per-start measurements or the owner's heat-launch nonce (`helper-comparison-before-normalisation.json`); with the normalisation of #1154 the recorded snapshots compare PASS with identical projection hashes |
| Native under load | `flashnext-4gpu`, RAM-only | `bc-serve serve` started in 750 s (two measuring starts, then the real one), canary correct, 538 requests, 44,939 completion tokens, 0 errors at 1/2/4/8 sessions, clean stop (`native-flashnext-4gpu-load.json`) |

## What the runs found and fixed

Eight attempts were needed; each failure was a defect in the driver or in the native owner, not in the launch semantics:

- Driver: missing launcher environment, a status timeout during the measuring plan, per-start memory readings compared as settings, pinned-manifest descriptors read from the wrong process, the launcher's heat file and layout definitions, and the n-gram placement a 91 GiB host needs for the two-card helper layout (#1134, #1136, #1137, #1139, #1141, #1142, #1146, #1154).
- Native owner: no wait for the helper cards' VRAM to be released between the measuring start and the final manifest write (#1147), and the heat guard stopping a recording-only session (#1151).

## Not run

The ABBA drift-control pairs and the RHP12 quality and performance gate. One pair per cohort proves launch equivalence;
the performance gate is separate. The launcher is retired by #1131 and stays at the tag `archive/pre-s16-launcher`.

`heat-recording-launcher-run.json` is the launcher run that recorded `tools/lab/radiance/heat/flashnext-mixed.*` (#1150).
