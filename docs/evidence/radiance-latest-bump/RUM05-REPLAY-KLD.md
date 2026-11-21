# RUM05: stored replay and offline KL on cards

Installed release bc-radiance-1170.37.0 (the Gate E prefix, see RUM09-GATE-E.md), one gpu lease per run. All paths below
are under `<dev-root>/tmp/bigcherry/gate-e-1170.37.0/`.

## Starts: fresh, second named start, stored-manifest replay

Each config was started three ways in its own run directory with the same `--work-root`: a fresh named start, a second
named start (`rum05/<config>/cachehit`), and `start --manifest <the fresh start's manifest> --run-dir NEW` (replay).
Every start reached `up` and answered the canary with the expected completion; every one stopped cleanly.

| config | cards | fresh | second named start | replay |
|---|---|---|---|---|
| flashnext-r9700 | R9700 | up, canary ok | up, canary ok, measured again | up, canary ok, 0 measurement attempts |
| qwen27b-tp3 | XTX, XTX, R9700 | up, canary ok | up, canary ok, measured again | up, canary ok |

Result lines: `rum05/flashnext-r9700/result.txt`, `rum05/qwen27b-tp3/result.txt`.

The second named start measures again on this build: a named start keeps its plan under its own run directory and only
`start --manifest` reuses one. The behaviour change that lets a named start reuse the last measured plan is tracked
separately.

## Replay restart cost (flashnext-r9700, UTC)

| start | created | power admitted | running | total |
|---|---|---|---|---|
| fresh | 22:55:04 | 22:55:10 | 22:56:16 | 72 s |
| replay, no `--work-root` | 22:57:33 | 23:05:59 | 23:06:52 | 9 min 19 s (8.4 min before power admission) |
| replay, with `--work-root` | 23:17:21 | 23:17:26 | 23:18:05 | 44 s |

Without `--work-root` the replay fingerprints the whole model before power admission (qwen27b-tp3: 2 min 4 s, created
23:08:14, power admitted 23:10:18). With it, the replay skips measuring and the fingerprint and is faster than a fresh
start by the measuring phase. Session event times: `rum05/<config>/<arm>/run/session.json`.

## Offline KL start on the mixed-rank config

`bc-serve quality qwen27b-tp3` (a copy of the config with `num_speculative_tokens` 0) scored against the bf16 reference
(`kld/out.json`, `kld/quality.log`, exit 0): 8240 positions, mean KLD 0.0012823, median
0.000444, p99 0.01112, top-1 agreement 0.9727, perplexity ratio
1.00049. The serving-split result recorded in RUM06 for the same ranks (6, 6, 4) is 0.001282, so the
offline Config keeps the serving start's physical card order.
