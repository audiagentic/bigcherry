# RSC02 slice B: shared acceptance of the native managed start

Build: the experiment build of the slice B branch with rad0070 composed (source #1639, canary fix). Two cohorts, stock power profile (no write), one lease window each.
Dense: qwen27b-tp3 (two gfx1100 and one gfx1201, mixed architecture). Helper/MoE: flashnext-4gpu (three helpers plus the engine).

| Check | Dense mixed TP3 | Helper/MoE flashnext |
|---|---|---|
| First start: measuring engine(s) on native `--config`, plan measured, final engine, canary, running | pass (about 30 s) | pass (about 3.5 min, three helpers measured then final) |
| Owner event order | planned, engine, healthy, canary passed, runtime guard ready, running | segment selected, power applied, helpers measuring, engine measuring, planned, helper0-2 started and ready, engine, healthy, canary passed, running |
| GET /v1/models | `qwen27b` only, no `default` alias | `flashnext` only, no `default` alias |
| Completion with the listed model | owner canary passed (thinking model: an 8-token raw check returns only reasoning) | " Paris. The capital of Germany is Berlin" |
| SIGHUP to the engine | engine pid unchanged, health 200 to 200, log: "reload refused: a managed start is not reloaded" | same |
| POST /admin/reload, POST /admin/models/ID/unload | 403, 403 | 403, 403 |
| Stop | stop requested, stopping, power restored, stopped; owner exit 0 | stop requested, stopping, segment retired, power restored, stopped; owner exit 0 |
| Second start, same work root | stored plan reused, running in 21 s | stored plan reused, running in 100 s |
| SIGKILL of the engine | stopping, power restored, failed; owner exit 1; no new engine | stopping, segment retired, power restored, failed; owner exit 1; no helper process left; no restart |

Defects found and fixed on the branch before this passed: the native Fleet kept only cards of the first card's architecture (mixed TP refused `cards: [0,1,2]`); the owner start canary still sent the removed implicit model name `default`.
Not run here (by design): offline KL (card-free admission quality tests cover the argument shape), real power apply and restore (PWR10).
