# Dual-XTX production-parity bench log

Real-hardware validation runs against Brutus's dual RX 7900 XTX pair
(GPU 0/1, `HIP_VISIBLE_DEVICES=0,1`), compared against the `dual-xtx-27b`
production profile (`/mnt/vault/development/llmhosts/llamacpp/bench/config/model-profiles.json`,
model Qwen3.8-27B-Q8_0 with built-in MTP draft).

Production launch flags (see also memory `project-dual-xtx-baseline`):

```
-sm tensor --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0
--repeat-penalty 1.0 --flash-attn on --ubatch-size 512 --batch-size 2048 -ngl 99
--threads 8 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4
-ctkd q8_0 -ctvd q8_0 -c 64000 (production: -c 222000)
```

**IMPORTANT — build source**: always build from a clean git checkout
(`~/bigcherry` on Brutus). Never `git`/`cmake` against
`/mnt/vault/development/llmhosts/bigcherry` -- `docs/reference/BUILD.md`
documents that path as file-transfer-only (SMB share), and its
vendor/llama.cpp is a mutable checkout with patches applied in place, not
one of bigcherry's isolated per-identity worktrees. Use the canonical
command:

```
PYTHONPATH=tools python -m bigcherry build \
  --lane <source>:<build>:<platform> --binary-relative-path bin/llama-server
```

---

## 2026-08-19 -- documented production baseline

- Build: BigCherry with the HI71 fix, matching production flags exactly.
- Method: `mtp-dual` bench-runner config via `server-bench`, 5 reps.
- pp4096: ~1249 t/s (1242-1254 across reps)
- tg2048: ~93-109 t/s (avg ~102)
- tg512: ~61-105 t/s (avg ~81, higher variance)

## 2026-08-26/27 -- post-pin-rebase re-measurement

- Build: `bigcherry:control:linux-multi` at `tuning-code-rebase` HEAD
  (~1ec29ce..4b6e784), pin `b10502`, built via the canonical `bigcherry
  build` command above on `~/bigcherry` (never the `/mnt/vault` checkout).
- Method: direct `llama-bench -sm tensor -fa 1 -ub 512 -b 2048 -ngl 99
  -t 8 -ctk q8_0 -ctv q8_0 -p 4096 -n 128 -r 5` (pp4096 only, 5 reps --
  NOT the full `mtp-dual`/`server-bench` harness run, see "not yet logged"
  below).
- **pp4096: 1425.51 +/- 1.54 t/s** -- ~14% above the 2026-08-19 baseline,
  equally tight across reps.
- Separately, a live `llama-server` launch with the full production flags
  above confirmed a working MTP draft context (`common_speculative_init_result:
  creating MTP draft context...`) and a real completion request generated
  coherent output at 69.89 t/s with 53% draft acceptance (168/316 draft
  tokens accepted) -- a single-sample liveness check, not a 5-rep number,
  but within the 2026-08-19 baseline's tg range.

**Why pp went up**: NOT attributable to any single session's own changes.
The upstream llama.cpp pin advanced `b10362` -> `b10502` on 2026-08-20 (the
day *after* the 2026-08-19 baseline), and roughly 101 commits touched
`patches/`/`src/`/`config/external-sources.toml` between 2026-08-19 and
2026-08-27 -- including at least one documented prefill-specific
optimization (a ported MoE MMQ launch-grid compaction patch, AMD-reported
+1.9..5.4% prefill across the Qwen3 family; see
`config/external-sources.toml`'s PR #63 entry). Treat 1425 vs 1249 as
"the project improved over about a week of real work", not evidence for
any specific patch -- a real controlled A/B needs the same pin and patch
set, varying only the change under test.

**Not yet logged in the bench harness**: this run bypassed
`run_bench.py`'s normal `--lane`/results.json/results.db tracking (no
harness lane is registered for an ad-hoc current-HEAD build; run via
`llama-bench` directly instead). If this comparison needs to be
authoritative later, either register a proper lane in
`config/harness-definitions.json` or re-run via `run_bench.py
--server-url` against a manually launched server so results land in the
harness's real result store.
