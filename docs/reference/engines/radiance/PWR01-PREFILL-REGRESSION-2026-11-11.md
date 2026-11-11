# Flash-Next PWR01 prefill gap: October–November card-free audit (2026-11-11)

**Scope and owner.** Research only: no GPU operations and no GPU lease. The lead owns hardware measurements. This document audits the `flashnext-2xtx` *single-request PWR01* long-prefill rate independently of undervolting. Do not compare the current four-card default with two-XTX runs as if they were one workload.

## Evidence-based verdict

- Valid October 27 stock `runs/pwr01-baseline-10271708/summary.txt`: **3312.3 prefill tok/s**, **93.5 decode tok/s**, 40-second phase, checkout `c37fe8c2` and build `67a435be11abe6053872355915df70a4/e315c3373c1f397eb16273556a7509ab`. The once-quoted `10271712` run is absent; `10271703` was invalidated by a cached prefill (`docs/reference/power-tuning-evidence-2026-11.md`).
- November 11 `runs/pwr01-baseline-11111104/{summary.txt,launch.log}`: **2963.8 prefill tok/s** (**−10.52%**), **98.9 decode tok/s** (**+5.78%**), 60-second phase, `flashnext-2xtx`, build `daf793f13be8d5762deecb73e3372641/41f15405cc38031fe6591674f460b8f1`, kernel 7.0.13-gfxoff; no CPU-package RAPL reading. It uses a patched copy of the PWR01 script under `runs/pwr-stock/pwr01-2xtx.sh` to pin the build and layout.
- An intermediate PWR01 on **October 29**, `runs/pwr01-baseline-10290922/summary.txt`, checkout `871e9b2` (Radiance v1.13-era recipe), measured **3336.6 prefill** and **93.1 decode tok/s**. This brackets the *observed change* after Oct 29, or to subsequent runtime settings/host changes; it does not identify a faulty commit. The Oct 29 power settings are not an assured stock control.
- **Verdict: an end-to-end serving-profile throughput gap is observed; an independent engine/software regression is *unproven*.** Chunk size, ownership, instrumentation, memory admission, clocks/power and host environment are confounded. In particular the single-stream decode prompt was modified after the October 27 stock run, so even its +5.8% is not a fully fixed-prompt A/B.

## PWR01 harness history: prompt and phase definition

`git log --follow -- tools/lab/radiance/pwr01-baseline.sh`: original PWR01 #435 (`9b955fea8`, Oct 27); #446 (`34d9de39e`, Oct 27, decode text identity); #450 (`6ee60222f`, Oct 28, lab SDK/device environment); #451 (`ce2769af9`, Oct 29, optional CPU RAPL); #574 (`dfb5e5139`, Nov 5, build logging).

The **prefill workload did not materially change**: 800 copies of “The quick brown fox jumps over the lazy dog.” and “Summarise.”, a unique leading `Request RANDOM...` sentence per request to defeat prefix caching, temperature 0 and one output token; rate is sum of `usage.prompt_tokens` divided by measured elapsed phase duration. The run loop completes its last HTTP request, so 40 vs 60 nominal seconds may include different fractional tails and warm-state effects. Original script default is 40 seconds; November passed `60`. In both versions, phases are idle-loaded, sequential 1024-token decode requests, sequential long prefills, idle-after. Both record GPU board `SOCKET_POWER` with `amd-smi metric -p`. The RAPL counter was mandatory on Oct 27 and optional subsequently; without it, November joules/token is GPU-only, but throughput counting is unchanged. Original script builds a selected engine, while November copy hardcodes the build and `--layout flashnext-2xtx`.

**Non-identical decode prompt:** original Oct 27 `run_phase` prepended a unique `Request ...` prefix on *all* prompts; PR #446 changed **decode only** to a fixed lighthouse-keeper prompt and began hashing text. Prefill retained the randomized prefix. The October 29 benchmark already had this fixed-decode implementation.

## Relevant code/configuration differences

| Mechanism | October launcher/build | November actual manifest/build | Commit or PR, priority |
|---|---|---|---|
| Prefill scheduling | `--max-num-batched-tokens 4096` | **2048**, checkpoint interval 4096 in both | **#465 / `23f73f67b` (Oct 31, RAD37)** moved chunk 4096→2048 for long-context. Highest-yield single setting test. RAD37's unrelated 4K-prefill measurements showed no loss, so causality still unproven. |
| Reserved context/sessions | max model length 32768, max sequences 6, no explicit prefix-resume mode | 262144 context, 8 sessions, prefix resume nearest, rope scaling none | #465 enlarged context and changed sessions; **#483 / `73a02f9ed`** later made 8 sessions default. More reserved state can affect VRAM/KV and scheduler despite one active request. |
| Engine speculative/MTP | 3 draft tokens | 3 draft tokens | **No default depth change.** #699 / `0976fa82d`, #702 / `57b87c093` restored draft/verify pass replay on v1.17 (~1.7–2% decode step); #809 exposed `--num-spec`, #822 left default 3. Plausible decode improvement, not proved prefill explanation. |
| KV and checkpoint | FP8 KV, checkpoint 4096 | FP8 KV, checkpoint 4096 | **Same explicit settings**; different pinned runtime algorithms remain possible. |
| Host pool/host KV | 22528 MiB host pool, no explicit host KV tier | **15507 MiB pool, 0 MiB host KV** from November derived ledger | #512 / `3933aad61` interim 9-GiB pool/12-GiB host KV work; **#821 / `e3298eea5` (RSC03)** derives host budgets at launch. Different host contention/admission and expert spill possible; not established by rate alone. |
| XTX expert placement | 22528 MiB **per helper**, heat source `runs/perf51/heat.rank0.hot-first` | **23074 MiB per helper**, source `heat/flashnext-mixed.hot-first`, owner digest `7a85c7d749...` | #477/#488 changed heat/layout choices, #821 derives per-device budget. Owner-map shifts could change remote misses and PCIe traffic on repeated prompts. Both heat files exist. |
| Heat recording | Not enabled by original launcher | **ON at 1-second cadence** in November `launch.log` | #500 / `7a4035827` moved costly old route histogram (previously 5–24% on a 42K prefill) to mapped pinned host memory and LDS counting; #502 enabled collection by default, #802 made interval 1 s. New method judged cheap, but matched PWR01 on/off still needed. |
| Live-placement policy | No production live placement | **OFF** on `flashnext-2xtx`, explicitly skipped for head/head with no tail helper | #693/#798/#830 change policy machinery, but the November log *rules out active live migration* as this run's cause; histogram recording remains on. |
| Radiance upstream pin | **v1.9.0** (`c37fe8c2` recipe; source record `685ec5b`) | **v1.17.0** (`19d4e8fe`) | **#451** 1.9→1.13; **#465** 1.13→1.14; **#555 / `61610f89f`** 1.14→1.16; **#627 / `d63ed2a88`** 1.16→1.17. Source/attention/GDN/graph/stream scheduling may differ, but no *identified distinct graph/stream option* appears in the two recorded launcher settings. Profile ENG01 before attributing cause to a kernel. |
| BigCherry patch set | Original recipes serving-core `rad0008_hc_upmix_rt_sweep` and handoff `rad0002` as experiment | Production serving-core includes `rad0008`, `rad0002`, `rad0011_plan_memory`, `rad0013_mtp_pass_replay`, `rad0009_serving_manifest` | #546 promoted handoff, #538/#663 memory-plan/manifest, #699/#702 MTP replay, #759 switched launcher to measured manifest, #818 typed R3 configuration. Strict new launcher cannot boot old Oct build (no bc-serve). Evaluate patch diffs only after settings are matched. |
| Helper kernels/CPU | Original RAD02-era helper; small/large lane behavior from its build | #509 packed W4 decode GEMV, #777 decode-sized fused gather/HC, route hist instrumentation, current large-lane code; PWR03 small-lane E-core automatic CPU pin | #509 and #777 chiefly target decode/TP, *not proven* as long-prefill slowdown. `moe-remote/component.toml` PWR03 warns pinning the *whole process* to E-cores previously slowed combine-prefill about 9%; current intended pin leaves large lane on all cores. Read process affinity and stage times before inferring a CPU problem. |
| Host/kernel/toolchain | Kernel 7.0.13, Oct Ubuntu GCC 14.2, ROCm 7.2.4 in historical report; Oct metadata lacked SDK digest | Kernel **7.0.13-gfxoff**, Debian GCC 14.2, ROCm 7.2.4 metadata digest `b811dea...` | Different kernel/GFXOFF and build environment, actual firmware/MES/PCIe link/clocks, power offset and thermal parity *not independently checked* in this audit. Power settings are a separate experiment; active clocks still need recording in hardware bisect. |
| Actual layout | R9700 TP1 engine, 2 XTX head helpers; no helper-row combine | Same named 2-head TP1 topology; **default when omitted is 4-GPU** with 6900XT tail/combine | #488 changed *default*. 4-GPU November ≈87.4 decode/2610.6 prefill is a different topology, not evidence of a regression on the 2-XTX layout. |

The old `docs/planning/FLASHNEXT_EXECUTION_STATUS.md` same-build natural **25K/14K/72K** prefill **3297/3053/2998 tok/s** on two XTX and `docs/archive/radiance/qwen3.8-flash-next-pre-v117-reference.md` v1.16 vs v1.17 **512-token** prompt cold prefill (~3493–3739 vs 3473–3733) are relevant context, **not** identical PWR01 prompt/phase controls. The v1.14 archival fresh-44K **3837 tok/s** similarly cannot be used as a direct Oct→Nov ratio. No retained PWR01 run after Oct 29 and before Nov 11 was found.

## Minimal lead-only hardware bisect

**Protocol:** use one short, exclusive GPU lease per arm on `flashnext-2xtx`, stock power readback, same device topology, same fixed PWR01 prefill and decode requests and 60-second phases. Stop engine/helper processes and release the lease before the next arm. Keep the same current build until flags are exhausted. Record **decode tok/s and prefill tok/s separately**, count/elapsed each phase, request completion hashes, admission key and manifest/digest (incl owner map, budgets, chunk, MTP, KV, contexts, recording), engine/helper per-stage times, clocks/watts/temperature/PCIe state during both phases, kernel/firmware/MES, and competing host jobs. A failed start/plan is not a throughput result. For any apparent >5% recovery do A/B/A, then repeated same-session arms to establish a noise band. These commands are for **the lead**, not executed here.

Start each arm with `bash tools/lab/radiance/launch-flashnext-best.sh start --build "$B" --layout flashnext-2xtx [arm flags]`, run the fixed PWR01 client-only phase against the server at port 36080, then `bash tools/lab/radiance/launch-flashnext-best.sh stop`. **Do not run the stock PWR01 script unchanged after a manual start:** it starts its own server. Adapt the lead's fixed-build PWR01 wrapper to accept a tightly controlled option list, without changing the client workload.

| Order | Arm flags on current November build | Hypothesis / acceptance |
|---|---|---|
| A0 | none | Repeat **2963.8 / 98.9** as same-session stock baseline. Heat recording on; live policy off. |
| B1 | `--chunk 4096` | Primary test: reverse October→November chunk change. If the 8-session/262K plan cannot admit this chunk, reduce sessions/context on **both** 2048 and 4096 arms before comparing. |
| B2 | `--record-heat off` | Test route-histogram instrumentation cost separately. |
| B3 | `--heat "${BC_WORK_ROOT}/runs/perf51/heat.rank0.hot-first"` | Test original heat/owner ranking with current budgets; log owner digest, remote misses/swaps. |
| B4a/B4b | `--max-num-seqs 6`, then *separately* `--max-model-len 32768` | Test each reserved-capacity variable, not both bundled. |
| B5a/B5b | `--budget gfx1100=22528`, then *separately* `--host-pool-mib 22528` | Test October helper budget and pool in isolation, under measured-plan admission; retain host KV0. |
| B6 (lower priority) | `--num-spec 0` | Decode acceptance/replay sanity, not a first-line long-prefill lever. |

For full October-equivalent recipe, only after isolated tests, combine 4096 chunk, 6 sessions, 32768 context, old expert/host budgets and old heat map in a labeled **composite** arm; do *not* interpret it as identifying an individual cause. Explicit `--record-heat off` for the October comparison. Never turn an unqualified configuration into a default based on one run.

### Existing build candidates with bc-serve (disk-verified, **not** runtime-validated)

Under `${BC_WORK_ROOT}/builds/` (each listed directory contains `radiance_home/bin/bc-serve`; dates from filesystem mtime, **not** proven build commit dates):

| Approximate date | Source slice / plan directory | Bisect use |
|---|---|---|
| Nov 2 | `b01e8fea3754c75f48c76eb66721f3fe/657e90b16df4c54c7cb83b44cfe1b347` | Early v1.14-era bridge, likely manifest incompatibilities |
| Nov 4 | `4d35a544686effb7cf60ce5d373143f6/5845ec3c5de25f916454c1a8e1ed4a12` | Around v1.16 pin |
| Nov 5 | `e0968731eb6da813f90edbf4f08a1fbc/00793ded12727d93cc60e66c00cc5637` | v1.16/1.17 transition |
| Nov 8 | `e54805824dddfa10235003a9fe764388/4d808005a1b97f0cb22767268012e480` | Earlier manifest-based serving |
| Nov 10 | `076dac8f2c767964eda8ba348c6755aa/8f1cfdcd7d9647976925f51cc3c6a472` | Near-current, before Nov 11 RSC03 derivation |
| Nov 11 (measured) | `daf793f13be8d5762deecb73e3372641/41f15405cc38031fe6591674f460b8f1` | Current control |

If current-build toggles do not explain the gap, prioritize **Nov 10 → Nov 8 → Nov 5** in a matched configuration and inspect each candidate's metadata/recipe/`bc-serve` schema before launch; then narrow by further build/commit. Nov 2/4 are optional earlier boundaries. **Do not assume old bc-serve matches today's typed serving v1/launcher**; use its matching checkout/launcher if needed, not a legacy-compatibility shim. The original Oct build `67a435be.../e315c337...` does **not** have bc-serve and was refused by the current launcher (`runs/pwr-stock/outold.log`). A period-correct October run would require the historical launcher and separate lead approval.

### Disposition / pass criteria

No-regression **PASS** only after matched current stock-power runs attain the Oct prefill rate within a measured noise band, with retained decode quality/throughput and named configuration tradeoffs. Independent-regression **FAIL** only after matched builds with fixed chunk, heat, ownership, memory, clocks/power/kernel and request definitions produce a repeatable prefill loss, bracketed to a source/pin change. If a loss remains under controlled settings, ENG01 owns rank-0 per-op/CPU gaps/helper-lane attribution; diagnose engine vs PCIe handoff before changing kernels. Current result: **unqualified observation, not a measured 10% source regression**.

**Evidence trail:** `tools/lab/radiance/{pwr01-baseline.sh,launch-flashnext-best.sh}`; historical `c37fe8c2:tools/lab/radiance/launch-flashnext-best.sh`; `engines/radiance/{layouts.toml,recipes.toml}`; `docs/reference/power-tuning-evidence-2026-11.md`; `docs/planning/FLASHNEXT_EXECUTION_STATUS.md`; `docs/planning/active/tuning-radiance-engine/ENG01.md`; `docs/planning/active/patching-radiance-heat-placement/RHP02.md`; `docs/archive/radiance/qwen3.8-flash-next-pre-v117-reference.md`; historical/current `runs/pwr01-baseline-*/summary.txt` and November `launch.log`; `runs/pwr-stock/{out2xtx.log,outold.log}`; both `bigcherry-build-metadata-radiance.json` files. No GPU/lease operations were performed.


## 2026-11-11 addendum: November 10 prefill ≥3600 FOUND; prior bisect ordering superseded

**Evidence correction.** A read-only scan of recent run directories found the user's Nov 10 report: `runs/policy-onoff-ab/on1.doc3.log` has **49,367 prompt tokens in 13.3 s = 3,708 tok/s**, and `off1.doc3.log` has **49,367 in 13.5 s = 3,667 tok/s**. The ABBA arm status reports 10 documents per arm, **roughly 3,612–3,708 tok/s** on all regular documents, and verifies policy ON/OFF/ OFF/ON from manifests. Specific examples: `on1.doc8.log` 3,651, `off1.warmdoc3.log` 3,654, `on1.warmdoc3.log` 3,685 tok/s. This was NOT PWR01's 800-repeated-sentence prompt and NOT the two-XTX layout.

**Primary target, exact build/layout:** `flashnext-4gpu`, R9700 TP1 engine, two 22,528 MiB RX 7900 XTX heads, RX 6900 XT 14,336 MiB tail with CPU helper-row combine; `builds/f17f21fda44979e9abe84f13e96b8896/b43f943466729a2eb484e082dc45069f`; Radiance 1.17.0, binary SHA-256 `df7d79b2542856f052f72925e25c59d9bb9ebba61cfbcf2cbd69b21bd49defcd`. `on1.engine.log` and frozen `flashnext/manifest-1110-172210-1957336.json`: **2048 chunk**, checkpoint 4096, MTP3, FP8 KV, max 8 sessions, 262,144 context, host pool **9216 MiB**, host KV/prefix cache **12288 MiB**, nearest prefix resume, RAM n-gram, heat file `heat/flashnext-mixed.hot-first`, 1-second histogram period; policy ON for on1/on2, OFF for off1/off2. The Nov 10 run used pre-RSC03 derived-budget defaults.

**Benchmark definition:** `runs/policy-onoff-ab/doc_prefill.py` takes arguments `18450 44000 K`, builds text from sorted **live** checkout `docs/**/*.md` files, selects approximately 171,600 characters starting at a K-dependent offset, prepends a doc-slice marker, sends a **single** completion request at temperature 0 with `max_tokens=1, ignore_eos=True`, and reports `usage.prompt_tokens / request wall time`. Effective requests are about 44K–57K tokens. It does NOT freeze or hash the doc text; changes in the checkout can change the slice even when K matches, and the exact Oct/Nov prompt payload cannot be proved identical. PWR01 instead sends back-to-back ~7K-token repeated sentences for 60 s and includes per-request/phase overhead. No comparison between the two rates is like-for-like.

**Power:** The saved Nov 10 document runs contain no `--gpu-power` opt-in; the launcher defaults to off. No voltage, power-control, clock or firmware readback is captured with the document-prefill measurements. **Actual power state is unverified**; do not assert it was equivalent to the PWR01 arm or use the policy A/B to infer a power effect.

### Other November 9–11 matches

All entries below are **document-prefill observations on the four-GPU layout**, never 2-XTX PWR01 matches. The build token identifies `source_slice_id / build_plan_id` under `builds/` (prefixes are abbreviated in this overview; exact full paths are in each folder's launcher logs). **These digests are not Git commit SHAs.**

| Date | Saved run folder | High document-prefill tok/s (observed range) | Build source / plan prefixes |
|---|---|---|---|
| Nov 9 | `rad49-hw1` | 3,624–3,728 | `ccb649e.../d7394b8...` (base 4GPU arm) |
| Nov 9 | `rad49-hw2` | 3,637 | `ccb649e.../b108955...` (base 4GPU arm) |
| Nov 9 | `rad49-4gpu-regress` | 3,626–3,722 | `ccb649e.../a4040b3...`; `f17f21f.../0dcf6a3...` |
| Nov 9 | `rad49-4gpu-coldtail` | 3,669–3,712 | `f17f21f.../0dcf6a3...` |
| Nov 10 | `rad49-4gpu-tailsweep` | 3,605–3,692 | `f17f21f.../0dcf6a3...` |
| Nov 10 | `rad49-4gpu-longfill` | 3,622–3,679 | `f17f21f.../0dcf6a3...` |
| Nov 10 | `rad49-tail-capture` | 3,715 | `f17f21f.../0dcf6a3...` |
| Nov 10 | `hl04-combine-trace` | 3,711 | `f17f21f.../77561c1...` |
| Nov 10 | `hl04-policy-on` | 3,604–3,705 | `f17f21f.../77561c1...` |
| Nov 10 | `histogram-period-default` | 3,678–3,741 | `f17f21f.../08b5dd8...`; heat period **60 s** |
| Nov 10 | `tk01-e2e-ab` | 3,600–3,685 | `f17f21f.../77561c1...`; `f17f21f.../c136e9d...` |
| Nov 10 | `tk04-e2e` | 3,618–3,707 | `f17f21f.../08b5dd8...`; `f17f21f.../c136e9d...` |
| **Nov 10** | **`policy-onoff-ab`** | **3,612–3,708** | **`f17f21f.../b43f943...`**, on/off policy A/B target |
| Nov 10 | `policy-accept` | 3,652–3,696 | `f17f21f.../b43f943...` |
| Nov 10 | `policy-lag` | 3,618–3,724 | `f17f21f.../b43f943...` |
| **Nov 11** | **`pwr-stock/docpf.out`** | **3,704–3,723** | **`daf793f.../a29e62a...`**, current source, policy OFF |

**Stronger same-day control:** Nov 11 `runs/pwr-stock/docpf.sh` used `builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3`, `flashnext-4gpu` **policy OFF**, six document-prefill requests at **3719, 3716, 3704, 3711, 3723, 3715 tok/s**. Its `docpf.launch.log` reports derived XTX budgets **23074 MiB** each, tail **14882 MiB**, host pool **1033 MiB**, host KV **12288 MiB**, heat recording every 1 s. Crucially, its Radiance **binary SHA-256 `60acedbbe20941221c09b5d94b0f542f078a8f6673f9a50ce3eee2548a60bb80` is identical to the November 11 PWR01 two-XTX build** `builds/daf793f13be8d5762deecb73e3372641/41f15405cc38031fe6591674f460b8f1` (different build plan; same engine binary). Hence **Nov 11 3,700+ doc prefill and 2,963.8 PWR01 prefill coexist on the same engine binary**; it is untenable to call their difference a general kernel prefill regression.

**Git commit provenance:** These run folders/metadata capture **source-slice digests, build-plan IDs and binary hashes but not a verifiable Git source commit SHA for each benchmark launch**. The linked reporting **commit `c9a8aced0` / PR #806** (merged Nov 10) documents the live-placement ABBA result *after the measurements*; it is **not** proof that this was the build source commit. `docs/planning/active/patching-radiance-heat-placement/RHP16.md` records the ABBA document-prefill effect (+0.6–1.1%) and `RHP17.md` the same run; `FLASHNEXT_EXECUTION_STATUS.md` has only the *older, different* natural 25/14/72K 2-XTX result (3297/3053/2998). `CURRENT_RELEASE_LEDGER.ndjson` stops at Nov 5, so it contains no Nov 9–11 3600+ event (although its Nov 4 policy record cites documentation prefill 3702→3687). No source commit should be invented.

### Corrected lead-only bisect order

1. **Use the Nov 10 run as the historical target**, not the October build: `f17f21fda44979e9abe84f13e96b8896/b43f943466729a2eb484e082dc45069f` on `flashnext-4gpu`, **2048 chunk, MTP3, FP8 KV, host pool 9216, host KV 12288, 22528/22528/14336 MiB helper budgets, 8 sessions, 262144 context, mixed heat, 1-s histogram**. Start with its documented **policy-OFF** arms (3612–3667), then the independently documented ON state (up to 3708). Use the saved manifests and compatible launcher rather than assuming current strict `bc-serve` can open all older plans. No GPU or power change in this audit.
2. **Match exactly the *document* workload across builds and options** on the same four-GPU layout. Existing current-binary `pwr-stock/docpf.out` already attains **3704–3723**, so a broad Nov10→Nov11 prefill regression is *not supported*. A formal performance comparison must freeze the doc payload with SHA-256, use the same heat/owner map, KV/host pool, chunk, policy and power readbacks, and report prefill/decode separately. Changing live docs makes K-specific doc-prefill runs nonidentical.
3. **Then do a same-workload PWR01 cross-build control**, first on this **same 4GPU** layout with the Nov 10 build versus the current build (same PWR01 prompt and 60-s phase), then only after that on **2-XTX vs 2-XTX**. This separates source change from 4GPU/2-XTX placement and from the very different PWR01 vs document workload.
4. **Only if a matched PWR01 decline persists**, test 2048↔4096 chunks, histogram off, heat ranking, host pool and XTX budget one variable per arm (the original audit's toggles). Record runtime settings, actual power/clock/firmware readbacks, fixed prompt checksum, build SHA/source-slice, owner-map digest and both prefill/decode rates independently. No conclusion from unrelated stream-bench decode throughput.

**Revised verdict:** The recalled November 10 3600+ figure **exists** and is a different workload/topology; a November 11 **3700+** record also exists on the **same Radiance executable** as the 2964 PWR01 record. There is **no demonstrated 3600→2964 like-for-like regression**. October→November PWR01 3312→2964 remains an observed but configuration-confounded gap, not a proven software regression.
