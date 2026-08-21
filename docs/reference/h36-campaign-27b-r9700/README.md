# HI35/HI36 GPU campaign — 27B dense + MTP on R9700 (gfx1201), b10502

Campaign artifact set from the 2026-08-21 pipeline on Brutus (R9700 gfx1201
32GB, GPU2 only). Consumed by HI35 (kernel-fraction ceiling) and HI36 (winner
generalisation: stability pair + holdout miss log). See HI36.md for the
GPT-adjudicated ship gates (req_09656b782a5d449a) these artifacts feed.

**Status: all stages complete (S1–S7). GPU2 released.**

## Workload identity

- Model: `/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
  (27B dense, Q8_0, 29.0 GB)
- MTP: `--spec-type draft-mtp --spec-draft-n-max 4`, same model as draft
- Server: `--ctx-size 8192 --batch-size 512 --ubatch-size 256 -ngl 99 -fa on
  --jinja --parallel 1 --no-webui` (miss-log run: ctx 16384, q8 KV)
- Bench: harness server-bench endpoint mode
  (`--bench-type server-bench --server-url http://127.0.0.1:<port> --model
  Qwen3.8-27B-Q8_0`), configs `default` (pp512 + tg128) rep 3 and
  `long-prompt-12k` (pp12288 + tg128) rep 1
- Vendor pin: b10502 = `0adcc3bb5710` (record build, tune build, replay build)
- **Tooling revision: `5020c68`** (the J: tree HEAD at campaign time; the
  tree was last synced before `b2d2d75`/`663e2c3` landed). `patches/*.py` is
  unchanged across `5020c68..HEAD`, so the vendor build is source-identical to
  what the current tip would produce. Consequences: S4 catalog was generated
  without HI73 skip logic (a few extra unreachable candidates — conservative),
  and S6 used the pre-HI67-slice-3 promotion CLI (run A promoted alone; the J:
  tooling rejects duplicate hypothesis identities from A+B combination).

## Artifacts

| file | producer stage | content |
| --- | --- | --- |
| `record-27b.jsonl` | S3 | 69-signature record-mode observations, header `source_revision=0adcc3bb`. 27B MTP workload. |
| `27b-inventory.json` / `.sqlite` | S3 | inventory: q8_0 only (mmq+mmvq), widths 1–5, no blas, no float |
| `tune-t1.jsonl.measurements.jsonl` | S5 run A | full tune run A: 59 results, `GGML_CUDA_DISABLE_GRAPHS=1`, b10502 tune build (workload-max catalog, 88 generated / 88 applicable) |
| `tune-t1.jsonl.journal.jsonl` | S5 run A | tuner journal for run A |
| `tune-t2.jsonl.measurements.jsonl` | S5 run B | stability pair: identical policy, 69 results (10 extra signatures vs A — MTP draft-width churn) |
| `tune-t2.jsonl.journal.jsonl` | S5 run B | tuner journal for run B |
| `t1-promoted.measurements.jsonl` | S6 | `tune-promote` over run A: 42 hypotheses → 40 promoted, 17 native, 2 confirmation-rejected |
| `dispatch-27b.cache` | S6 | v4 replay cache exported from the promoted file (59 entries, manifest `f3dc3027`) |
| `dispatch-27b-v5.cache` | HI74 (2026-08-21) | first real v5 replay artifact: the same promoted file re-exported with the v5 tooling (93-byte entries). 59/59 entries field-identical to the v4 cache; every entry's first 90 wire bytes match the v4 original (the v5 layout is append-only: `transform_id=0` + `match_kind=EXACT`) |
| `cov-baseline.json` | S7 baseline | same-workload replay coverage: **19077/19077 executed, 0 misses, 58 exact / 59 entries** |
| `cov-misslog.json` | S7 miss-log | 12k-prompt replay: 58 exact, **39 misses recorded** |
| `miss-misslog.jsonl` | S7 miss-log | the 39 uncached calls: 11 mmvq (width-3 decode), 26 blas (tiny native GEMMs), 2 mmvf |
| `pipeline.sh` | driver | final deployed stage driver S1–S7: stage-skip markers, GPU2 idle guard, atexit-flush poll fixes, ctx parameterisation for the 12k run |

Not in git (too large; on Brutus at
`/mnt/vault/development/llmhosts/bigcherry/artifacts/h36-pipeline/`):

| path | size | note |
| --- | --- | --- |
| `kf-prefill/brutus/*_kernel_trace.csv` | 13 MB | HI35 prefill trace. **Binary: stale 22dc605** — see below. |
| `kf-decode/brutus/*_kernel_trace.csv` | 508 MB | HI35 decode trace (p0 n256 r3). Stale 22dc605 binary — **validated by S1b** (see below). |
| `kf-decode-b10502/*_kernel_trace.csv` | 512 MB | S1b equivalence trace, b10502 record build, same command (p0 n256 r3). |

| `miss-s7b.jsonl` / `cov-s7b.json` | S7b | parallel-2 holdout: 18,173/18,173 executed, 28 exact, **80 misses** (38 mmq, 10 mmvq, 31 blas, 1 mmvf) — committed small artifacts |

## Results summary (2026-08-21)

**S5 stability pair (run A vs run B) — all ship gates PASS:**

| gate | measured | limit | verdict |
| --- | --- | --- | --- |
| max cross-run regret | 1.10% | ≤5.0% | PASS |
| p95 regret | 0.00% | ≤1.0% | PASS |
| call-weighted mean regret | 0.014% | ≤0.5% | PASS |

Winner-name agreement 54/59 (91.5%) — secondary metric only, per the GPT
adjudication. Regret is localized to two mmvq keys (`K=5120, M=10240/12288`);
mmq is 100% stable. Analysis: `tools/` script `tmp/h36-ab-stability.py`
(commit copy in this folder is not required — regenerate from the two
measurements files).

**S6 promotion:** run A alone through `tune-promote` (J: tooling rejects
A+B combination: "duplicate hypothesis identity"). 40 promoted / 42
hypotheses; 17 signatures keep native; 2 confirmation-rejected.

**S7 same-workload replay (ship gate: 100% coverage):** 19077/19077 calls
executed from the cache, 0 misses, 58 exact hits over 59 entries,
`rerun_required=0`, `stale=false`. **PASS.**

**S7 holdout miss-logs (two shape-perturbing workloads):** the long-prompt-12k
run gave 39 misses and the parallel-2 run (S7b) gave 80. All miss call-counts
are 1 (per-request bench; production multi-turn would repeat them).
what_if against run-A tuned winners with the offline safe keys
(mmq: family+types+K+M; mmvq: family+types+full ne0):

| family | misses | converted | note |
| --- | --- | --- | --- |
| mmvq | 11 | **11 (100%)** | width-3 decode signatures resolved by same-ne0 tuned siblings — the width-invariance the conditional GO relies on, confirmed on 27B |
| blas | 26 | 0 | **new finding:** 12k-context workload surfaces tiny native GEMMs (64×64 / 256×256) never seen at ctx 8192; not in record inventory (`uses_blas: false`). Reinforces the blas NO-GO; they stay native (zero risk) |
| mmvf | 2 | 0 | no mmvf candidates exist for this q8_0 workload (nothing to generalise from) — consistent with the NO-GO |

**S7b (parallel-2) what_if, same keys:** mmq 38/38 (100%) — batched-decode
signatures absent from the parallel-1 record, resolved by the K+M key;
mmvq 10/10 (100%); blas 0/31; mmvf 0/1.

**Combined holdout: 59 GO-family misses across both holdouts (11 mmvq + 38
mmq + 10 mmvq), all 59 converted — 100%. The unresolvable set is exactly the
NO-GO families (blas 57, mmvf 3).**

## S1b equivalence trace — STALE S1 TRACES VALIDATED (2026-08-21)

The kernel-fraction traces (S1) were captured with the pre-existing
`~/bc-build-record-4b` binary from `22dc605` (b10257), not b10502. The
original justification ("no kernel-identity changes in the pin window for
dense workloads") was **withdrawn**, and a b10502 decode equivalence trace
(`kf-decode-b10502`, `llama-bench -p 0 -n 256 -r 3`, the S2-rebuilt b10502
record build) was run and compared:

| check | stale (22dc605) | b10502 | verdict |
| --- | --- | --- | --- |
| total kernel dispatches | 1,410,908 | 1,410,908 | identical |
| gpu busy % | 77.74 | 77.78 | Δ0.04pp |
| mmvq fusion busy share | 63.988% | 63.943% | Δ0.05pp |
| mmvq small_k busy share | 30.205% | 30.229% | Δ0.02pp |
| all other kernels | — | — | each Δ ≤ 0.006pp |
| dominant matmul geometries | 32×8 wg, grids 557056/163840 (fusion), 327680/196608/163840 (small_k) | byte-identical | unchanged |
| **wall ceiling (mmvq_pct × gpu_busy)** | **73.224%** | **73.243%** | **Δ0.018pp ≤ 1.0pp → PASS** |

**Gate 1 as literally worded ("no new/missing kernel ≥0.5%") FAILS, and the
failure is real, not a measurement artifact:** the non-fusion `small_k`
instantiation gained a 7th template parameter (`rows_per_block_explicit`)
between pins — `mul_mat_vec_q<(ggml_type)8, 1, false, false, 0, 0>` (stale,
6 params) vs `...false, false, false, 0, 0>` (b10502, 7 params). The
fusion-path instantiation is unchanged. Busy share, dispatch count, and
dominant geometry of the mapped pair are indistinguishable, so the change is
a **kernel-identity rename, not a behavioural change**. Consequences:

1. **S1 traces are validated for HI35 ceiling purposes** (ceiling within
   0.018pp of its 1.0pp acceptance) — the provisional status is removed.
2. **Any kernel-name keyed cache/artifact from a stale-pin build is invalid
   for the small_k matmul path at b10502** (and vice versa). The replay v5
   format must treat mangled kernel names as pin-specific identifiers; this
   is recorded as an input to HI74 and the rebase review (A-list: pin moves
   invalidate name-keyed caches by construction).

Analysis script: `tmp/h35-s1b-equivalence.py` (gitignored; the inputs are
the two 512 MB CSVs on Brutus — `kf-decode/brutus/*` stale, `kf-decode-b10502/*`
b10502 — too large for git, checksums below).

## HI35 Part 2 — kernel-fraction ceiling bands, PUBLISHED (2026-08-21)

Full family-level `bigcherry kernel-fraction` report over the validated traces
(committed `kf/` copies; header-keyed parse, family attribution per standards
7.1). This is the adjudicated point-3 publication (ceiling_low = known matmul
share × busy; ceiling_high = (known + unmapped) × busy; band >2pp would mean
the method is not precise enough).

**Decode** (27B Q8_0 dense + built-in MTP, ctx 8192, R9700 gfx1201,
p0/n256/r3): matmul 95.3% of GPU kernel time (mmvq 94.2% + quantize_q8_1 1.1%),
unmapped 1.6%, GPU busy 77.8% (b10502) / 77.7% (stale):

| | ceiling_low | ceiling_high | band |
| --- | --- | --- | --- |
| decode (b10502) | **74.1%** | **75.4%** | **1.2pp ≤ 2.0pp → precise enough** |
| decode (stale 22dc605) | 74.0% | 75.3% | 1.2pp (pin-invariant within 0.1pp) |
| prefill | 39.5% | 47.7% | **8.2pp > 2.0pp → NOT precise enough** |

The prefill band fails the precision bar: 14.9% of prefill kernel time is
unmapped (the family pattern table does not cover the prefill kernel set), so
the prefill matmul share is understated and only a lower bound. Decode — the
phase the tuning items target — meets the bar.

The S1b table above quotes **73.2%**; that variant matched matmul kernels only
(`mul_mat_vec_q`, no `quantize_q8_1`). Standards 7.1 counts the activation
quantisation as part of the matmul path, so the 7.1-compliant lower bound is
74.1%. Both are the same measurement with a 1.1pp attribution difference; the
7.1-compliant band is the published one. The pin-invariance gate (Δ0.018pp)
holds identically for both variants.

**Consequence for prioritisation:** on this workload a 10% matmul saving is
worth at most 7.4–7.5% of decode wall before Amdahl's law meets the
CPU-bound remainder; the mmvq dominance (94.2% of kernel time) is exactly why
HI36a GO'd mmq/mmvq generalisation and the S7b batched-decode misses are the
highest-value coverage gap on the machine.

## Checksums (sha256)

```text
42cf1ec4fd4f32f5c3ce4ceda316e550aae81615fe273dd48daf5159aa22c3cb  27b-inventory.json
619d55a19f18e94e42a8c79969bfa9ab3d5a56a6969089fa9138847391c70efb  27b-inventory.sqlite
b715cac6b41ef20bbf561050a97d738f2a91ea043e7967ba5d564de821cb0b84  record-27b.jsonl
1e04d7d98035cf7e575dd81b0c84afedc40a99b6c6104b399b938303b1c7e549  tune-t1.jsonl.journal.jsonl
a24682b956c6bd216f11ff8fbec37144cecfce46374fd3612711f7462435bfcf  tune-t1.jsonl.measurements.jsonl
b95d0eb797a308b0793968a874c68b60cab90c1fa6827e576098e45e179bf97e  tune-t2.jsonl.journal.jsonl
536eaada3eb0afdd47fa68b30d5abc7a913e27a31134d4282df43b5ffa2d5f31  tune-t2.jsonl.measurements.jsonl
93d9228c14c5f3f944083fec584f6cea187988bba0072788d30f3eebd8d2b4dc  t1-promoted.measurements.jsonl
826aac996e7d3451c08c4f570705ce1d4001898e00c9a6209564bc5f07d1218f  dispatch-27b.cache
1ed60d4dae948423bd7a0eea2a95933bf693bcd34a9ca1c6cb5ebaad555adb6f  dispatch-27b-v5.cache
5469063a0b4046779a8a1c4a9e1e6167c80c817edeab8c08b270b50fce92b80a  cov-baseline.json
d6fc6067fb7332b4f9d6281824c465aae9dd849e7555f328a3d5c83d57b19c5a  cov-misslog.json
7ab84a0c7da246eedd26f56b98ae760881e1a1c63e40f44b736f90e3328e5382  miss-misslog.jsonl
c3ecf9053f1598b3e02439cdc7f16b486a911948793a0ad412f4f25e0d12af46  pipeline.sh
```

Large traces (too large for git; on Brutus at `kf-decode/` (stale 22dc605) and
`kf-decode-b10502/`, local copies in this folder's `kf/` which is
`git/info/exclude`d):

```text
66dacf5ba899a2c644101ed2b1342753c2baedcd541d63c9cc55145e79c4bfd8  kf/b10502-decode.csv
d18525c620518036b49c6864b65cbfac5bafc581546f2df6a0b02696c2b90978  kf/stale-decode.csv
b9fd1dd353d93b292ac6c7e0a0790ee06bbcaa11af58a1aeec3983abccf30977  kf/stale-prefill.csv
```
