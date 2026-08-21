# HI35/HI36 GPU campaign — 27B dense + MTP on R9700 (gfx1201), b10502

Campaign artifact set from the 2026-08-21 pipeline on Brutus (R9700 gfx1201 32GB,
GPU2 only). Consumed by HI35 (kernel-fraction ceiling) and HI36 (winner
generalisation: stability pair + holdout miss log). See HI36.md for the GPT-
adjudicated ship gates (req_09656b782a5d449a) these artifacts feed.

## Workload identity

- Model: `/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
  (27B dense, Q8_0, 29.0 GB)
- MTP: `--spec-type draft-mtp --spec-draft-n-max 4`, same model as draft
- Server: `--ctx-size 8192 --batch-size 512 --ubatch-size 256 -ngl 99 -fa on
  --jinja --parallel 1 --no-webui`
- Bench: harness server-bench endpoint mode (`--bench-type server-bench
  --server-url http://127.0.0.1:<port> --model Qwen3.8-27B-Q8_0`),
  configs `default` (pp512 + tg128) repetitions 3
- Vendor pin: b10502 = `0adcc3bb5710` (record build, tune build, replay build)

## Artifacts

| file | producer stage | content |
| --- | --- | --- |
| `record-27b.jsonl` | S3 | 69-signature record-mode observations, header `source_revision=0adcc3bb`, `manifest_hash=6fab7febbbdb6bf25b0a19b5e7710b5a` (the pre-existing shared-tree inventory-set catalog). 27B MTP workload. |
| `27b-inventory.json` | S3 | inventory: q8_0 only (mmq+mmvq), widths 1–5, no blas, no float |
| `27b-inventory.sqlite` | S3 | same inventory, DB form |
| `tune-t1.jsonl.measurements.jsonl` | S5 run 1 | full tune run A: 3-screen/15-final samples, `GGML_CUDA_DISABLE_GRAPHS=1`, b10502 tune build. Complete (flushed at server shutdown). |
| `tune-t1.jsonl.journal.jsonl` | S5 run 1 | tuner journal for run A |
| `pipeline.sh` | driver | full stage driver S1–S7 as deployed, including stage-skip markers, GPU2 idle guards, and the atexit-flush poll fix. Kept as the executable provenance of how every stage ran. |

Not in git (too large; on Brutus at
`/mnt/vault/development/llmhosts/bigcherry/artifacts/h36-pipeline/`):

| path | size | note |
| --- | --- | --- |
| `kf-prefill/brutus/*_kernel_trace.csv` | 13 MB | HI35 prefill trace. **Binary: stale 22dc605** — see below. |
| `kf-decode/brutus/*_kernel_trace.csv` | 508 MB | HI35 decode trace (p0 n256 r3). Same stale-binary caveat. |

## S1 stale-binary caveat (WITHDRAWN justification, pending equivalence trace)

The kernel-fraction traces were captured with the pre-existing
`~/bc-build-record-4b` binary from `22dc605` (b10257), not b10502. The original
justification ("no kernel-identity changes in the pin window for dense
workloads") is **withdrawn**: `22dc605 → 0adcc3bb` contains real ggml-cuda
changes (cpy.cu Q8_0 copy geometry #26731, mmvq.cu nwarps=8 for bs=1 #26843,
rope.cu, ggml-cuda.cu ×8). S1 is provisionally usable pending one b10502
decode equivalence trace (acceptance: no new/missing kernel ≥0.5% GPU busy,
no material launch-geometry change, ceiling within 1.0 pp of the stale mean).
See HI35.md 2026-08-21 note.

## Checksums (sha256)

```text
42cf1ec4fd4f32f5c3ce4ceda316e550aae81615fe273dd48daf5159aa22c3cb  27b-inventory.json
619d55a19f18e94e42a8c79969bfa9ab3d5a56a6969089fa9138847391c70efb  27b-inventory.sqlite
b715cac6b41ef20bbf561050a97d738f2a91ea043e7967ba5d564de821cb0b84  record-27b.jsonl
1e04d7d98035cf7e575dd81b0c84afedc40a99b6c6104b399b938303b1c7e549  tune-t1.jsonl.journal.jsonl
a24682b956c6bd216f11ff8fbec37144cecfce46374fd3612711f7462435bfcf  tune-t1.jsonl.measurements.jsonl
58472d1bd4e68380b65e36300377a2c7aa388d797ea3a4b2d587c030d598f09f  pipeline.sh
```
