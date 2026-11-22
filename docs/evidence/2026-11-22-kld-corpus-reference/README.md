# KLD corpus and reference provenance (corpus-4, flashnext tp2 exact)

The four-document KLD corpus and the provenance of the full-vocabulary reference recorded on it. The corpus itself is deliberately
NOT committed (see below); its identity and rebuild recipe are recorded here, and the reference logits are regenerated from it.

## Files

| File | What |
|---|---|
| `kld.json` | The reference manifest (format `radiance-kld-1`) with host paths replaced by `${BC_WORK_ROOT}` and `${BC_MODEL_ROOT}` |
| `reference.sha256` | sha256 of each reference file, paths made relative |
| `inputs.sha256` | sha256 of the radiance binary and corpus used, paths replaced by placeholders |
| `record.sh` | The exact recording command (paths parameterised) |

Because paths in `kld.json` and `reference.sha256` were rewritten, the first line of `reference.sha256` (the digest of the
original `kld.json`) does not match the sanitised `kld.json` here. The other digests describe the unchanged reference files.

## The corpus

The corpus lives at `${BC_WORK_ROOT}/corpus/rad-kld-corpus-4.jsonl` (24,874 bytes, 4 JSONL rows). It is deliberately not
committed: the documentation text inside it contains host-specific text (machine names and absolute host paths) that the
project forbids in committed files, and editing it would change its identity. Protect that work-root file and rebuild it
from the recipe below if it is lost.

- 4 documents, sources `kld-docs-00` to `kld-docs-03`, each prompt exactly 6000 characters of documentation text,
  `score_from` 0. Token counts 1951, 2212, 2133 and 1948 give 8,240 scored rows (first row of each document 0, 1950, 4161, 6293).
- It is the first four documents of the older 24-document corpus `rad-kld-corpus.jsonl` (42,761 rows).
- File sha256: `c335b9a3128dc94947c5d78c6380187431e554ab7462d4c5dcfd6e77641a125b` (recomputed when this was written: match).
- Corpus digest recorded by the engine in every reference `kld.json` and used to bind scoring to the corpus:
  `corpus_fnv1a64` `66bc5f8a27eda3e6`. Recomputing it here with plain FNV-1a 64 (standard offset basis) over the line bytes,
  with or without newlines, gave different values (`1149a165600fc190`, `dd16163de796c704`), so the engine's exact hashing
  input is not reproduced by that reading and the digest is carried as recorded. The `corpus` path in the recorded
  `kld.json` pointed at a copy with the same sha256 as this file, and the 2026-11-20 recovery matched the digest through
  the engine (RUM09). Treat the sha256 as the identity of this file and the FNV as the engine's binding.

### How it was rebuilt (recipe, if the file is ever lost)

No generator script exists. The corpus was recovered by decoding the token ids stored in the 24-document reference
(`tokens.i32` and its `kld.json`, work-root run `vtp44-flashnext-kld/ref-short`; the first four documents are rows 0 to 8239)
back to text:

1. Take the model tokenizer (`tokenizer.json`, byte-level BPE) and invert the GPT-2 byte-to-unicode map.
2. For document `i`, decode `tokens[off : off + docs[i].tokens]` using the per-document counts above.
3. Emit one object per document with keys in the order `prompt`, `score_from`, `source` using `json.dumps` defaults,
   join the lines with `\n` and end the file with a trailing `\n`.
4. Accept the result only if the engine's recorded digest and the sha256 above match.

## The reference

- Same model as the arms: `qwen3.8-next-flash-fp8-iq4r-moe` (Radiance `.rad`). It is not a BF16-weights model.
- Recorded tensor-parallel 2 with exact wire on the two 7900 XTX cards, no helper: `--kernels libr3,libr3moe,libref
  --placement expert_tiered --tp 2 --tp-wire exact --host-pool-mib 14336 --gpu-headroom-mib 320 --kv-cache-dtype fp8
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 64`, then `--kld-record` and `--kld-corpus`.
- Build: bc-radiance 1170.34.0 (Radiance 1.17.0); binary sha256 in `inputs.sha256`. Recorded 2026-11-20 in about 112 s of
  engine time (about 3 minutes with load).
- Contents: full-vocabulary fp16 log-probs `logp.f16` (4,092,313,600 bytes), `top.bin` (top 16), `rows.f32`, `tokens.i32`,
  `kld.json`; 8,240 rows, vocabulary 248,320. About 3.9 GB in total.
- The 3.9 GB of logits are NOT committed: they are pin-bound and cheap to regenerate. Scoring needs the full vocabulary,
  so the small files alone cannot score.
- Location: `${BC_WORK_ROOT}/references/kld/flashnext-tp2xtx-1170.34.0/reference` (with the original `record.log`,
  `exit-code` and `inputs.sha256` next to it).

### Regeneration

With a free `gpu` lease and no unleased GPU process, run `record.sh` (needs `BC_MODEL_ROOT`, `ROCM_PATH`, `FFMPEG_LIB`).
The run directory must exist and the reference directory inside it must not:

```
mkdir -p ${BC_WORK_ROOT}/references/kld/flashnext-tp2xtx-<pin>
agent-lease acquire gpu --ttl 30m --est 10m --purpose "KL reference record corpus-4" --wait -- \
  bash docs/evidence/2026-11-22-kld-corpus-reference/record.sh \
    ${BC_WORK_ROOT}/references/kld/flashnext-tp2xtx-<pin> <radiance-release-dir> ${BC_WORK_ROOT}/corpus/rad-kld-corpus-4.jsonl
```

`bc-serve quality --kld-record DIR --kld-corpus FILE` (PR #1365) was OPEN and not merged when this was written, so it is
not available on main yet; once merged it records the same reference through the native owner, and the campaign form
(`bigcherry.campaign.benchmark --server-config tools/lab/radiance/helper-kld.json`) records it when the reference
directory is absent. Before trusting a re-recording, confirm `corpus_fnv1a64` and 8,240 rows in the new `kld.json` and
compare document-0 rows with the existing reference's `rows.f32`.

## Caveat: the 1.16.0 reference is lost

The reference recorded on Radiance 1.16.0 on 2026-11-03 (the one the Nov 3 to 8 placement-policy KL numbers were scored
against) was deleted with its run and is not recoverable. This 1170.34.0 reference has the same model and layout but a
different pin, so absolute KL values differ from the Nov 3 to 8 numbers (for example policy-off p99 0.7036). Arm-versus-arm
deltas scored against one reference remain comparable; compare absolute values only within one reference.
