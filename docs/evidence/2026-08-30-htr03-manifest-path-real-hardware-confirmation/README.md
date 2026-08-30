# HTR03: real-hardware confirmation of the manifest-driven corpus path

GPT's confirmation review (session `ses_330ae3c055084f38`) required one
narrow real-hardware replay before HTR03 could close: reuse the original
known-guilty HI141 cache and confirm the NEW manifest-driven
corpus/applicability path (`production-dual-xtx` -> `behavioral_classes`
-> corpus edition -> resolved vector -> `requires_mtp`) still reproduces
the exact original `hard_fail`, with the promised provenance fields all
present and correct.

## Result

`validate_htr03_manifest_path.txt` -- real run, real dual-XTX hardware,
the exact same known-guilty cache artifact used throughout the HI141/
HTR01 investigation (`dispatch.cache.provisional` from
`hi141-proof-20260829-2231`, verified beforehand to still contain the
guilty `mmvq:q8_0:w4:nw8:rpb1:sk0:v1` winner for the implicated dispatch).

- Corpus/profile resolution worked end-to-end through the real
  `config/recipes.toml`: `production-dual-xtx` -> `behavioral_classes=
  ('mtp-speculative',)` -> `behavioral_corpus_edition=
  qwen38-production-v1` -> resolved to exactly the one real vector
  (`hi141-mtp-4096-v1`), with `requires_mtp=True` correctly derived from
  the vector's own `requirements` (not the old server_args string-match).
- **native**: `draft_n=107, accepted=100`
- **candidate** (known-guilty cache): `draft_n=145, accepted=90`
- **verdict: `hard_fail`, first_output_divergence: 1**

This is an EXACT match to the original HI141 finding and every prior
real-hardware run of this same guilty cache throughout this
investigation -- the refactor changed only *how* the vector is selected
and gated, never the comparison itself, and this confirms that holds on
real hardware, not just in the byte-identical-vector-content check done
offline beforehand.

The emitted provenance record includes everything GPT's review required:
`behavioral_gate_contract_version`, `corpus_edition_id`/
`corpus_schema_version`/`corpus_content_digest`, `runtime_profile_name`/
`runtime_profile_digest`, and a full per-vector snapshot (static manifest
parameters AND the real comparison result: verdict, draft traces,
`first_output_divergence`, exact native/candidate generated-token
digests, token count) -- a future reviewer can reconstruct exactly what
this run checked and what happened, without re-deriving anything from
code as it exists at review time.

## Files

- `validate_htr03_manifest_path.py` -- the real-hardware driver script.
- `validate_htr03_manifest_path.txt` -- its real captured output.
