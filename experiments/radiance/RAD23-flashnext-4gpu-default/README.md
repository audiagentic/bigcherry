# RAD23: Flash-Next four-card launcher default

Archived from `feat/flashnext-4gpu-default` (tip `c33cf7683`, historical base `a40258234`), with branch-specific commits `88545dcd8` and `c33cf7683` dated 2026-11-01.

## What and why

Make the layout with one engine, two XTX expert helpers and an RDNA2 tail helper the launcher default, using the mixed workload heat profile. Record comparison with two XTX helpers alone.

## Outcome and why stopped

**(a): runtime default, heat configuration and layout already on main via #488 (`5076e78d2`) and subsequent launcher work; (a): measured model-page findings already represented by the main RAD23/RHP10 record. (c): none.** The current launcher still defaults to `flashnext-4gpu` and `heat/flashnext-mixed.hot-first`. The historical branch-specific model-page section is not reapplied wholesale: main's model page and serving contract have continued to evolve. It was independently merged through #488; no patch.

Historical A/B recorded by `c33cf7683`: eight sessions / 1.04M prompt tokens, 1,353 → 2,003 tok/s (+48%) and engine swap traffic 3,582 → 1,279 GiB. At 675K code tokens, 1,842 → 2,351 tok/s (+27%); at 688K docs tokens, 1,845 → 1,902 (+3%). KLD passed with recorded 14 GiB tail heat (`rad23-kld-t14b`, `rad23-kld-mixed`). The old numbers are preserved here as historic evidence, not an assertion about current builds.

## Evidence and resumption

- PR #488 and main commit `5076e78d2`; branch commits `88545dcd8`, `c33cf7683`.
- `engines/radiance/layouts.toml`, `tools/lab/radiance/launch-flashnext-best.sh`, `docs/reference/engines/radiance/models/qwen3.8-flash-next.md` and `docs/planning/active/patching-radiance-rdna3/RAD23.md`.
- Resume with explicit layout/heat selection and revalidate against current serving manifests; don't restore older launcher defaults or claim the 2026-11-01 A/B applies unchanged today.
