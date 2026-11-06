# RSC06: helper startup from a serving manifest

Archived from `feat/rsc06-helper-manifest` (tip `b67b32ea4`, base `a234b35f`); ten commits through 2026-11-02 including merges. PR **#527** was closed unmerged.

## What and why

The branch added helper-side strict configuration, removed helper-process `R3_*` getenv reads, added a helper-config plugin interface and tests, and carried `rad0010_helper_kernel_config` plus packaging fixes. It built on an earlier RSC01 schema.

## Outcome and why stopped

**(a): RSC01 loader foundation already shipped via #521. (b): original RSC06 helper/patch interface superseded by #633.** Main commit `9d5d0b808` (#633) delivers RSC06 under the revised v1 contract: a shared strict reader, `--manifest FILE --role helperN`, and typed projection, not this branch's independent kernel-config interface. No independently useful class-(c) change identified. No patch is archived. The contract was replaced; no hardware A/B result for #527 was recorded.

## Evidence and resumption

- PR #527 (closed), PR #633 (merged), main commit `9d5d0b808`; PR #521 (RSC01).
- `docs/planning/active/run-serving-configuration/RSC06.md`, especially its 2026-11-05 amended-contract note; `docs/releases/CURRENT_RELEASE_LEDGER.ndjson`.
- Original commits `817980537`, `69b380232`, `5270c0fd3`.
- Resume against today's `serving/v1` schema and typed loader; do not restore the old `rad0010` design or getenv fallback.
