# RUM09 Gate E: installed release bc-radiance-1170.37.0

Release bundle workflow for the tag finished successfully and published `bigcherryrad-<plan>.tar.gz`, its `.sha256` and
the tag-bound `.receipt.json`.

| step | result |
|---|---|
| archive checksum against the published `.sha256` | OK |
| `bc-release install --tag bc-radiance-1170.37.0` (receipt from the same release, isolated root, ROCm 7.2.4) | verified |
| `bc-release verify` / `status` | verified; current `bc-radiance-1170.37.0`, previous none |
| install with the ROCm 10.1 SDK | refused: `BC_RELEASE_9` SDK version mismatch (the receipt pins 7.2.4), as designed |
| installed `bc-serve serve flashnext-4gpu` (named, from the installed prefix, no `--set`, no explicit native config), gpu lease | up; canary `" Paris. The capital of Germany is Berlin. The capital of"`; stop clean, exit 0 |

## Full roster from the installed prefix

All 16 shipped configs (10 Flash-Next, 6 Qwen 27B) were started by name from the installed prefix, one short gpu lease
each, with no `--set` and no explicit native config: every one reached `up`, answered the canary with the expected
completion and stopped cleanly (exit 0). That includes the mixed-rank TP3 Qwen configs and the helper-offload Flash-Next
configs.

## Relocation

Moving an installed root is refused by design, by both `bc-release verify` and `bc-serve`: `BC_RELEASE_4: stored install
prefix changed`, because the install receipt binds the absolute prefix. The supported way to place a release elsewhere is
a fresh `bc-release install` of the same tag, archive and receipt at the new root. That install verified, and
`bc-serve serve qwen27b-tp3` from it reached `up`, answered the canary and stopped cleanly.

Explicitly unsupported: relocating an installed root by moving or copying it.
