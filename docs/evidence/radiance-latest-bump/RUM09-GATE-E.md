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

The installed bundle ships all 16 configs (10 Flash-Next, 6 Qwen 27B). Only `flashnext-4gpu` was started from the
installed prefix; the other configs were started from the identical build before release (see RUM09-GATE-D.md and the
plan note). Starting the full roster from the installed prefix and relocating the root are still to do before RUM09 closes.
