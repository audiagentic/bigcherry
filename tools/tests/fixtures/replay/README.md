# Replay compatibility fixtures

These two binary caches are immutable compatibility inputs for the replay
format tests. They were extracted from the HI35/HI36 campaign bundle
(`h36-campaign-27b-r9700`) so tests do not depend on a reference-document
directory that is being relocated to `artifacts/`.

| Fixture | Source | SHA-256 |
| --- | --- | --- |
| `dispatch-v4-h36-27b.cache` | `dispatch-27b.cache` | `826aac996e7d3451c08c4f570705ce1d4001898e00c9a6209564bc5f07d1218f` |
| `dispatch-v5-h36-27b.cache` | `dispatch-27b-v5.cache` | `1ed60d4dae948423bd7a0eea2a95933bf693bcd34a9ca1c6cb5ebaad555adbf6` |

The original files remain in the campaign artifact bundle for scientific
provenance. A missing fixture is a test failure, not an optional environment
condition.
