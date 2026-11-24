# Gate E: install, verify, start, canary and stop of each radiance release (2020.8.1 to 2020.9.0)

Gate E takes a published release, downloads its archive, checksum and receipt, installs it into an isolated root with the ROCm SDK (7.2.4),
verifies the root, then starts, canaries and stops two production layouts under a short GPU lease: `flashnext-4gpu` and `qwen27b-tp3`.
The canary is a temperature-0 completion of "The capital of France is" for eight tokens; the expected text begins " Paris". Every release below passed
all of it: the archive checksum matched, `bc-release install` and `bc-release verify` both reported `verified`, both layouts reached running, both
canaries answered, and both stopped cleanly (stop and serve exit codes 0).

| Release | Archive id | flashnext-4gpu to running | qwen27b-tp3 to running | Contents (kernel and engine changes) |
|---|---|---:|---:|---|
| 2020.8.1 | 98d7b6cf | 207 s | 24 s | IQ8R K-parallel spill fix |
| 2020.8.2 | 8e55578d | 195 s | 25 s | K-parallel accumulate-once |
| 2020.8.3 | 973b93c4 | 240 s | 24 s | drafter 4-bit row as 16-token launches |
| 2020.8.4 | e013b53b | 199 s | 25 s | logits head native sums |
| 2020.8.5 | 86ccad7f | 226 s | 24 s | K-parallel look-ahead rule |
| 2020.8.6 | 73d1bfed | 222 s | 25 s | decode attention native sums |
| 2020.8.7 | a80f9f8f | 196 s | 24 s | serving-test race fix (no kernel change) |
| 2020.9.0 | a80f9f8f | 210 s | 24 s | head-builder package, not selected by any recipe (payload identical to 2020.8.7) |

## How to read this, and when Gate E need not be run again

- The archive id is the content id of the engine runtime payload. Two releases with the same id (2020.8.7 and 2020.9.0) ship byte-identical runtimes:
  the second needs no new start, canary and stop; its install and checksum can be compared with the first's instead.
- A superseded release is not re-run. Gate E runs on the newest release only, and only when its archive id differs from the last verified one.
- The flashnext-4gpu start time is dominated by the helper expert load (about 195 to 240 s); the qwen27b-tp3 start is about 24 s. A start time more than
  about 25% above these ranges on a new release is worth a look; none of the releases above showed one.
- The raw logs of these runs were kept in the task scratch only and are not reproduced here.
