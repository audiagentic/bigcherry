# Gate E: install, verify, start, canary and stop of each radiance release (2020.8.1 to 2020.23.1)

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
| 2020.9.1 | 5079a795 | 216 s | 29 s | vocabulary-head tile native sums (IQR 6 streams -34%) and the uneven-ownership weighted-head package (untested, not selected by any recipe) |
| 2020.9.2 | 11146cb8 | 287 s | 28 s | IQR 16-row tile accumulate-once (native integer sums; drafter and 6/7-bit trellis forms); the flashnext start was 33% above the usual range while other builds and A/B runs shared the host |
| 2020.11.0 | 3b60a351 | 220 s | 25 s | IQR 16-row tile native fragments, the validated weighted-head ownership patch and the IQR-V4 three-card configs. 2020.10.0 was superseded before its bundle published and was not run. The flashnext start is back in the usual range on a quiet host |
| 2020.11.1 | e65c73a7 | 183 s | 25 s | IQR wide tile (17-64 rows) keeps its integer sums native per scale block |
| 2020.11.2 | 6246f8ad | 193 s | 24 s | host-facts records no longer carry the kernel builder's account and machine (engine owner-start snapshot) |
| 2020.11.3 | 7b0e8a96 | 292 s | 26 s | embedding gather butterflies, NT4 head prefetch, chained long 16-row tiles. The flashnext start overlapped heavy builds on the host (same pattern as 2020.9.2); serving against 2020.11.2 is level on every row |
| 2020.11.4 | 6fe8e142 | 231 s | 25 s | fused single-band IQR prefill tile (65-128 rows). Recorded with the repository's gate-e.sh tool |
| 2020.11.5 | 4082dda9 | 184 s | 28 s | the gfx1201 fused single-band IQR prefill tile as an untested package (not selected by any recipe): no behaviour change expected |
| 2020.12.0 | f56ca462 | 251 s | 29 s | the gfx1201 fused single-band IQR prefill tile promoted into the serving patch set |
| 2020.13.0 | d8caa2c1 | 195 s | 25 s | no kernel or engine change (the unit harness tool only) |
| 2020.14.0 | d8caa2c1 | 194 s | 25 s | patch ownership metadata and naming lint only; same archive as 2020.13.0 |
| 2020.14.1 | 86d08b45 | 248 s | 27 s | the fused single-band IQR tile as one arch-neutral copy (gfx1201 fused kernels renamed, records and output hashes identical); release build input tracking |
| 2020.15.0 | cacec67c | 186 s | 26 s | the embedding gather as the first owned kernel of the bc-iqr overlay plugin (libbciqr listed first for libr3 layouts); output hashes and resource records equal to the base row |
| 2020.16.0 | 9890c229 | 226 s | 25 s | the vocabulary head as the second owned unit of the bc-iqr overlay (libr3 still built a copy of the unit: its exclusion entry did not match, see 2020.17.0) |
| 2020.17.0 | bea7cb21 | 196 s | 26 s | the IQR GEMM family as the third owned unit (rad0065 archived); libr3 builds none of the three owned units, so both layouts ran their IQR embedding, vocabulary head and GEMM rows from libbciqr only; a cold release build |
| 2020.17.1 | 70e8068a | 237 s | 20 s | the owner writes the draft and measured manifests in-process (no writer child); helpers take typed settings on a sealed descriptor (no manifest); card selection through one mapping; shared-memory identity from the owner segment. A release build with the compiler cache took about 6 minutes |
| 2020.18.0 | cc31784b | 196 s | 20 s | the power profile is selected by x-bigcherry.power.profile from one power-profiles.yaml; --power-profile removed |
| 2020.19.0 | 90568ccb | 184 s | 21 s | power apply qualified from verified controller receipts, per-BDF site overrides, typed power plan straight to admission |
| 2020.20.0 | a1207813 | 245 s | 21 s | the managed native entry patch (rad0070) and the engine admission document, not yet composed |
| 2020.21.0 | 90734b28 | 196 s | 20 s | the first release that starts every engine on stock `radiance --config` with a sealed admission (rad0070 validated, rad0009 archived, manifest entry and managed front deleted). The first Gate E run of this archive failed in both layouts (never reached running, serve exit 1) and its logs were not kept; a rerun of the same archive passed; unexplained, now a failed run keeps its logs |
| 2020.21.1 | 20e7c4ea | 181 s | 20 s | the dead `--write-manifest` mode and `start --manifest` command removed; the engine's early host-room refusal is named and keeps the stored plan |
| 2020.22.0 | 3715e346 | 244 s | 21 s | the plan check reports a host-pressure mismatch as a room refusal (exit 4) and the owner keeps the stored plan; kernel qualification resolved from build groups |
| 2020.23.0 | 0637ce74 | 230 s | 21 s | a KL quality start scores the trunk with speculative depth 0; native qualification scope bound to physical evidence |
| 2020.23.1 | 60448148 | 231 s | 21 s | reversible RDNA4 shader offsets in both power owners |
| 2020.24.0 | 09b02984 | 196 s | 25 s | the first release build aborted in a heat-guard unit test (fixed by #1729); the rerun published |

## How to read this, and when Gate E need not be run again

- The archive id is the content id of the engine runtime payload. Two releases with the same id (2020.8.7 and 2020.9.0) ship byte-identical runtimes:
  the second needs no new start, canary and stop; its install and checksum can be compared with the first's instead.
- A superseded release is not re-run. Gate E runs on the newest release only, and only when its archive id differs from the last verified one.
- The flashnext-4gpu start time is dominated by the helper expert load (about 195 to 240 s); the qwen27b-tp3 start is about 24 s. A start time more than
  about 25% above these ranges on a new release is worth a look; none of the releases above showed one.
- The raw logs of these runs were kept in the task scratch only and are not reproduced here.
