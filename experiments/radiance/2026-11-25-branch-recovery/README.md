# Old branch recovery — 2026-11-25

Inert archive, not production or lab entrypoints.

| Branch | Exact tip | Disposition |
|---|---|---|
| feat/iq6r-v4-qualification | `1d1ef307952501f66d110ab7eaef37e0a01e8903` | Protected; retain unchanged. Newer converter lane owns consolidation. |
| agent/rum02-archive-superseded | `a8f3c6f3cab82d0e2cba3e7a2c9dc47c22a6ede2` | Five retired packages identical on main. |
| agent/rum11-e4m3-page16 | `5d9735720d5a4390b5e2acf762f451297d45cc0f` | Generalized store and tests shipped. |
| agent/model-identity-cache | `27153c157bafacf4cbf314e9f622b346a333fede` | Superseded by stat identity #1275; inert delta. |
| agent/rum05-planner-room | `78f6375a3867623478929b910c8ce05f0dbc3e60` | Measured resource fit and placement shipped. |
| agent/rum06-qwen27b-tp2 | `7082337c92e07e31a4dac59ce7aa09dd2340ca35` | MXFP4 unretirement rejected; IQ8R chosen; inert delta. |
| agent/rum09-warm-decode | `05d6ff1916c5280b1d5a1d6be95a1b056b3b1394` | Pool fit and trace shipped; recover evidence. |
| node/rum05-reuse-text | `3413905faf242e79e7c34f990bd2edb065f5a512` | Reuse shipped #1275, #1286, #1292, #1332; final residual review. |
| agent/rum06-clean-cancel | `338199f74ebba0155d665cfdd8818d6769227a7f` | Borrow/drain and clean cancel shipped #1272. |
| node/serving-control-socket-hardening | `2d15736b72f948e0d0dea88f2cc268d5bfe70382` | Shipped #1339, removed #1372; inert delta. |
| ci/rum21-ccache-publish-on-change | `c8c57bd0dffceafd392d10154e19498700155210` | Rejected #1361: cache expiry; inert delta. |
| node/ci-pin-action-shas | `3d896bbaf98c3f8232c5c029d152d4ed86f38f6e` | Rejected #1366: owner decision; inert deltas. |
| agent/rum15-compact-factors | `b8bc6532ea94e565738bfde310ca272f27a252fb` | Negative results; recover scratch prototypes. |

[Original code deltas](PATCHES.md) · [Scratch sources and receipts](SCRATCH.md). Do not delete branches until worktrees and full archive refs are verified. No new qualification implied.

## Evidence and resumption

- [RUM09 warm decode](../../../docs/evidence/radiance-latest-bump/RUM09-WARM-DECODE.md), historical pool and tier results; later warm-step evidence supersedes its open question.
- [RUM15 compact factors](../../../docs/evidence/2026-11-22-rum15-compact-factor-triage/README.md) and [K-parallel triage](../../../docs/evidence/2026-11-22-rum15-kpar-speed-triage/README.md): rejected on speed, not numerical correctness. Revisit only via the existing discrete unit harness or lab if justified.
- Scratch files are a selected recovery, not an exhaustive inventory. Original source and results remain untouched. Archive copies substitute local paths; no engine build or GPU run was performed.
