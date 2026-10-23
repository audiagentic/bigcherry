# 1315_mtp_draft_trace

**Status:** validated
**Plan item:** QFP15

Default-off trace of sampled draft token/probability, draft hidden-row hash and post-accept target hidden-row hash. The first differing step identifies the first observed divergence, **not** whether the target or draft caused it: the initial pre-draft target seed is not recorded and accept-row selection depends on accepted count. The FNV-1a helper uses nonstandard offset 1469598103934665603; old hashes remain comparable within the same implementation but are not canonical FNV-1a. Do not infer a GPU race or numerical identity from hashes alone.

Historical qualification: QFP18 lightweight tier, b11474, dev 2026-10-08, build deploy-v4-diag; 92 DRAFT_TRACE, 23 SPEC_TIMING, 23 DRAFT_TIMING, 0 error lines; greedy text matched production (md5 a34a34c1f48d). This proves activation and observed neutrality, not causal attribution, b11552 qualification or a throughput gain.
