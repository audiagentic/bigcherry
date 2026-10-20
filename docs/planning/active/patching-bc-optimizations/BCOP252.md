---
id: BCOP252
order: 252
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-16T08:06:47Z'
created-by: agent
priority: P2
work: S
---

# QFP29: fence ROCmI4 W4A4 MTP on executed verify width

**Discovery:** The RDNA4 fork documents W8A8/MMVQ for <=8 rows and IU4 W4A4 for larger MMQ batches, whereas BigCherry QFP16 records single-request four-token verification (<=5 target rows). The previous QFP29 premise that MTP inherently activates W4A4 is therefore not established. The latest fork selector source could not be retrieved; require a real kernel witness. This is a source/documentation admission correction, not an observed GPU bug.

**Owner/consolidation:** QFP29 owns ROCmI4/IU4 qualification. QFP16/FMTP owns verify-width receipts, QFP05 MTP policy, QFP17 prefill, QFP23 profiles, MET01/RPL01 residency; standard 1307-1313 quant/MMQ remains unchanged. No new scheduler, dispatcher, allocator, cache, hardware queue or active neighbouring plan was modified.

**Eligibility/novelty:** `main` selected at independent `08441aa4` (MEN10/RAD09, 2026-10-15 05:21 UTC). Last QFP29 independent `8b93942` 2026-10-06; no relevant commits/PR updates found in 12h GitHub window after 2026-10-15 20:06 UTC. Excluded Radiance/RHP expert-helper work, BCOP241 placement and recently reviewed BCOP242-251 capabilities; external hardware queue status unavailable.

**Unresolved action / terminal disposition:** Inspect fork MMVQ/MMQ selector at exact SHA and reuse existing terminal R9700 traces for actual target row counts, quant and IU4 symbol. No trace -> `NO_ROUTE_EVIDENCE`; <=8/no IU4 -> `NO_VERIFY_W4A4_ROUTE`, close MTP port with zero theoretical removable W4A4 verify cost. Do not pad/inflate work. Only witnessed IU4 MTP permits same-GGUF W4A4 off/on numerical/KLD/top-1/tail/acceptance ABBA, >=10% verify-wall and >=5% E2E improvement, <=2% long-context regression, protected ssm_out and no host spill. Else `QUALITY_FAIL`, `NO_E2E_GAIN` or `NO_PORT`. Prefill-only qualification is separate and cannot borrow MTP claims.

**Sources:** [RDNA4 fork](https://github.com/OwNhj/llama.cpp-RDNA4-FP/blob/master/README.md), [upstream #29953 MMQ diff](https://github.com/ggml-org/llama.cpp/pull/29953), QFP16 and QFP29. External same-file PPL 4.6937 off vs 5.4493 on is not BigCherry measurement. Host-only 8/8 unittest and Python compile PASS; no BigCherry pytest, HIP/C++ build, GGUF load, GPU execution or new benchmark.
