---
id: BCOP207
order: 207
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T10:06:00Z'
created-by: agent
priority: P2
work: S
---

# QFP07: admit physical two-head KV layouts before cross-rank QSA sharding

**Discovery/change:** Pinned b11552 `llama_meta_device_get_split_state` and validated 1303 round boundaries to whole KV heads. The observed two-head Qwen4Exp model cannot occupy all three ranks within one QSA layer; three nonzero weights can produce an empty rank. Rotating pairs can balance *across layers* but need model-load verification. `build_qsa_sel` produces global top-k and dense selection masks; 1303 supplies neither sequence-partitioned KV/mask nor partial softmax-state export.

**Owners:** QFP07 owns static attention/KV placement and conditional sequence-shard admission; 1303/QFN03 owns validated split implementation; QFP09 owns rank-skew evidence; QFP23/0910 owns startup profile identity; existing Meta collective/event infrastructure owns transport. No new selector, allocator, communicator or kernel authorized.

**Already acting/exclusions:** 1303 is validated; default `1,1,0;rotate=0` has historical +3%/+7% adoption evidence and 240K f16 fit. QFP07 last independent edit 2026-10-05, 1303 patch last tracked edit 2026-10-09; no relevant PR or queued experiment identified in the preceding 12h. Active RAD03 helper/mailbox/mask diagnostics, RAD04 two-half prefill and engine registry/release work remain protected. BCOP197-206 novelty checked; this is a distinct placement mechanism.

**Unresolved action:** QFP07 specifies a host split-map oracle, at most three distinct static ABBA candidates, per-rank critical-path/VRAM admission, and global-index-to-local-QSA-mask correctness before any sequence-shard implementation. vLLM's partial-output/LSE A2A requires `H % world_size == 0`; SGLang's KV ring requires P2P. Neither is a direct two-head/three-rank no-P2P port.

**Terminal:** `NO_DISTINCT_LAYOUT`, `NO_CAPACITY`, `NO_EXPOSED_SKEW`, `TRANSPORT_BOUND` or `NUMERIC_DRIFT` close the candidate. Static promotion needs CI95-low >=3% decode, <=2% representative regression and unchanged max context/correctness; sequence shard needs >=1 ms/token exposed skew and >=5% decode at two >=160K lanes with communication+combine <=50% of attention savings. Retain production 1303 otherwise.

**Validation:** Seven exact host layouts, synthetic 12-layer rotation and 375 integer-weight/rotation cases passed; Python compilation passed. No patch application, repository pytest, HIP build, hardware execution or new benchmark.
