# LTX03 native three-rank cross-modal attention declaration

**2026-11-22: host-only PASS; LTX TP3 serving remains refused.**

## Scope

The existing Radiance LTX attn_declare and C02 native real-checkpoint test now cover cross-modal a2v and v2a for all 48 DiT blocks, both legal 8/8/16 and 12/12/8 head splits and three ranks: **576 additional source-backed cross-modal declarations**. Together with earlier self and text-cross coverage the existing suite now verifies 1,728 declarations across all six attention kinds.

The test reads each Q, KV, output, gain and gate geometry from actual model entries, including different Q, KV and output widths. Its native Q and KV shard partitions use one and two IQ8R banks respectively, the same full-head normalization, native head offsets, RoPE and once-only output bias. Query and KV row counts differ (4 and 2). No new engine, collective, kernel, loader or graph is added.

## Red to green and checks

1. The original TP3 geometry validation returned RAD_E_UNSUPPORTED for unequal Q/KV/output channel widths. The existing full-width self restriction remains; non-self now validates positive channel dimensions and 256-group-aligned source K dimensions from the declared model encoding.
2. The new source-anchored rad0060 patch is derived from the same original pinned Radiance 2.2 source: **23 edits apply**, the replay is idempotent and its generated C++/header are byte-identical to the tested staged source.
3. Nice-19, single-worker native host C01/C02/C05 build: PASS.
4. Native CTest suite: **13/13 PASS**, including all six attention-kind geometry checks; the existing host issued C06 video/audio self-attention continues to pass.

This is declaration proof only. It does NOT issue cross-modal operators on GPUs, demonstrate parity with TP1, execute FFN/residuals, or generate a TP3 video. Those remain LTX03, LTX02, RUM14 and LTX04 acceptance gates.
