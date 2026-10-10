# Testing — NRO03 P2P AllReduce (archived)

**No current campaign.** Patch 1252 is rejected at b11474 after the 2026-10-07 lab fault. Do not rerun the historical Brutus dual-gfx1100 P2P arm without PNRO03's conditional re-entry gate. The stored receipt has `correctness=pass`, `activation=fail`, `contract passed=false`.

The standalone `tools/tests/hardware/test_p2p_copy_correctness.py` exercises `hipMemcpyDefault`, not `ggml_cuda_ar_allreduce_p2p_impl`. Its PASS does not qualify 1252.

**Only on a new supported topology/driver:** test directed peer capability/enablement and actual `p->dev_tmp`, `p->p2p_stream`, `p->ev_pool`, `p->p2p_done` with source-current async copies, threshold boundaries, concurrent streams and ring reuse. Require positive subject-only provider completion, multi-request/graph replay, no crashes, bit-identical full-vocab logits/tokens, then matched end-to-end against exact-F32 host/RCCL. On failure preserve rejected state and host fallback. No experiment is queued.
