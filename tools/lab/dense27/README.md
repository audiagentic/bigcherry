# Dense 27B mixed-GPU placement lab

Card-free, non-serving, experimental. It does not alter Radiance, llama.cpp,
model quants, launch layouts or the pinned vendor engine. Hybrid TP2 + R9700 is
a hypothetical graph, **not an available serving layout**.

## What exists

- Radiance Qwen3.8-27B MXFP4: qwen27b-2xtx, dual XTX TP2, wht6 host wire,
  no P2P, FP8 KV (engines/radiance/layouts.toml).
- llama.cpp Qwen3.8-27B Q8_0: tierL-qwen27b-q8, 29,047,086,048 bytes,
  tensor-2 (config/models.toml).
- Reuse KAB01 for dense kernel A/B, RAD48 and RR10-AR for XTX GEMMs and
  collectives, RAD49 for existing TP2 engine work.
- RAD50 is currently in the separate exp/rad42-cpu-bar-push worktree, not
  this branch's base. RAD50/RAD51 prove or propose *remote-MoE result*
  transport, **not** CPU scheduling of dense layers. Any reuse requires
  an independent activation handoff protocol and measured critical path.
- Qwen4-27B has no verified released checkpoint/model card in this branch.
  Qwen3.8-27B is a proxy, never a substitute for release-specific validation.

## Card-free experiment

From the repository root run:

    python3 tools/lab/dense27/placement_model.py --input tools/lab/dense27/synthetic-example.json --topology layer3 --objective decode_pipeline_interval --allow-synthetic
    python3 tools/lab/dense27/placement_model.py --input tools/lab/dense27/synthetic-example.json --topology hybrid_tp2_r9700 --objective decode_latency --allow-synthetic
    pytest -q tools/tests/lab/test_dense27_placement.py

The example is SYNTHETIC and requires explicit opt-in. Never cite its stage
times, fractions, cuts or budgets as hardware evidence.

The tool enumerates contiguous layer cuts and rejects overcommitted VRAM,
missing group costs/links, overlapping stage devices, and incomplete memory
accounting. Its decode/prefill latency sums stage compute + crossings for
one request. Its pipeline interval is an *optimistic lower bound* (max of
stage costs and transfer stages), not a promised throughput result.

## Measured input contract

Copy the example JSON to an external run artifact, set data_kind=measured
and provide evidence_id identifying the captured run. Replace ALL synthetic
numbers from actual profiles and memory reports:

1. devices.*.budget_mib = allocatable VRAM after driver, runtime, fixed
   graph/safety and other unmodeled per-device reserves.
2. layer_pattern = actual ordered layers, each weights_mib, kv_mib,
   state_mib plus decode_us/prefill_us measured for each group. Explicit
   layer entries are preferred if layer shapes vary; Qwen4 cannot assume
   Qwen3.8's 3-GDN/1-attention architecture.
3. groups.*.fractions = measured per-device resident fractions for each
   memory class; factor in replicated metadata separately. In particular
   TP2 must not be assumed to use exactly half VRAM.
4. transfer_us = real decode and prefill activation handoff cost, including
   copies, completion/publication and wakeup, at recorded payload widths.
   A MoE result-plane benchmark is NOT a dense activation measurement.
5. topologies = ordered, disjoint execution groups. Hybrid requires graph
   features that DO NOT exist yet. No arbitrary interleaving or dynamic
   placement is modeled.

Results are placement hypotheses only, not a serving configuration or an
end-to-end quality/performance qualification.

## KV arithmetic

For conventional full-attention cache, raw bytes are:

    2 * context_tokens * attention_layers * kv_heads * head_dim * dtype_bytes

Qwen3.8-27B has 16 full-attention layers, 4 KV heads and head dim 256.
At 32768 context FP16: ~2 GiB logical full-attention KV, ~1 GiB per rank
with even TP2 KV-head partition. Three-way even sharding of four KV heads
is impossible; don't assume TP3 without explicit uneven/replicated handling.
This excludes recurrent GDN state, speculative decoding, sparse indexer
state, session/slot duplication, padding and allocator arenas. Measured
per-card allocation supersedes these arithmetic estimates.
Current llama.cpp tensor split can reject quantized KV; do not assume
Radiance FP8 KV support means it is supported for llama.cpp -sm tensor.

## Stages and gates

1. **Card-free:** inspect source/model/PCI topology; run tests and calculate
   conservative per-device memory budgets. Do not edit vendor sources.
2. **GPU lease, controls:** same model/quant on each XTX, R9700, TP2 XTX;
   8K/32K/64K contexts where capacity allows; decode M=1/2/4/8, MTP,
   prefill 512/2048/8192 and concurrency 1/4/8. Record correctness first.
   Use KAB01's kernel A/B before any GPU kernel refactor.
3. **Layer canary:** pin and test llama.cpp -sm layer loading. Its prior
   27B validation reports a non-tensor vector::_M_range_check failure for
   one build/workload; do not claim it works until a fresh isolated canary.
   Verify model continuation and KV-device accounting.
4. **Transport harness:** standalone producer/consumer for decode/pre-fill
   activations, bounded ticket/epoch ownership and exact equality; compare
   pinned-host relay, CPU push and verified P2P if supported. Validate
   stream and publication ordering before throughput. Do not import
   RAD50's BAR protocol without a fresh proof.
5. **Hybrid review:** XTX TP2 stages + whole R9700 layers require new
   graph-stage ownership, TP2-to-TP1 residual conversion, scheduling,
   independent KV/GDN state and a physical handoff. Build only opt-in
   after review and positive critical-path budget.
6. **Promotion:** same checkpoint and quant across arms, KL/long-context
   checks, deterministic continuation, ABBA results, p50/p95/p99
   latency, per-card peak VRAM and measured copy/reduce overhead.
   Incomplete or failed evidence => no production changes.

## Quantization

Use high-quality Q8_0 as reference, then compare available Q6_K,
Q5_K_M and Q4_K_M conversions from the SAME checkpoint, each with a
quality gate. Treat Radiance MXFP4 and llama.cpp GGUF as DIFFERENT
lanes (not direct speed/quality A/B). A 27B FP16 weight-only lower bound
is ~54GB decimal; the registered Q8_0 GGUF is ~29.05GB decimal BEFORE
KV/graph costs. A 32GB R9700 may not safely fit Q8 with long context.
No KV quantization in llama.cpp -sm tensor until explicitly supported.

No hardware run without an exclusive agent-lease. Keep local run IDs,
GPU PCI identities and captured reports out of git; retain only the
run recipe and evidence summary.
