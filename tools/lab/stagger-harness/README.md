# Staggered half-batch harnesses (QFP49)

Small proofs for the cross-card sum work, built up before the engine: each rung adds one thing.

| Program | What it runs | Needs |
|---|---|---|
| `../hip-probes/rccl_stagger_probe.cpp` | The stagger schedule with raw kernels and `ncclAllReduce` | ROCm only |
| `meta_pair_harness.cpp` (`run.sh`) | A toy 96-layer model on the real tensor-split (Meta) backend: two halves in sequence, as a pair, in the wrong order, and as one full batch; compared bit for bit | A build with the pair entry of patch 1361 (`ggml_backend_sched_graph_compute_async_pair`); not on main - PR #147 |
| `llama_prefill_harness.cpp` (`run-llama.sh`) | Prefill of the real model through the llama library alone: no server, drafter or sampling. One process an arm, last-token logits compared bitwise, per-call host time, optional kernel trace | Any build |

`run-llama.sh` is the acceptance test for anything that changes how prompt batches are submitted:

    ARMS='seq: ; new:SOME_FLAG=1' SRC=<source tree of the build> OT='^per_layer_token_embd\.weight$=CPU' \
      VIS=0,1,2 SCRIPT lh1 tools/lab/stagger-harness/run-llama.sh @<build run> <out-dir> -n 4096 -b 512 -ub 256 -c 65536

With `TRACE=1` every arm runs under rocprofv3 and `../flash-next/kernel-overlap-stats.py` reports, per card, how the
prefill divides into compute only / sum only / sum + compute / idle and which stream each kernel ran on. A change
that claims to overlap the sum with compute has to show it in `sum + compute`; a sequential run shows 0.0%.

Read speeds from several interleaved repeats: with a second stream in play single runs fall into a fast and a slow
mode (QFP49, 2026-10-11).

Rules the Meta backend imposes on a standalone program, found the hard way and written as comments in
`meta_pair_harness.cpp`: reserve every scheduler with a graph of its own before allocating any graph; copy results
out of the per-device compute arena inside the graph; keep caches out of a weights buffer.

Results: `docs/planning/active/patching-qwen-flash-next/QFP49.md`.
