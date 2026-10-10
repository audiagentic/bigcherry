---
id: QFP49
order: 49
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T13:54:01.542976+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P1
work: L
---

# Prefill cross-card sum: overlap the transfer with compute, narrower wire, rank balance

## Description

The large (RCCL) AllReduce is the largest single cost in Flash-Next prefill and nothing is hiding it.

Measured (run dp1, main at 1042e84d + lock counters, production profile, tools/lab/flash-next/ar-prefill-stats.py over the prefill window without model load; the same at 24K and 98K), per target card: computing 44-49% of the time; in ncclDevKernel_Generic_4 32-37% (10.0 / 9.7 / 11.0 s of a 29.6 s prefill at 24K); idle with no kernel queued 17-19%. There are 96 sums per 512-token batch (7,296 at 24K), about 1.4 ms each on the card. Per sum: the cards arrive 1.3 ms apart on average (median 0.9; the R9700 is usually last), and the exchange after the last card arrives takes 0.9 ms (6.6 s of the 29.6 s). So of a card's time in the sum kernel about two thirds is the exchange through host memory (no peer-to-peer between these cards) and one third is waiting for a slower card.

Already tried and rejected: sending large sums through the chunked cpu-root path (1291, BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=64 MiB): 1,339 -> 1,048 t/s at 24K with 1 MiB chunks and 883 with 8 MiB (runs ar1, ar2); decode 89 -> 77 t/s. RCCL is the better of the two existing paths.

Owner, 2026-10-11: "can we copy from or to the card while it is busy so we parallel the work and keep them busy". Yes: a transfer on its own stream runs beside kernels on the compute stream. What is missing is independent work to run meanwhile: within one batch the next op needs the sum.

## Steps

1. Message size and PCIe rate: bytes per sum (n_tokens x n_embd x 4) against the 0.9 ms exchange, per link (x8 / x8 / x4). States how far the exchange is from the link limit.
2. Asynchronous transfer: the exchange on a transfer stream with the consuming kernel waiting on its event, not a kernel spinning on the compute stream. Shows how much of the 32-37% is pure waiting.
3. Half-batch staggering in the Meta split backend (TokenWeave, arXiv 2505.11329): split a prefill batch in two; while half A's sum is in transit the cards compute half B's part of the layer. Default-off flag; attention ordering within a layer must be kept exact.
4. Narrower wire for large sums (1250 q8 wire, 1272 host compressed wire: both unqualified): f16 first, with a CPU f32 reference check.
5. Rank balance: move the tensor split (TS=0.31,0.27,0.42) so the cards arrive together, or let an early card start its transfer before the last arrives.
6. The 17-19% idle: host-side profile of the server during prefill (perf), since the dispatch call itself is about 1 s per 100 s (1356 lock counters).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each step is one patch or one profile setting, lightweight tier: offline mechanics + patch-lint, activation marker, ABBA on one binary at 8K / 24K / 98K with complete separation, greedy identity where the mechanism is exact (async transfer, staggering) and a CPU f32 reference comparison where it is not (narrow wire), repeated-run identity at 98K (at least twelve runs: the 1356 race was only visible there).

## Effort & Risk



## Standards



## Acceptance Criteria

- The exchange's rate against the PCIe link limit is stated.
- The asynchronous transfer and the half-batch staggering are each promoted or rejected with their measurement.
- The narrow wire is promoted with a reference comparison no worse than production, or rejected.
- A card's computing share of prefill time is reported before and after (ar-prefill-stats.py), not only tok/s.

## Notes

The same structure exists in radiance on two 7900 XTX (collectives through pinned host memory, 33% of two-card time; a lossy wire took prefill from 1,535 to 2,120 tok/s), so what works here should carry over. Threads: QFP41 owns the threading rules and QFP46 the prefill concurrency contracts; the dispatch workers (1356) are evaluated, not validated, since 2026-10-11.

2026-10-11 step 1, message size against the links (arithmetic from the trace and the model header, not a separate measurement). Flash-Next n_embd = 2560, so a 512-token sum is 512 x 2560 x 4 = 5.24 MB of f32 a card; 96 sums a batch = two a layer over 48 layers. Links (queue log, every run): XTX 0 and 1 at PCIe 16 GT/s x8 (about 15.7 GB/s each way), the R9700 at 16 GT/s x4 (about 7.9 GB/s, the slot's maximum), the 6900 XT at 8 GT/s x4. 5.24 MB over the R9700's x4 link is 0.66 ms one way; the measured exchange after the last card arrives is 0.89 ms median. So the exchange is already close to what the R9700's x4 link can carry: it is link-bound, not software-bound. Consequences: (a) a narrower wire is the direct lever (f16 halves it to about 0.33 ms one way); (b) overlap hides the time but cannot shorten it; (c) the R9700 in an x8 slot would halve it with no code - a hardware decision for the owner; (d) any design that makes the R9700 exchange less data per sum (a smaller share of the split on that card does not help: every rank exchanges the full activation).

2026-10-11 design for the staggering (owner: "do staggered batches next"), from reading the composed source.

How the split backend runs a graph today (ggml-backend-meta.cpp, `ggml_backend_meta_graph_compute`): the graph is cut into subgraphs at every PARTIAL node (97 for a Flash-Next prompt batch). The execution loop is `for i in subgraphs: submit subgraph i on every backend (asynchronous); comm_allreduce(last node of subgraph i)`. The all-reduce goes onto each card's COMPUTE stream (NCCL path in ggml-cuda.cu, bf16 wire above 131,072 elements for three cards), so the card's queue is held for the 0.9 ms exchange plus the wait for the slowest card, and subgraph i+1 needs its result. One token stream has nothing independent to run meanwhile.

Where independent work comes from: two consecutive prompt batches. Batch B (tokens after A) needs from A only what A has written to memory for the SAME layer - the KV cells and the recurrent (delta-net) state - and A needs nothing from B. So the subgraphs can be interleaved by index: A_0, B_0, A_1, B_1, ... with every A_i before B_i on each card's stream, exactly the order of dependencies. While A_i's sum is in transit, the cards compute B_i; while B_i's is in transit they compute A_(i+1), which waits only for A_i's sum.

What that needs, in four pieces:
1. HIP backend: an all-reduce issued on a per-device TRANSFER stream. The transfer stream waits for an event recorded on the compute stream after the subgraph's last node; the compute stream waits for the all-reduce's completion event before the first kernel that reads the sum. (The standalone probe `overlap` shows the transfer stream runs beside the compute stream at no cost to either; `stagger` shows 95% of the exchange cost recovered on the three target cards.) On its own, with one token stream, this must be output-identical and no slower: that is the first build and the first test.
2. Split backend: `compute_pair(graph_A, graph_B)`: the loop above over both graphs' subgraph lists, alternating, with the asynchronous all-reduce of piece 1. Both graphs have the same topology (the last, shorter batch keeps the same subgraph count), which the function asserts; if not, it falls back to one after the other.
3. Scheduler / llama: two graphs alive at once need two compute-buffer sets. There is no VRAM for a second set at ub512 and ctx 245760, so the pair is two batches of 256: the large per-batch tensors (the QSA masks, [n_kv, T]) scale with T, so two half-size sets cost about what one full-size set does. To be measured before anything else: the reserve sizes at ub256.
4. llama_context::decode: when at least two prompt batches remain, build both (all tokens and KV slots are known up front) and submit them as a pair; otherwise the existing path. 1359's fences are per decode call and need one per batch of the pair.

Order of work: piece 1 alone (identity, no slowdown); then the ub256 memory measurement and a plain ub256 A/B (ub256 alone will be slower: more, smaller batches); then pieces 2-4 behind one default-off flag; then the usual evidence with twelve identity runs at 98K. Risks: the recurrent state and any other per-layer memory written by A and read by B must be on the stream-ordered path (to be inventoried per op, not assumed); HIP graphs are captured per subgraph and the pair changes what is captured when (the 1356 race needs HIP graphs and the fusion pass - same area); RCCL's group calls on a second stream.

Expected ceiling: the sum kernel is 32-37% of a card's prefill time, of which about two thirds is the exchange; the probe's 95% recovery applied to that is roughly a quarter more prefill throughput. Not a promise: ub256's own cost comes off it.

Also measured and closed here: sending large sums through the cpu-root path is slower at both chunk sizes (runs ar1, ar2); a 16-bit wire is already what the large path uses; moving the tensor split to help the R9700 does not fit at ctx 245760 (out of memory on an XTX).

## Change Log

- 2026-10-10T13:54:01.542976+00:00 (created-by): Created by claude
- 2026-10-10T13:55:13.057445+00:00 (updated-by): Updated: section:notes
- 2026-10-10T21:32:48.900499+00:00 (updated-by): Updated: section:notes
