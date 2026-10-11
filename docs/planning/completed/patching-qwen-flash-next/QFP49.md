---
id: QFP49
order: 49
plan: patching-qwen-flash-next
state: deprecated
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

2026-10-11 ub256 on its own (run ub256, build b-pipe2 with the pipeline on, Flash-Next 24K, ABBA, B = UB=256 B=256): prefill 1,134.9 / 1,129.7 t/s against 1,443.9 / 1,427.8 at ub512, -21%. The text differs from ub512's (a different batch size is different arithmetic) and decode in that arm is 75 against 91 t/s with lower acceptance, so ub256 is not a setting to adopt by itself. Memory: the Meta compute buffer is 459 MiB a card at ub256 against 1,021 MiB at ub512, and card memory in use drops by 0.45 GB on each XTX and 0.19 GB on the R9700 - so two ub256 graphs alive at once fit in what one ub512 graph takes (918 against 1,021 MiB). Consequence for the staggering: the memory premise holds, and the pair starts 21% behind. It has to win that back and more purely from hiding the exchange (at ub256 there are twice as many sums a token, each half the size), so the first thing the paired build must show is that it beats ub512, not ub256.

Gate for staggering, 2026-10-11 (kernel traces, 38,725-token prompt, build b-pipe2; run dp256 = ub256, run pipe4 = ub512, both under rocprofv3). Per RX 7900 XTX: ub512 kernel time 24.8 s = 10.4 s cross-card sum + 14.4 s compute (prefill 1,330 t/s traced); ub256 kernel time 31.1 s = 12.8 s sum + 18.3 s compute (1,048 t/s traced). R9700: 11.1 s / 13.5 s at ub512, 14.7 s / 16.8 s at ub256. So compute per token is 27% dearer at ub256, and a sum costs 1.43 ms at ub512 against 0.88 ms at ub256. A staggered ub256 pair has a floor of 18.3 s of compute per XTX against 24.8 s of compute plus sum today: it beats ub512 if more than about half of the sum time is hidden (the standalone stagger probe hid 95%). Gate passed; build pieces 1-4. Note for later: a pair of ub512 batches has a 14.4 s floor but needs a second 1,021 MiB compute buffer per card, which does not fit at ctx 245760.

Standalone check with RCCL, 2026-10-11 (tools/lab/hip-probes/rccl_stagger_probe.cpp, run rs1, 2x RX 7900 XTX + R9700): 96 sums of 2.62 MB with ncclAllReduce from one host thread. Compute only 172.0 ms; as today 255.4 ms; half-size rounds unstaggered 266.0 ms; staggered halves with the sum on each card's transfer stream, fenced by events, 185.6 ms. Staggering hides 84% of today's sum cost, 0 wrong values, and RCCL takes a non-compute stream on the same communicator. Piece 1 (patch 1360, the sum on a transfer stream, default off) is with GPT.

Piece 1 done, 2026-10-11: patch 1360_ar_transfer_stream merged (#145, untested, default off). Flash-Next 24K, build b-ar1, run tsar: text identical, prefill 1411.9 / 1438.3 on against 1424.1 / 1424.4 off, marker in the on passes only. Identity and speed reference for the pair (run ref256, same build, -b 512 -ub 256, stagger off): prefill 1081.2 / 1081.5 t/s, greedy md5 e04e5eb0, decode 76.4 / 77.4, acceptance 333-334 / 531-534; ub512 in the same run 1444.8 / 1437.3, md5 a789c151. Two ub256 halves a call are slower than -b 256 -ub 256 (1,135) because 1359 only fences a one-ubatch call. Targets for the pair at half 256: text e04e5eb0, prefill above 1,445. Pieces 2-4 are patch 1361_prefill_stagger_pair (branch feat/qfp49-stagger-pair, with GPT).

Ladder of proofs, 2026-10-11 (owner: build up from small proofs before engine complexity). Rungs A/B = tools/lab/stagger-harness/meta_pair_harness.cpp: a toy 96-layer model on the real Meta backend (Q8_0 weights split across the cards, one cross-card sum a layer, a static cache half B reads after half A writes it), run in sequence, as a pair, in the wrong order (B, A), and as one full batch. Build b-stg6 (1360 + 1361 at PR #147 head), 2x RX 7900 XTX + R9700. Bugs it found: (1) half B's views of a shared static buffer lost when half A is captured (GPT fix 3c4b1fdd); (2) 1360's state kept a pointer to the caller's tensor array, freed before end() in a pair: SIGSEGV (fixed in 1360, 1c76a17a); (3) harness rules now written in it: reserve every scheduler with a separate graph before allocating any graph, results copied to static tensors by the graph. Results: pair bit-identical to sequence at half 128 / 256 / 512; wrong order differs in every value; ms full / sequence / pair: half 256 k=1024 102 / 135 / 101; k=2048 109 / 139 / 98; k=4096 118 / 151 / 95; half 128 k=4096 75 / 91 / 59; half 512 k=1024 203 / 209 / 178; k=4096 234 / 244 / 163 (one repeat 228). Unexplained: the pair's absolute time falls as k grows. Rung C = real Flash-Next, no drafter, token_embd on the cards, ctx 65536, 10,154-token prompt (run rc1, rc1f): ub512 1,500 / 1,371 t/s; -b 512 -ub 256 sequence 1,214 / 1,220; stagger flag on 908 / 902 with identical text and NO marker: the pair never engages, because the prompt graph has two scheduler splits (host input split + Meta split; sched_reserve logs splits = 2) and the pair entry requires one; and the decline path costs 25%. Both sent to GPT. GPT's assessment of the gate: counting host time the 256 pair needs about 70% of the sum hidden, expected 1,450-1,600 t/s against 1,445; competing levers without the half-size tax: split balance (R9700 last), 8-bit wire, the remaining 12.9% idle, fewer sums a layer.

Rung C with the pair engaged, 2026-10-11 (build b-stg9 = PR #147 head with GPT's prefix-split support and decline latch; real Flash-Next, no drafter, token_embd on the cards, ctx 65536, 10,154-token prompt, -b 512 -ub 256; run rc7, B passes under gdb). The marker appears: the pair runs. (1) OUTPUT IS WRONG: stagger off md5 02f04126 (prose), stagger on md5 ce016c15 in every completed pass, the text a run of '3'. Deterministic. The declined-pair run earlier matched stagger off, and the toy harness is bit-identical on the same build, so the fault is in what the toy lacks: host prefix split and input copies for two halves. (2) Intermittent stall: one pass 61 t/s (stalled ~160 s, recovered), one hung until interrupted (18 min), an earlier un-wrapped run hung 48 min. Stack of the decode thread at the hang: ggml_backend_sched_graph_compute_async_pair -> ggml_backend_meta_graph_compute_pair -> prepare callback -> copy_inputs -> ggml_backend_sched_copy_input -> ggml_backend_meta_set_tensor_async -> ggml_backend_cuda_set_tensor_async, blocked inside the HIP runtime; all other threads idle. (3) Speed of the (wrong) passes 1,242-1,248 against 1,224-1,236 in sequence; not meaningful until output is right. Both findings sent to GPT with identity first, then the host path (non-blocking input staging, per-call host work, timing line), and a harness extension that must reproduce both. Lab tools added: tools/lab/flash-next/gdb-wrap.sh (Brutus has ptrace_scope=1, so a server must be started under gdb to get a stack), hang-watch.sh. Owner's 2 x 512 pair at reduced context is queued behind the identity fix. State of the lever: mechanism proven on the split backend, not yet correct in llama; no promotion evidence.

Profiling pass, 2026-10-11 (owner: prove where and how the work happens). New tools on the PR #147 branch: tools/lab/stagger-harness/llama_prefill_harness.cpp + run-llama.sh (the real model through the llama library only - no server, drafter or sampling - one process an arm, last-token logits compared bitwise, per-call host time, optional rocprofv3 trace); tools/lab/flash-next/kernel-overlap-stats.py (per card: compute only / sum only / sum + compute / idle, and kernels per queue and stream). Baselines from existing traces: production ub512 (pipe4) and ub256 (dp256) both 0.0% of sum time under compute, one stream a card, about 43% compute / 31% sum only / 26% idle. Real model, build b-stg11, Flash-Next, per_layer_token_embd on CPU, cpu-root, ctx 65536, 4,096 pseudo-random tokens, -b 512 -ub 256 (runs lh1, lh2 traced, lh3): (1) the pair has never overlapped anything in the engine: the stagger arm declines on its first call with 'different Meta subgraph shapes' and runs as sequence; sum + compute is 0.0% in every arm. (2) logits bit-identical across all arms. (3) the host is not ahead of the cards without a drafter: llama_decode holds the host about 340 ms a call, 3,591 of 3,745 ms. (4) per XTX in sequence: compute 39.1%, sum only 36.2%, idle 24.7%; 1,528 sums of 0.91 ms; idle 0.94 s of 3.84 s, of which 0.51 s in 5-6 gaps of 10 ms or more (between calls) and 0.40 s in 10,075 gaps of about 40 us after mul_mat_q. (5) 1360 on puts the sums on their own stream (seen in the trace) and has a BIMODAL cost: six arms with the flag gave 883, 884, 871, 864 (traced), 1,047, 1,105 t/s against 1,076-1,146 without it; in the server every flag-on pass was the slow mode (898, 962, 891, 946 against 1,158-1,229). In the slow mode a sum is resident 1.01 ms against 0.91. At -ub 512: 1,295 / 1,252 without, 1,142 with. So 1360 alone is not neutral in this configuration (it was on the production profile: 1,412 / 1,438 against 1,424 / 1,424); cause not found (queue ids differ between runs - not yet correlated). (6) Server run rc11 (10K prompt): pair engaged for one call, then declined 'pair-safe pinned input staging unavailable' (input sizes grow with n_kv; 4 MiB cap). Earlier attribution of the 22% to the pair's fallback was wrong: run ts256 shows 1360 alone gives it. Acceptance now set for the pair: bit-identical logits, no decline, sum + compute well above zero in the trace, then speed. Fresh GPT brief sent with all of this (follow-ups on a kept gateway session do not get delivered; only new sessions do).

2026-10-11 gate and restart. PR #147 head 542f2729 (build b-stg12) failed the gate set for it (slim harness: bit-identical logits, no decline, sum + compute above zero): the pair aborts on its first call (GGML_ASSERT in bc_1361_copy_input_pinned: slot buffer smaller than the input) and the toy's new host-prefix mode fails its own setup. PR #147 parked as a draft; lab tools moved to main (#156). Restarted in slices; SLICE 1 = what does the sum on its own stream cost, and what does staggering hide, measured outside the engine (tools/lab/hip-probes/rccl_stagger_probe.cpp, new arms and options; 2x RX 7900 XTX + R9700, real RCCL, sums right in every run, runs rs3, rs7-rs12). (a) Sum on a transfer stream fenced by events, nothing overlapping: within +0.016 ms a sum of the compute stream, also with the stream non-blocking and events timing-off as 1360 creates them. So the 0.2-0.4 ms a sum that 1360 costs in the engine (bimodal) is engine-specific and unexplained. (b) DECISIVE: the share of today's sum cost that staggering hides falls with the number of kernels launched between two sums, same total work: 2 kernels 84%, 10 kernels 75%, 40 kernels 42%, 80 kernels 13%, 160 kernels -9% (compute only 172.0 / 172.7 / 181.3 / 191.1 / 212.2 ms; today 255.7 / 274.7 / 283.2 / 285.7 / 314.3; staggered halves 185.6 / 198.3 / 240.5 / 273.8 / 323.3). The engine launches about 44 compute kernels a sum on an XTX (68,819 kernels, 1,528 sums at ub256; 374,406 / 9,216 at ub512), so about 40% hidden at best, against the roughly 70% the ub256 pair needs to beat ub512 (GPT's gate arithmetic). This also explains the earlier proofs: first probe 2 kernels a step (84%), Meta toy 6-8 kernels a layer (pair 22-39% faster), real model 44. Mechanism not established (guess: the card alternates between streams at kernel boundaries, so a waiting sum kernel takes a turn at each). RECOMMENDATION recorded for the owner: reject 1361 as designed and 1360 with it (its only consumer; unexplained slow mode); redirect to fewer, larger kernels a step (QFP48 fusion, megakernel), which removes launch gaps directly (0.40 s of 3.84 s in 10,075 gaps after mul_mat_q) and is the precondition for staggering. Patch state changes held for the owner's word.



2026-10-12, owner decision: the staggering work is shelved. 1360 (transfer stream) is `evaluated`, default off, in no build. 1361 (staggered half-batch pair) never produced a correct real prompt and was never merged (PR #147 closed). Why it cannot pay on this workload is measured: the standalone probe hides 42% of a sum at 40 kernels between sums and 13% at 80, Flash-Next runs ~44 a step, and on b11552 the cards are 91-92% busy in a median step. The harnesses and probes stay in tools/lab (stagger-harness, hip-probes). This item is closed as not pursued.

## Change Log

- 2026-10-10T13:54:01.542976+00:00 (created-by): Created by claude
- 2026-10-10T13:55:13.057445+00:00 (updated-by): Updated: section:notes
- 2026-10-10T21:32:48.900499+00:00 (updated-by): Updated: section:notes
- 2026-10-10T21:37:41.672650+00:00 (updated-by): Updated: section:notes
- 2026-10-10T21:46:35.701673+00:00 (updated-by): Updated: section:notes
- 2026-10-10T21:49:01.610106+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261010_223737_groundwork-for-staggered-promp_5862
- 2026-10-10T22:37:44.216342+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-10T22:43:01.082723+00:00 (updated-by): Updated: section:notes
- 2026-10-11T01:19:48.122756+00:00 (updated-by): Updated: section:notes
- 2026-10-11T03:15:29.826530+00:00 (updated-by): Updated: section:notes
- 2026-10-11T04:43:44.890764+00:00 (updated-by): Updated: section:notes
- 2026-10-11T05:40:26.127515+00:00 (updated-by): Updated: section:notes
- 2026-10-11T23:07:23.946556+00:00 (updated-by): Updated: section:notes
- 2026-10-11T23:07:36.827693+00:00 (state-transition): State: pending → deprecated
