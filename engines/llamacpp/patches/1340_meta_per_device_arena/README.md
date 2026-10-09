# 1340_meta_per_device_arena — merged Meta arena mechanism

## PA44-E merge record

Packaging-only merge. The former 1339 memory-report diagnostic is now owned by 1340. `BIGCHERRY_META_MEM` and `BIGCHERRY_META_PER_DEVICE_ARENA` retain their existing behavior and emitted source rows.

The former READMEs are preserved verbatim below.


---

## Preserved README: 1339_meta_memory_report

# 1339_meta_memory_report

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md. This is a
diagnostic, promoted as a neutral enabler: with `BIGCHERRY_META_MEM=1` the tensor split reports the size of every
buffer on every device and the fusion overlap counters; without the variable it prints nothing and changes nothing.
1340 (per-device compute arena) is written on top of its hooks, so the two are promoted together.

## Evidence

- No effect when off: every "production" arm of the MSM02 / MSM03 runs of 2026-10-07 (builds `b-metamem-msm3kq`,
  `b-metamem-msm2s2`, `b-metamem-msm2t2`: production set + 1339 + 1340, flags off) gave the production greedy texts
  and probe distributions (production against itself: 24 probes, top-1 24/24, TV 0.0000).
- Activation: the report lines themselves (`BIGCHERRY_META_MEM compute|arena|static ... dev=`), which
  `tools/lab/flash-next/queue-meta-mem.sh` sums per device; they are the source of every memory figure in 1340's and
  1341's promotion records.
- Offline: package tests (`tools/tests/patch/test_1340_meta_per_device_arena.py` and
  `test_1341_meta_subset_mirrored.py` compose it), patch-lint, production composition check.



---

## Preserved README: 1340_meta_per_device_arena

# 1340 — Meta per-device compute arena

Purpose: remove the tensor-split Meta backend's common compute-arena reservation when devices own different transformed graphs.

## Runtime contract

On by default; `BIGCHERRY_META_PER_DEVICE_ARENA=0` restores the common-size arena.

At llama.cpp b11402 (`d89651a7b205`):

- `ggml_backend_sched_reserve()` owns the worst-case scheduler measure graph. 1340 calls a Meta reserve-only hook there, after logical split/reserve and before scheduler reset.
- `ggml_backend_meta_reserve_graph()` maps that same graph to every simple backend. Each simple backend owns one `arena_galloc` physical buffer. Per-shape `arena_plan_t` gallocr objects are bufferless allocation plans.
- `ggml_gallocr_reserve_grow()` preserves the maximum size reached by each physical vbuffer chunk across reserve shapes; it is not called in the normal compute path.
- `ggml_backend_meta_alloc_graph()` maps the current graph, validates the matching reserve-time plan, and binds it into the shared physical owner with `ggml_gallocr_alloc_graph_reuse_from()`.
- A graph that does not fit its stored plan gets a new layout inside the same reserved arena. That is normal: the graphs that run have other shapes than the reserved measure graph and inputs sized by the filled context grow with it (the scheduler's own allocator re-plans at the same moments). It is counted (`replans`) and reported only under `BIGCHERRY_META_MEM=1` (`arena_replan`). The invariant is that the physical arena never grows after reserve: growth logs `GGML_LOG_ERROR` `BIGCHERRY_META_MEM arena_grew ...` and increments `grows`, which must stay 0.

The design is device-count and topology agnostic: no GPU index, model name, attention split, or backend name is encoded in the arena logic.

## Reporting

With `BIGCHERRY_META_MEM=1`:

```
BIGCHERRY_META_MEM arena dev=<n> buft=<name> reserved_mib=<MiB> plans=<count> replans=<count>
```

Expected normal result after load: `replans=0`; `reserved_mib` does not grow with context fill.

## Validation

Before every push:

```bash
PYTHONPATH=tools python -m unittest tools.tests.patch.test_1340_meta_per_device_arena tools.tests.patch.test_1341_meta_subset_mirrored
PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --experiment meta-memory
```

The offline test reads b11402 from the vendor Git object (`git show`), never from its patched working tree.

## Promotion record

QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7, 2026-10-07. Opt-in memory-layout change of the tensor
split (`BIGCHERRY_META_PER_DEVICE_ARENA=1`); it pays off together with 1341's `BIGCHERRY_META_SUBSET_MIRROR_INPUTS=1`,
which takes the context-sized attention masks off devices with no attention share. Flash-Next UD-IQ4_XS, production
topology (2x RX 7900 XTX + R9700 tensor split, `BIGCHERRY_ATTN_TS=1,1,0`, MTP drafter on the RX 6900 XT), ctx 245760,
f16 KV, build `b-metamem-msm2t2`, both flags on against production on one binary.

- Memory, compute arena in MiB on XTX / XTX / R9700 (runs `metamem-msm2t2`, `metamem-msm2t98`): production
  1020.9 / 1020.9 / 1020.9; flags on 880.8 / 880.8 / 378.8, reserved at load and the same after a 2K and after a 98K
  fill (`grows=0`, no `arena_grew`). New layouts inside the reserved arena: 11 per device at 2K, 246 at 98K.
- Identity with fusion off (`GGML_CUDA_DISABLE_FUSION=1` on both arms): greedy text identical at 8K and 24K, probes
  24/24 top-1 and TV 0.0000 (`msm2t-nofuse`, `msm2s-nofuse2`). The mechanism computes the same values.
- With fusion on the output is NOT identical to production: `ggml_cuda_check_fusion_memory_ranges` decides by address
  overlap, so another layout fuses another set of nodes (98K run: 16713 of 145577 checks refused in production, 16118
  of 146735 here), and fused and unfused kernels differ numerically (FKE01). Flags on against production: top-1
  23/24, TV mean 0.0721, max 0.2211. Against the CPU f32 reference the two are equally far: production top-1 21/24,
  TV mean 0.0950; flags on 22/24, TV mean 0.0635 (`msm2s-ref`). Stated equivalence, not identity.
- Speed (ABBA `msm2t-ab`, A = production / B = flags on, t/s): prefill 8K 1048.9, 1074.4 / 1062.1, 1062.5; 24K
  1075.4, 1074.5 / 1068.1, 1067.1; 98K 991.1, 991.0 / 996.7, 996.7. Decode 8K 84.2, 85.0 / 82.4, 83.4; 24K 74.8,
  75.3 / 70.0, 71.3; 98K 57.6, 58.3 / 57.6, 58.0. The 24K decode difference comes with a different text and lower
  draft acceptance (333-335 against 345 of ~500); with fusion off, where the texts are identical, decode is equal at
  24K (64.7, 64.7 / 65.3, 64.4) and 98K (53.0, 55.4 / 53.3, 53.3), so planning has no measurable cost.
- Offline: package tests, patch-lint, production composition check.

On by default since 2026-10-07, after a second model (Qwen3.8-27B Q8_0, two RX 7900 XTX in tensor split, plain and
built-in MTP5, ABBA `pda27-1..4`, A = off / B = on): compute arena per card 316.1 MiB reserved against the common
520.1 / 560.3 MiB, greedy text identical on all four runs, prefill 1317.6, 1314.7 / 1310.5, 1314.0 t/s (MTP5) and
1481.9, 1483.0 / 1482.2, 1480.7 (plain), decode 80.1, 80.3 / 80.1, 80.2 and 37.9, 38.0 / 37.9, 37.9, no re-plans.
The evidence above was taken with the flag as an opt-in. `BIGCHERRY_META_PER_DEVICE_ARENA=0` restores the
common-size arena; on Flash-Next the default changes the generated text at near-tie points (fusion set), as
described above.


Third model, Gemma 4 26B A4B (UD-Q5_K_S) on two RX 7900 XTX in tensor split (runs `gem-1..6`): compute arena 370.1 MiB
reserved against the common 428.1 MiB, no re-plans, greedy text identical to the common-arena runs, prefill 3391.5,
3349.0 t/s against 3417.3, 3427.5 (defaults) and 3342.3, 3368.7 (today's kernel patches off) - inside the spread of
the other two arms; decode 93.1, 93.3 against 93.1, 93.4.

