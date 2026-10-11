---
id: MET06
order: 6
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:45:10.449925+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# Runtime expert slicing only after host-weight pipeline qualification

## 2026-10-11 audit: #29963 host-weight pipeline and HIP gate

**Disposition: qualify the existing upstream scheduler mechanism before any standard-GGUF expert slicing. No new loader/placement patch and no hardware job queued.** PR [llama.cpp #29963](https://github.com/ggml-org/llama.cpp/pull/29963) is **open and ready for review**, not a draft or merged baseline (head `f81e182b340c452adb4725cd6a44c5d6734baea1`, updated 2026-10-10 23:20 UTC). The b11552 pin still disables pipeline parallelism when tensor overrides are present and disables it with a MoE cache. Do not claim that #29963 runs on the production tensor-split target: its admission is explicitly **layer split**, and neither this run nor the external MI50 benchmark exercised BigCherry's two XTX + R9700 tensor split or the 6900 XT drafter.

### Exact source ownership, data flow and lifetime

1. Pinned `src/llama-context.cpp::llama_context` requires >1 device, all layers offloaded, `LLAMA_SPLIT_MODE_LAYER`, KQV offload, no tensor overrides and no MoE cache, plus backend async/event support. #29963 relaxes the override restriction **only** with op offload and `llama_host_weights_spread(model)`. Its host-byte test rejects any device with **more than two-thirds** of the host-weight bytes; this is a *spread* heuristic, not an equal-layer or PCIe-cost proof. Source observation: with zero host weights its map is empty and the helper returns true, so an unrelated tensor override can admit unnecessary pipeline mode. Before porting, require a positive nonzero host-weight byte/copy counter, and make this case an explicit negative fixture; do not label it a reproduced runtime regression.
2. `common/fit.cpp::common_params_fit_impl` balances layer slices, then bisects dense-only-to-full layer conversions under per-device memory targets. `src/llama-context.cpp::graph_get_cb` assigns ops consuming host weights to the layer's backend (except GET_ROWS), for ubatch >=32. Do not silently change the production `-sm tensor` placement or reimplement MET01/MET11's expert placement solver.
3. `ggml/src/ggml-backend.cpp::{ggml_backend_sched_split_graph,ggml_backend_sched_copy_input,ggml_backend_sched_upload_user_inputs,ggml_backend_sched_compute_splits}` aliases one user-input copy across scheduler slots and records `input_events`; host weights get one **transient, split-owned** destination rather than a persistent copy per scheduler slot. Host-weight upload uses `ggml_backend_tensor_set_async` on the consuming backend stream. The allocator may reuse the destination **after its consuming split**, not before its last GPU read. Audit event creation/free, user-input views, source-pointer lifetime, split-to-split waits, graph capture/replay, cancellation and repeated requests. A transfer-enqueue timestamp is not proof that its bytes were consumed correctly.
4. `src/llama-context.cpp::{process_ubatch,graph_reserve}` re-reserves a worst-case graph after smaller graphs to avoid per-ubatch allocator reallocation/synchronization; the source logs failed re-reservation and needs an explicit failure-path test. `sched_copy_experts` changes from selective IDs to whole-expert uploads when a prior graph is in flight and `n_ids >= 16*n_expert`, or for a full batch with `n_ids >= 64*n_expert`. `upload_all` remains sticky for the graph. Record exact bytes and graph-in-flight state; a large-batch PP gain may coexist with expensive whole-weight traffic, especially on PCIe 4.0 x4.
5. Existing `1336_sched_copy_callback` is marked superseded by native #29943, and `1337` is a distinct persistent cache with historically divergent logits/negative small-budget results. **No** new copy callback, cache, prefetch ring, fit selector, stream or telemetry registry belongs to MET06. MET12 alone owns *predictive pre-router* prefetch if exposed H2D stalls remain after this gate; #29963 performs exact per-split uploads, not that prediction. QFP49/1360/1361 and current b11552/1346/1359 scheduler work are protected by the 12-hour exclusion: record future patch-composition conflicts here, do not edit their files or queue competing jobs.

### Evidence: measured versus hypothetical

- **BigCherry measured (MET01, gfx1201 PCIe 4.0 x4):** effective host upload about **6.7 GB/s**; host-expert pp throughput at ub512/1024/2048/4096 **93.9/159.0/265.8/420.6 t/s**. At equal VRAM, whole-layer residency beat the 8 GiB LRU cache; 12 GiB LRU was about 24% slower on prefill. These are local controls, not measurements of #29963.
- **External AMD, 2026-10-10 comment on #29963:** four **gfx906 MI50**, ROCm 7.14, layer split 25/25/25/25, host experts and HIP graphs/RCCL, DeepSeek-V4-Flash-Vision MXFP4. At prompt 512: **81.98 -> 83.24 pp t/s**; 8192: **149.68 -> 191.60**; 32768: **122.47 -> 312.46** (three repetitions each). TG256 changed approximately +1.4% to +1.9%. This is **external** and used a PR head whose merge base differed from the tested master; no inference about gfx1100/gfx1201/no-P2P or correctness follows. [Exact comment](https://github.com/ggml-org/llama.cpp/issues/29963#issuecomment-6103252758).
- **External CUDA, #29963:** three RTX PRO 4000 PCIe 5.0 x16 devices achieved 1009/1019 -> 1808/2102 pp t/s for long Qwen3.8 server prompts; with CPU_Mapped rather than pinned host experts, a separate 35B comparison was only 2029/2052 -> 2088/2101. This demonstrates a host-memory-mode confound, not a portable gain. The PR reports an input-copy buffer reduction 9133 -> 2988 MiB/GPU on another configuration; no BigCherry allocation reduction has been measured.
- **External alternative:** [SGLang OffloaderV2](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py) uses a separate offload stream and recorded load events; its graph-capture branch has a distinct path. [vLLM UVAOffloader](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/offloader/uva.py) instead accesses pinned CPU weights via UVA and may fall back to on-demand copies. Use only their lifetime/stream/fallback invariants as references; neither is an AMD BigCherry throughput control.

**Cheap source-derived discriminator actually run:** 12/12 Python host assertions for layer-vs-tensor admission, balanced/skewed/no-host override, missing async/events, and 16x/64x ID thresholds including sticky whole-upload state. The no-host override case is *admitted by the inspected helper*, not a proven GPU failure. No repository pytest, C++/HIP build, GPU, NVMe or performance benchmark ran.

### Bounded qualification and terminal decision

**Gate A — offline, no hardware contention:** compare b11552 to #29963's actual changed code in `common/fit.cpp`, `ggml/src/ggml-backend.cpp`, `src/llama-context.{cpp,h}`; prove default-off/stock behavior, source-level backend async/event admission, positive host-weight spread and nonzero copy bytes, alias/event slot lifetime and graph-reserve error handling. Patch composition must wait for independent b11552/QFP49 work to stop advancing; do not edit those active paths in this audit. Reject the no-host-weights admission case or add a proven guard before any promotion.

**Gate B — isolated HIP control, only after Gate A and an available idle lane:** 2x XTX `-sm layer` first; compare (A) b11552 CPU host experts/pipeline off, (B) #29963 transient host-weight upload/pipeline on with **identical layer ownership and tensor overrides**, (C) B with matched pinned-host vs CPU_Mapped mode, and (D) equal-VRAM best MET01 whole-layer placement. Run fit-balanced placement as a separate comparison, not mixed into A/B. Only then test layer split with the R9700 PCIe x4 third rank; gfx1030 is a negative/aux control, not a P2P helper. **Tensor-split production is a separate unsupported control, not a positive #29963 lane.** No simultaneous QFP49/RCCL queue.
 
For pp512/2048/4096 at short/24K/98K context and tg128/512, record backend owner/layer, host buffer type and pin status, input/weight copies by device and byte count, actual PCIe GB/s, event wait vs copy/compute overlap, allocator reallocations/high-water mark, layer split count, graph capture/replay, queue gaps, MTP on/off and effective t/s. Require same-process cold/warm/workload-shift, alternating ubatches, graph on/off, no-P2P transfers, full logits/KLD/greedy token identity, exact expert work accounting, and no skipped uploads. Validate whole-copy thresholds on both sides. Compare matched controls over >=4 sessions with >=10 paired ABBA rounds; accept pipeline only with **CI95-low >=3% end-to-end PP** and <=1% TG/control regression plus correctness. Otherwise reject the port.

**Gate C — loader-slicing decision:** retain MET06's original threshold only after Gate B: implement a single loader-owned standard-GGUF expert-slab materialization primitive **only** if coarse whole-layer placement still costs >=5% measured PP/TG or >=1 GiB avoidable resident VRAM at equal throughput. Verify `nb[2]` quant-block boundaries, file offsets, model/profile fingerprint, complete expert ID bijection, exact offline-repack equivalence and fallbacks before any runtime switch. If pipeline/residency closes the gap, retire the slicing proposal. If #29963 remains unmerged or fails HIP correctness, keep the native scheduler and document negative evidence rather than creating a speculative parallel scheduler.

## Description

Remove the offline-repack requirement only after MET03-MET05 prove expert-granular placement has value. Before implementing a new per-expert GGUF loader, qualify the cheaper upstream host-weight execution mechanisms now emerging in llama.cpp. MET06 must not duplicate MET01 residency policy, MET05/1328 auxiliary-device transport, or upstream scheduler host-weight upload/pipeline machinery.

The original proposal was to synthesize compact `[ne0,ne1,n_tier]` tensors in `llama_model_loader`, copy selected expert slabs at load, permute router rows, and add `--expert-placement <json>`. That remains the fallback capability, not the first implementation.

## Steps

1. Establish standard-GGUF controls with no offline repack: current upstream, whole-layer host experts, and MET01's best equal-VRAM placement baseline.
2. Qualify llama.cpp PR #29963 (`pipeline parallelism with MoE experts in host RAM`) before writing loader code. Measure whether scheduler-side transient host-weight upload plus layer pipeline removes enough CPU/host-expert cost to make expert slicing unnecessary.
3. For #29963 record per device: assigned layer slice, full vs dense-only layers, host-weight bytes uploaded/token, H2D copy time, overlap with compute, allocator peak, split count, idle gap, PP and TG. Test gfx1100/gfx1201 independently and the production 2xXTX+R9700 topology. Do not assume CUDA results transfer to HIP.
4. Verify scheduler semantics required by #29963 on HIP: async+events capability, `LLAMA_SPLIT_MODE_LAYER`, op offload, KQV offload, multiple scheduler copies, input events, and host-buffer recognition. A path that silently falls back is a failed experiment.
5. Add a pinned-host/prefetch control inspired by the Fable MoE-offload forks. Measure actual host->GPU bandwidth before/after pinning and overlap; do not promote CUDA-specific registration code to HIP without a ROCm-safe implementation and locked-memory budget.
6. Compare three mechanisms at equal model/VRAM budget: CPU execution of host experts; transient host-weight upload/pipeline; MET01 static/LRU residency. The governing metric is end-to-end critical-path ms/token and PP wall time, not cache hit rate or copy bandwidth alone.
7. Only proceed to runtime per-expert slicing if standard-GGUF mechanisms leave a measured placement gap: >=5% TG or PP opportunity attributable to coarse whole-layer residency, or >=1 GiB avoidable resident expert memory at equal throughput.
8. If the gate passes, implement one loader-owned slicing primitive. `llama_model_loader` creates compact destination tensors `[ne0,ne1,n_tier]`; raw expert slabs copy from `old_expert * nb[2]`; router/expert-id mapping is explicit metadata rather than destructive global row permutation where possible.
9. `--expert-placement <json>` remains a separate common parser/model-param surface, not an `-ot` extension. Reject overlap with `-ot` expert overrides, wrong model fingerprint, duplicate/missing experts, unsupported quant/layout, and impossible devices.
10. Reuse MET01's placement JSON and generation. MET06 owns only materialization of that decision from a standard GGUF. It must not add a second solver, cache, prefetch policy, or auxiliary transport.

## Detailed Solution & Technical Design

### New upstream discriminator: #29963

PR #29963 opened as a draft on 2026-10-04 and was marked ready for review on 2026-10-09 (still open/unmerged at the 2026-10-11 audit). Its important mechanism is not expert caching: host expert weights remain in system memory and are uploaded to the backend that owns each pipeline split. The scheduler identifies weight buffers that are host-backed, creates a single transient backend copy rather than persistent copies for every scheduler slot, and allows allocator reuse after the split. User inputs are uploaded ahead of split submission using per-backend/per-copy events. The fit logic gives devices approximately equal layer slices and converts as many dense-only layers to full layers as each device can hold.

This is potentially cheaper than MET06's proposed loader surgery because it works with an ordinary GGUF and attacks a different bottleneck: making host-resident experts GPU-executable without permanently consuming VRAM. It is also complementary to MET01: persistent residency should be reserved for experts whose avoided miss cost/GiB beats transient upload.

Do not merge the mechanisms conceptually:

```text
MET01 = which experts/layers deserve persistent residency
MET05 = how the auxiliary 6900 backend executes/stages work
#29963 = transient host-weight upload + layer pipeline
MET06 = materialize expert-granular persistent placement from standard GGUF, only if still needed
```

### Transfer-overlap qualification

A CUDA fork derived from Fable reports two useful controls for host-resident MoE prefill: pin mmap'd expert pages to avoid driver bounce-buffer copies and prefetch the next layer's experts on a second stream. Its published RTX 3060 Qwen3.6-35B-A3B pp2048 result is ~1143 -> ~1880 t/s (+64%) with both enabled, with reported host->device bandwidth rising from ~6-7 to ~20 GB/s. Treat these numbers as mechanism evidence only. On ROCm, measure `hipHostRegister`/pinned-allocation behavior, OS locked-memory cost, PCIe throughput, and whether the copy actually overlaps the target compute stream.

The test matrix should distinguish:

```text
A  host experts execute on CPU
B  host experts transiently upload, no prefetch
C  B + safe pinned-host source
D  C + one-layer-ahead prefetch/overlap
E  MET01 persistent static/LRU tier at equal VRAM
```

For B-D, report `bytes_uploaded/token`, copy-engine busy time, copy/compute overlap %, GPU idle at layer boundaries, and allocator high-water mark. A bandwidth increase that does not reduce PP/TG wall time is not a win.

### Runtime slicing fallback

If expert-granular materialization remains justified, avoid changing GGUF on disk. Build compact tier tensors at model load from contiguous expert slabs. Preserve an explicit `original_expert_id -> compact_slot` mapping per layer/tier so routed IDs can be remapped at the execution boundary. Prefer this over mutating router weights because it keeps model semantics and validation straightforward and can coexist with upstream cache/pipeline code.

The loader must account for quant block alignment and tensor strides; raw slab copying is legal only when `nb[2]` exactly denotes an independently copyable expert plane for that tensor layout. Otherwise use the existing tensor conversion/copy path or reject the layout fail-closed.

## Code Samples & Guidance

Conceptual ownership only:

```cpp
struct llama_expert_tier_entry {
    uint32_t layer;
    uint32_t original_expert;
    int32_t  device;
    uint32_t compact_slot;
};

// Loader materialization only. Placement was already solved by MET01.
for (const auto & entry : placement.entries_for(tensor)) {
    const size_t src_off = checked_expert_plane_offset(src, entry.original_expert);
    copy_expert_plane(dst, entry.compact_slot, src, src_off);
}
```

Do not add scheduler upload/prefetch logic here if #29963 or its upstream successor can provide it.

## Files

Likely fallback implementation: model-loader/model tensor creation path, common model-parameter parsing, MET01 placement schema reader, focused loader tests. Upstream qualification reference: `common/fit.cpp` and `ggml/src/ggml-backend.cpp` from llama.cpp #29963. No MET05/1328 transport duplication.

## Validation

- Standard Q6 GGUF + placement JSON produces byte-identical tier tensors to MET03 offline repack for supported layouts.
- Greedy token parity and logits/KLD match the repacked run within the existing contract.
- Before loader implementation, #29963-style transient upload is tested with real nonzero host-weight copies on gfx1100/gfx1201; silent fallback is rejected.
- ABBA >=5 repetitions for PP512/2048 and TG128/512 at short and long context; report median and dispersion.
- Compare cold and warm requests, MTP on/off, and production scheduler copies.
- Record peak VRAM and locked host memory so pinning cannot hide a memory-capacity regression.

## Effort & Risk

High if runtime slicing is required: quant-layout correctness, ID remapping, loader lifetime, and interaction with upstream scheduler/cache paths. Low-to-medium for the new qualification gate because #29963 operates at scheduler/fit level and requires no GGUF rewrite. The upstream PR remains unmerged and its principal published gains are CUDA-based, so AMD support must be demonstrated rather than inferred.

## Standards

One residency policy owner (MET01); one aux transport owner (MET05); reuse upstream scheduler mechanisms before forking; standard-GGUF first; fail closed on unsupported layouts; equal-VRAM comparisons; hardware evidence on gfx1100/gfx1201; optimize end-to-end critical path rather than local bandwidth.

## Acceptance Criteria

- Do not implement runtime slicing until the >=5% performance or >=1 GiB memory opportunity gate survives #29963-style transient-upload and MET01 residency controls.
- Any #29963-derived path proves nonzero host-weight upload and correct pipeline execution on HIP.
- No duplicate placement solver, expert cache, prefetch scheduler, or ROCm3 transport is introduced.
- If slicing proceeds, standard GGUF materialization is byte-identical to the offline-repacked tier tensors for supported quant layouts.
- Unsupported/ambiguous expert-plane layouts and conflicting `-ot` rules fail closed.
- Promoted result improves end-to-end PP/TG or capacity under the stated gate without >2% regression in the unaffected regime.

## Notes

`-ot` selects buffer type per whole tensor and cannot express partitioning; that remains the reason a loader materialization capability may eventually be required. #29963 changes the urgency because whole host tensors can instead be transiently staged to the GPU owning a pipeline split.

External references verified 2026-10-05:
- llama.cpp #29963: https://github.com/ggml-org/llama.cpp/pull/29963 (draft, opened 2026-10-04T20:54:59Z). Host weights are transient scheduler split inputs; fit logic balances layer slices and converts dense-only layers to full layers as VRAM permits.
- Fable MoE-offload fork mirror/measured description: https://github.com/MaxDam/llama.cpp.35B.moe ; reports pinned host weights + next-layer expert prefetch at ~1143 -> ~1880 pp2048 (+64%) on RTX 3060/Qwen3.6-35B-A3B. CUDA evidence only; use as an AMD experiment hypothesis.
- llama.cpp release baseline visible 2026-10-05: b11396 (`2e7c58c`, released 2026-10-04 17:45 UTC). #29963 is newer/open and is not baseline behavior.

## Change Log

- 2026-10-02T04:45:10.449925+00:00 (created-by): Created by agent
- 2026-10-05: Reframed runtime slicing behind a standard-GGUF qualification gate; added #29963 transient host-weight pipeline and pinned/prefetch controls; consolidated policy/transport ownership with MET01/MET05.
