# Offline expert-placement feasibility spike

Plan item: untracked exploratory spike
Status: active
Owner: BigCherry
Question state: open

## Question

Can BigCherry pre-compute a reusable, model-fingerprinted MoE expert placement map before llama.cpp gains runtime expert-parallel execution, and can routing traces be replayed offline to estimate device participation and transport cost?

## Scope

This experiment is deliberately runtime-inert. It does **not** patch `build_moe_ffn`, change GGUF files, move experts at runtime, or create new RCCL communicators.

It provides:

- exact GGUF tensor/expert inventory using llama.cpp `gguf-py`;
- per-layer packed-expert byte accounting from real tensor `n_bytes`;
- PLE/Engram and named-MTP byte classification;
- deterministic static expert-home compilation from explicit per-device expert budgets;
- model-layout fingerprinting so stale maps fail validation;
- global expert ID -> device/local-slot maps suitable for a future llama.cpp executor;
- offline routing-trace replay to estimate how often auxiliary expert stores are touched and the minimum activation/result traffic implied by a map.

## Inputs

1. GGUF model file.
2. Topology JSON. `topology.brutus.example.json` is an illustrative starting point; its expert budgets are **not** a validated performance recommendation. Replace them after inventorying the actual model.
3. Optional routing JSONL, one record per routed layer evaluation:

```json
{"layer": 12, "experts": [4, 18, 91, 103, 122, 170, 255, 301, 411, 508], "tokens": 1}
```

`tokens` defaults to 1. For MTP verification traces it may be 2-8.

## Runtime

GPU required: no
Real compilation required: no
Mutates canonical BigCherry state: no

Inventory requires llama.cpp's existing `gguf-py` reader. No package download is required if a llama.cpp checkout is already available:

```bash
export PYTHONPATH=/path/to/llama.cpp/gguf-py
```

## Usage

Set an artifact directory:

```bash
OUT=artifacts/lab/expert-placement-offline/qwen38-iq4xs
mkdir -p "$OUT"
```

Inventory the actual GGUF:

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py \
python3 tools/lab/expert-placement-offline/expert_plan.py inventory \
  --gguf /models/Qwen3.8-Flash-Next-UD-IQ4_XS.gguf \
  --out "$OUT/inventory.json"
```

For strict content identity add `--full-sha256`. The default layout fingerprint is much faster on a ~94 GB model and covers file size plus every tensor name/type/shape/size/data offset.

Compile a static placement:

```bash
python3 tools/lab/expert-placement-offline/expert_plan.py plan \
  --inventory "$OUT/inventory.json" \
  --topology tools/lab/expert-placement-offline/topology.brutus.example.json \
  --out "$OUT/expert-map.json"
```

Validate it independently:

```bash
python3 tools/lab/expert-placement-offline/expert_plan.py validate \
  --inventory "$OUT/inventory.json" \
  --topology tools/lab/expert-placement-offline/topology.brutus.example.json \
  --map "$OUT/expert-map.json"
```

Replay a routing trace. Qwen3.8 hidden width 2560 with FP16 activation/result is 5120 bytes per token, so `5120` is the useful first model for minimum payload traffic:

```bash
python3 tools/lab/expert-placement-offline/expert_plan.py simulate \
  --map "$OUT/expert-map.json" \
  --trace "$OUT/routing.jsonl" \
  --activation-bytes 5120 \
  --out "$OUT/simulation.json"
```

Run the lab tests:

```bash
python3 -m unittest -v tools/lab/expert-placement-offline/test_expert_plan.py
```

## Planner semantics

The first policy is intentionally simple: `static-budget-weighted`.

- `expert_budget_bytes` is the maximum permanent expert storage assigned to a device. It must already account for target weights, MTP, KV, graph buffers, and any desired safety reserve.
- `placement_weight` biases allocations toward or away from a device without violating the hard budget.
- every `(layer, global_expert_id)` has exactly one permanent owner;
- local slots are contiguous per `(device, layer)` and are ready for a future compact `MUL_MAT_ID` store;
- `primary_tensor_group` describes the devices considered the dense/tensor core by the simulator; those members are not counted as remote expert transport;
- transport entries are descriptive constraints for the future executor. This offline compiler does not perform transport.

The initial policy is a capacity/cost baseline, not the final routing policy. Later candidates can consume the same schema and add trace-weighted placement or XTX LRU replicas without changing the GGUF.

## Expected next evidence

Run inventory first and record:

- exact routed-expert bytes;
- exact PLE/Engram bytes;
- exact non-expert bytes;
- per-layer bytes per expert;
- whether a 96 GB VRAM placement has sufficient expert budgets after target/MTP/KV reserves.

Then capture a representative top-k routing trace from stock/diagnostic llama.cpp and replay multiple candidate maps. The key go/no-go metrics before runtime implementation are:

- fraction of routed layer evaluations touching R9700;
- fraction touching the chipset RX 6900 XT;
- number of auxiliary stores touched per layer evaluation;
- estimated minimum bytes over each remote path;
- sensitivity to MTP verification batch width.

## Safety

- Canonical-state mutation: none.
- No RCCL communicator is created.
- The RX 6900 XT remains a descriptive expert-only store in the example topology and is excluded from the primary RCCL tensor group.
- Do not import this lab experiment from production `bigcherry` modules or maintained analysis.
- Generated outputs belong under `artifacts/lab/expert-placement-offline/`.

## Disposition

If the offline evidence supports the design, graduate the stable sidecar/map contract to maintained code and separately implement a gated llama.cpp expert-placement executor. If the model does not fit or simulated participation/transport is structurally unattractive, retain the evidence and discard the runtime implementation idea.
