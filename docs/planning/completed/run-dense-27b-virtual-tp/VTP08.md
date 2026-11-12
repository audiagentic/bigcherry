---
id: VTP08
order: 7
plan: run-dense-27b-virtual-tp
state: superseded
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Strict native-W8A8 FFN plus FP8 KV physical VRAM lower bounds

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs (tools/lab/radiance/xtx27_*), tests and early patch drafts are in that tag. Superseded by VTP14 (native Radiance TP3, completed 2026-11-12), which kept these programs as reference oracles only.

## Why this is a separate gate

VTP02 uses synthetic 30-GiB weight and user-assumed overhead inputs
to explore virtual TP4 on two XTX and one R9700. VTP06 identifies a
real native W8A8+DFlash2 RAD2 artifact:
qwen3.8-27b-w8a8-df2.rad with full-file SHA-256
5a9af2dfaf3150e59cfc0001f8fe4f33549b201c6be19700c09c9e9209cc2701.

Pinned rad-info -v confirms 128 fused FFN tensors (64 gate/up, 64
down) encoded i8*bf16[1x128]/fwht128. The original BF16 source had
192 separate FFN gate/up/down tensors. Native FFN shapes are
34816x5120 and 5120x17408. Each weight element uses 1 INT8 code
byte plus a 2-byte BF16 scale for every 128 elements. The exact
unreplicated 64-layer FFN payload is 17,380,147,200 bytes =
16.1865234375 GiB, excluding RAD2 page padding. This matches the
rad-info rounded 16.187 GiB FFN category.

Full-attention FP8 KV payload: key plus value for 16 full-attention
layers, four KV heads, 256 head dimension and 1 byte per element =
32 KiB per aggregate resident token. The tentative four logical KV
heads are placed on physical GPUs 1/1/2, not proportional to FFN
weight ratios. GDN state is not counted.

## Implemented card-free lower-bound tool

Source: tools/lab/radiance/xtx27_rad_vram_floor.py
Tests: tools/tests/lab/test_xtx27_rad_vram_floor.py

The tool runs only rad-info -v, rejects non-RAD2 or non-native W8A8,
incorrect metadata and any missing, duplicate or nonconforming FFN
tensor, then derives legal 256-wide intermediate tiles and exact
code+scale and FP8 K/V bytes per physical GPU. It also computes the
hypothetical dual-XTX equal-FFN, two-KV-heads-per-XTX baseline.

Example read-only command:

    INFO=${BC_WORK_ROOT}/builds/f17f21fda44979e9abe84f13e96b8896/08b5dd81b4874390bf0cbf489db0d7df/bin/rad-info
    MODEL=${BC_MODEL_ROOT}/radiance/bigcherry/qwen3.8-27b-w8a8/qwen3.8-27b-w8a8-df2.rad
    python3 tools/lab/radiance/xtx27_rad_vram_floor.py --rad-info "$INFO" --model "$MODEL" --candidate capacity --streams 4 --tokens-per-stream 262144

Scratch receipts are in
${BC_WORK_ROOT}/xtx27-prefill/quant-shape/vtp08-capacity-4x256k-v2.json
and vtp08-capacity-8x256k.json. They identify model file path/size,
but do not rehash the 31-GB artifact on every invocation; VTP06 holds
that separate full-file fingerprint.

## Actual model-shaped capacity lower bounds

For the candidate 30/30/20/20: legal logical FFN widths are
[5120,5120,3584,3584]; physical XTX0, XTX1, R9700 widths are
[5120,5120,7168]. All numbers are GiB.

| Streams x length | XTX0 FFN + FP8 KV | XTX1 FFN + FP8 KV | R9700 FFN + FP8 KV | Strict result |
| --- | ---: | ---: | ---: | --- |
| 4 x 256K | 4.761 + 8 | 4.761 + 8 | 6.665 + 16 | Not disproved; NOT admitted |
| 6 x 256K | 4.761 + 12 | 4.761 + 12 | 6.665 + 24 | R9700 only 1.335 GiB remains |
| 8 x 256K | 4.761 + 16 | 4.761 + 16 | 6.665 + 32 | HARD NO-GO: R9700 requires 38.665 GiB |
| 8 x 128K | 4.761 + 8 | 4.761 + 8 | 6.665 + 16 | Same bound as 4 x 256K |

R9700 nominal physical VRAM is 32 GiB. At 4 x 256K its FFN plus
FP8 attention KV floor is 22.665 GiB, leaving at most 9.335 GiB
for every other weight family, scratch, graphs, GDN state, KV
allocator padding and all runtime buffers.

Pinned rad-info -v reports around 3.570 GiB embedding/head and
6.873 GiB other text. If those two families were fully replicated
and resident on R9700, their rounded ~10.443 GiB would exceed the
9.335-GiB remainder, without even counting vision/drafter/scratch.
This is a conditional failure, not proof a correctly sharded or
partially staged engine cannot work.

### Matching-precision dual-XTX TP2 lower bound

With equal W8A8 FFN halves and two FP8 KV heads on EACH XTX,
4 x 256K resident tokens require per card:

8.0933 GiB FFN + 16 GiB KV = 24.0933 GiB.

That exceeds nominal 24-GiB XTX capacity BEFORE non-FFN weights or
runtime. Thus this precise fully resident dual-XTX FP8 target is
impossible. It does not rule out fewer tokens, other KV formats,
host-spill or another model.

## Nonclaims and acceptance work

- A positive hard-floor margin is NEVER admission. This analysis
  excludes all non-FFN weights, GDN recurrent state, drafter, vision,
  activations, graph arenas, scratch, memory fragmentation, alignment,
  pinned-host transport, PCIe synchronization and replication.
- No physical/virtual TP4 implementation exists. Tensor ranks cannot
  be aliased just by choosing --tp 4; current engine bind and
  collective ownership require an architecture-level change.
- FP8-V4/K6-V4 packing is a separate format with a separate native
  correctness and allocator-capacity gate; do not presume a KV gain.
- Next: instrument a precise per-physical-GPU owner and allocation
  ledger for ALL 1101 native W8A8+DFlash2 RAD2 entries; distinguish
  the R9700's two logical FFN shards from actual physical VRAM.
  Report refusal if owner, replication, device code or real resident
  KV budget is unresolved, then benchmark the end-to-end three-way
  exact reduction versus the best genuinely feasible control.

## Change Log

- 2026-11-12T22:13:34.940667+00:00 (state-transition): State: in_progress → superseded
