# LTX-2.5 TP3 card-free reference (LT3 foundation)

Owner: future native LTX architecture / upstream migration track, captured in docs/design/ltx25-iq8r-tp3.md. This is a **lab-only**, self-contained C++20 reference, **not** a production engine component, runtime plugin or model implementation.

## Question

Before changing upstream Radiance's LTX TP1 refusal, can three uneven ranks preserve all 32 video/audio attention heads, reconstruct fused QKV bank slices, reject unaligned IQ8R K cuts, normalize Q/K with one global RMS denominator, and enforce a disjoint four-device role/memory sketch? Can we bound the shape of the Gemma-to-connector transfer correctly?

## Inputs, outputs and requirements

Inputs are synthetic fixed-seed values and the *source-declared* LTX-2.5 geometries (32 x 128 video heads, 32 x 64 audio heads; Gemma 3840 channels and 48 layers; maximum two 1024-row prompts), not checkpoint weights. The reference includes IQ8R alignment constraints (128-K FWHT, 256-K scales) but **does not parse or execute real IQ8R files**.

The separate ltx_tp3_protocol.h and ltx_tp3_protocol_test.cpp also test identical collective schedules for conditional, STG and isolated guidance, leader-only non-DiT writes, bias-once reductions, and an in-memory conditioning mailbox that refuses stale epochs, mismatched shapes, overwrites and consumed/cancelled requests. The mailbox is intentionally not a GPU transfer or production wire format.

The ltx_tp3_numerical_test.cpp independently compares multi-row dense self-attention and cross-attention with three uneven rank slices across six cases: fused projection banks, global Q/K RMS, per-head softmax and 2*sigmoid gate, and bias-once output reduction. Uses f64 tiny data and omits RoPE, IQ8R encoding and BF16 rounding; it does not establish real GPU parity. Shape/transfer mock arithmetic now refuses integer overflow.

Outputs are process exit status and the CTest result. No generated evidence receipts or recordings are claimed. Requires a C++20 compiler and CMake 3.21+; no ROCm/GPU/model/network access. The reusable header deliberately has no imported production entrypoint, and this lab must not appear in an engine release.

From the repository root, for a direct native CTest run:

    cmake -S tools/lab/radiance/ltx-tp3 -B "$BC_WORK_ROOT/ltx-tp3-reference" -DBUILD_TESTING=ON
    cmake --build "$BC_WORK_ROOT/ltx-tp3-reference"
    cmake --build "$BC_WORK_ROOT/ltx-tp3-reference" --target test

For the standard tooling test suite (Python test harness only, never engine runtime):

    PYTHONPATH=tools python3 -m pytest tools/tests/lab/test_ltx_tp3_reference.py -q

The model-aware numeric tool also audits all 48 DiT blocks: 960 required IQ8R projection entries share expected shapes; audio 10/10/12 would break scale-group K cuts for four output projections per block (192 invalid). Audio FFN-down remains aligned because its K width is four times the audio head-inner width. This is an admission oracle only, not kernel or TP3 execution coverage.

## Mutation/safety

Writes only the explicitly selected CMake build directory or pytest temporary directory. Does not touch GPUs, allocate model weights, mutate engine source, weaken the TP1 guard, or hold a hardware lease. No background processes. A mock 16-GiB gfx1030 memory budget is not a residency proof: the real Gemma workspace, allocator and connector placement must be measured separately.

## Disposition

Concluded experiment, kept as a lab (see the section below): the card-free reference and the model-aware gate remain the
executable record of the TP3 design and its rejection. Do not promote to shared tooling or production scheduling. Delete only through
the tooling disposition/retirement process; preserve it while the design or the evidence cites it.

## Concluded: tensor parallel over host-staged reductions is rejected for LTX (2026-11-23)

Status: **concluded, rejected as a speed lever; retained as a lab and a measured reference.** This lab and
`native-c01-gate/` hold the whole TP3 experiment. Nothing from it is promoted to production scheduling.

Question answered: can LTX-2.5 IQ8R run faster over two or three cards by splitting each layer (tensor parallel) when
the cards have no peer-to-peer path and every cross-card reduction goes through pinned host memory?

Answer: **no**. One video block at 2048 tokens takes 6.68 ms on the R9700 and 13.05 ms on one RX 7900 XTX. Two XTX with an
exact all-reduce take 19.7 ms (0.67x one XTX), three cards about 21 ms. Each reduction moves the full activation
(about 16.8 MB at 2048 tokens, about 260 MB at the largest step) twice per block, about 3 ms each, as long as the block's own
compute. Rejected variants, all measured: pipelining GEMM against the reduction (no gain at two ranks), the `ts_*` split
axes (only the three-rank exact path), a 6-bit rotated wire (parity with one card at 5-7x the exact error). The R9700 alone
beats every tensor-parallel arm. Evidence: `docs/evidence/2026-11-22-ltx03-issued-block-timing/` (timings, kbench rows, int8
reference).

What replaced it: guidance-pass parallelism (each card runs whole passes, no per-layer reduction), measured in
`docs/evidence/2026-11-23-ltx-pass-parallel-contention/` and `docs/evidence/2026-11-23-ltx-pass-placement/`
(`native-c01-gate/ltx_pass_residency.h`, `ltx_pass_full.h`).

What this experiment built and where it lives (each step is on main, with its PR):

| Step | What it proved | Lives in |
| --- | --- | --- |
| LT3 reference, C01 weight receipt (#1297) | uneven three-rank slices keep scale-group K cuts and head ownership | this directory, `native-c01-gate/ltx_c01_weight_receipt_test.cpp` |
| C02 `Lin::declare`, C06 builder (#1306, #1309, #1301, #1315) | the patched TP linear declare and native weight shares/relayout match the checkpoint | patch `rad0060_ltx2_tp3_linear_declare`, `ltx_c02_lin_declare_test.cpp` |
| C05 all-head QK RMS (#1296), C07/C08 exact reduction, mixed-card AQL (#1262) | exact collective across mixed cards works | `ltx_c05_issued_test.cpp`, `ltx_native_aql_roster.h` |
| Issued attention at 16/64 rows, all 48 blocks (#1398, #1396), numeric check across XTX and R9700 (#1321) | declarations hold on every block; the two architectures differ numerically | `ltx_c02_lin_declare_test.cpp` |
| Text and cross-modal attention, FFN and modulation declarations (`d98914ba9`), issued FFN (#1414) | cross-attention, FFN and residual shard correctly | same |
| Pipelined FFN with reductions, block timing (#1416, #1422) | the timings above | same, evidence directory |

Would reopen the question: a peer-to-peer or other fast cross-card link (the reduction would drop well below the block's
compute), a reduction that sends far fewer bytes without the lossy wire's error, or models whose tensors per layer are
small. If tensor parallel is revisited, the all-head QK norm needs only a reduced sum of squares, not the full Q/K (prior-art
scan), and the 6-bit wire must be re-qualified on generated video, not on a block.

The abandoned worktree tips for this work were not kept as archives: their content is merged above, so the record is
this section, the PR numbers and the evidence directories.

## Optional real-model validation (separate test tooling)

The corresponding Python test tooling can inspect the actual local IQ8R container with bounded disk reads, without running Radiance or modifying production: `tools/tests/lab/test_ltx_iq8r_planes.py` checks every declared IQ8R plane extent and `tools/tests/lab/test_ltx_iq8r_numeric.py` performs sampled row-level BF16/U8/I8/FWHT128 dense-versus-TP3 arithmetic. Both accept an explicit model-file positional argument and have CPU-only synthetic pytest fixtures. Evidence and limitations: `docs/design/ltx25-iq8r-model-plane-evidence.md`. The former's pure header/plane validation and latter's f64 oracle do not qualify the native HIP kernel. A 10/10/12 audio **K** split is forbidden at group 256 even though it owns whole attention heads.

## Per-component harnesses and promotion gates

The lab now owns **eleven independently selectable native CTests**: `ltx_tp3_reference_test`, `ltx_tp3_protocol_test`, `ltx_tp3_numerical_test`, `ltx_tp3_linear_test`, `ltx_tp3_rms_test`, `ltx_tp3_handoff_test`, `ltx_tp3_guidance_test`, `ltx_tp3_output_test` and `ltx_tp3_micro_pipeline_test` and `ltx_tp3_i8a8h_test`, plus `ltx_tp3_builder_shard_test`. The two additional real-container Python oracles inspect model planes, sampled quantized rows, all 48 DiT block layouts and both eight-layer connector stacks. For the exact matrix of component pass/fail gates, device promotion steps and end-to-end acceptance see `docs/design/ltx25-tp3-component-test-ladder.md`. The original component-suite native tests are synthetic. The separate C01/C02 gate invokes the real Radiance native builder and patched LTX source with the read-only model, but does not execute a full engine or GPU.

## IQ8R activation and real-plane C01 gate

`ltx_tp3_i8a8h_test` independently models the native `had_quant_act_i8g` math on BF16-rounded input: separate H128 rotations of each 256-K group, **one shared group scale**, nearest-even int8 quantization, invalid K cuts and three-rank code/scale equality. It does not call the kernel.

`tools/tests/lab/test_ltx_iq8r_shards.py` is another optional real-file test with explicit-path CLI, verifying logical raw-plane K slicing and fused QKV/KV bank offsets. It checks the presence or absence of checkpoint biases for all 48 blocks and refuses the bad audio 10/10/12 output split. See `docs/design/ltx25-tp3-component-test-ladder.md` for the exact model counts, negative controls and E2E promotion gates. These tests do not ship with the engine.

## C01 native builder diagnostic patch (2026-11-20)

The 2.2 integration now adds a read-only, real-model native RadBuilder
weight-view receipt for all DiT blocks; see
`docs/evidence/2026-11-21-ltx25-2p2-native-builder/README.md`.


The new Radiance patch package `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare` adds an optional `C01Slice` argument to the real LTX `Lin::declare` builder. Its native `ltx_tp3_builder_shard_test` includes the exact packaged header. `test_ltx_tp3_builder_patch.py` checks anchored applicability, idempotence and failure on changed upstream source. `native-c01-gate/CMakeLists.txt` compiles the patched frozen 2.2 LTX translation unit using its ABI44 headers in an isolated work-root stage. The production one-rank path is unchanged; no TP3 serving admission or GPU code is activated. See `docs/design/ltx25-c01-native-builder-stage.md`.


## C02 real LTX native Lin::declare (2026-11-22)

The opt-in native-c01-gate now builds the actual patched 2.2/ABI44 LTX
translation unit and a separate test **including that same source**. The
C02 test calls the real `ltx2::Lin::declare` with a read-only native
`RadFile`, the matching `RadBuilder`, and exactly one reference kernel.
It verifies **3,024 positive declarations** across every DiT block, seven
IQ8R projection types, three rank layouts and all ranks, with exact code,
scale and bias spans, host kernel resolution, and `Builder::finish()`.
Four negative cases (wrong scale group, biased output without deferral,
invalid audio K cut, deferral without a source) refuse before any model
weight is declared. The reference kernel is
staged alone into the CMake build root, avoiding unrelated SDK plugins.

Set `LTX_SOURCE_ROOT`, `LTX_STAGE_ROOT`, `LTX_NATIVE_CORE_LIB`,
`LTX_REFERENCE_LIB` and `BC_LTX_IQ8R_MODEL` to the exact 2.2/ABI44 source,
isolated patched source, matching core archive, matching `libref.so`, and
explicit model path. CMake refuses a core with a different Radiance
version or source root. Omit `LTX_REFERENCE_LIB` to use the original
compile-only gate. Full commands and limitations:
`docs/evidence/2026-11-22-ltx25-main22-c02/README.md`.

**Not proven:** model payload loading, GPU kernels, TP3 full model startup,
global Q/K RMS, collectives, bias-once reduction or video generation.


## C04: model-backed native output reduction (2026-11-22)

The diagnostic Lin finalizer reuses Radiance 2.2 exact TP3 all_reduce,
then applies BF16 bias once on each full reduced output replica.
The native-c01-gate now has three model-backed CTests including C04.
The C04 test verifies 2,016 legal projection declarations across all
48 blocks and both legal splits, but does not run a GPU collective.
See docs/evidence/2026-11-22-ltx25-c04-native-reduce/README.md.

## C05 native all-head RMS graph (2026-11-22)

The real-source native-c01-gate CMake project now registers
ltx_c05_global_qk_rms_graph next to C01, C02 and C04. It calls the
**actual patched LTX architecture** through an opt-in diagnostic,
reuses native Radiance fill, cast, exact all_reduce and
full-width rmsnorm, and checks two uneven layouts across all DiT
blocks and six projection types with the read-only IQ8R model.
The existing ltx_tp3_rms_test additionally checks zero-packing,
disjoint-rank reduction and local extraction against the dense
and compact-statistic reference. No production engine call site or
GPU path was enabled. See
docs/evidence/2026-11-22-ltx25-c05-native-global-rms/README.md.

## C06 real LTX Attn builder (2026-11-22)

The same native-c01-gate executable registers
ltx_c06_real_self_attention_graph. It exercises the real Attn::declare
with checkpoint IQ8R QKV shards, input/output activation quantizers,
C05 Q/K norm, native RoPE/head gate/attention and C04 output reduction.
All 48 blocks, video/audio, two legal head splits and three ranks are
covered (576 successful host-only graphs). No separate runtime exists.
See docs/evidence/2026-11-22-ltx25-c06-native-self-attn/README.md.

## C07 real native exact three-rank GPU collective boundary (2026-11-22)

The RUM13-owned `tools/lab/radiance/native-mixed-ar/` lab uses the actual Radiance 2.2 ABI44 exact BF16
host-mailbox all-reduce from libr3/libr4d and the real BigCherry shared-slot
service. Seven video/audio Q/K/output patterns at C06 geometry are checked per
rank. GREEN: three ranks on the R9700 (21/21 exact comparisons) and across two
XTX cards using libr3 (21/21). RED: adding a mixed gfx1100/gfx1201 participant
causes SIGSEGV at the first native collective kernel launch, after all three
native rank initializations succeeded. No production TP3 execution or TP1
parity is enabled. Evidence and commands:
docs/evidence/2026-11-22-ltx25-c07-native-mixed-ar/README.md


## C08 Radiance AQL three-card collective (2026-11-22)

The follow-up links the HIP-enabled Radiance 2.2 core as a whole archive
and executes both native exact BF16 three-rank host-mailbox rows through
Radiance's actual AQL streams, HIP fat-binary registration and collective
launch fencing. With two XTX (gfx1100) cards plus an R9700 (gfx1201),
one-shot and two-shot both passed 33/33 exact output checks, including
16- and 64-row video/audio cases. The C07 direct-HIP crash does not
reproduce on the actual AQL device backend. Native full self-attention,
TP1 model parity, and production admission remain open. Evidence:
docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md

## LTX02 C05 native executed graph, host and AQL

The sole `native-c01-gate/ltx_c05_issued_test.cpp` test source
reuses the patched native LTX2 C05 Q/K global-RMS graph,
native `RadBuilder`/`RadCtx`, and the real checkpoint BF16
Q/K gains. The host CTest `ltx_c05_issued_host_global_qk`
is card-free. The optional native
`ltx_c05_issued_aql_test` target is available only when
`LTX_HIP_NATIVE_CORE_LIB` identifies the matching
HIP-enabled Radiance 2.2 archive. It must **never** be
run automatically in CTest or without a GPU lease.

Under AQL the builder finishes all three ranks concurrently,
as native exact collective initialization itself
rendezvouses. The same source has passed **2,088,960 exact
BF16 comparisons** over both modalities, 1/4/16/64 rows,
two legal head partitions and both normal/reversed gfx11/gfx12
ordinal layouts. See
[the numerical evidence](../../../../docs/evidence/2026-11-22-ltx25-ltx02-issued-aql-rms/README.md).
This does not qualify a full C06 block or serving admission.
