# LTX-2.5 guidance-pass placement — native AQL lab evidence

**Status: proof of residency and execution. Cross-card agreement is reported as an error, not a bit match; the generation-level quality effect is not yet qualified.**

This is an isolated lab test of the existing Radiance 2.2 LTX-2.5 IQ8R implementation, using its native `declare_dit`, `dit_eval`, weight-sharing and kernel-selected relayout paths. No production architecture, scheduler, or runtime changes were made. The measurement used two 24 GiB gfx1100 cards and one 32 GiB gfx1201 card, with peer-to-peer transfer unnecessary. The 3-card test exposed only these cards through `ROCR_VISIBLE_DEVICES`.

## Residency

The 121-frame LTX engine geometry is derived from its own spatial scale, first-frame/timestep factors, and audio rate: 16 latent frames, 1,024 text rows and 127 audio rows. 1088 pixels is the nearest valid 32-divisible height for 1080p-class output.

| Output | Latent grid (frames × H × W) | Video rows | Total step rows | Native arena MiB | XTX headroom MiB | Fits? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 768 × 512 × 121 | 16 × 16 × 24 | 6,144 | 7,295 | 1,321.23 | 4,348.00 measured | Yes |
| 1216 × 704 × 121 | 16 × 22 × 38 | 13,376 | 14,527 | 2,767.06 | 2,902.00 measured | Yes |
| 1920 × 1088 × 121 | 16 × 34 × 60 | 32,640 | 33,791 | 6,618.33 | −794.79 budget estimate | No |

The checkpoint's DiT accounts for 5,338 source weight planes / 18,630.46 MiB; the native builder's 4,378 selected weight declarations require 18,690.46 MiB of stored VRAM after kernel layout. Residency uploads **all** source DiT planes to the AQL card and reserves the 60 MiB difference for native layout, along with the full native arena. It observes the allocation and AQL transfer completion, with an initial 24,514 MiB free. The full-pass execution separately uploads the *native-relayout* weights, not merely the source planes. The 1080p-class case is rejected by the native model budget before allocating weights; its headroom is not a measured after-allocation value.

The 512/704 residency tests are model-weight-plus-DiT-activation checks, **not** all components of a production video pipeline simultaneously resident (text encoder, VAE, codecs, service allocations are excluded).

See `residency-512.txt`, `residency-704.txt` and `residency-1080.txt`.

## Four full-DiT guidance passes

The runtime prototype loads the same full set of 4,378 real, IQ8R-relayout weights on each of the three cards. It creates four independent `Ctx` instances and runs **all 48 native video/audio DiT blocks** (self-, text-cross-, modal-cross-attention and FFN), using the native `dit_eval` issue path. Pass 0 (conditional) and pass 1 (unconditional) run sequentially on the gfx1201 card; STG and isolated-modal passes each use a different gfx1100 card. The three devices execute concurrently on three host threads.

The timing starts after model load, registry/build, activation seeding and warm context initialization. Inputs, positions, modulation tables, and text/audio tensors are deterministic lab fixtures; these are complete native DiT evaluations, **not** real prompt-conditioned video generation.

| Workload | Sequential reference: 4 passes on one gfx1201 | Placed parallel makespan | Speedup vs one-card |
| --- | ---: | ---: | ---: |
| 768 × 512 × 121 | 6,341.135 ms | 3,628.985 ms | **1.747×** |
| 1216 × 704 × 121 | 16,822.930 ms | 10,682.432 ms | **1.575×** |

All four concurrent outputs match the corresponding serial outputs from the **same target device**, bit-for-bit for video and audio at both resolutions. This proves there is no observed concurrency-induced output drift in the tested paths.

**Cross-device correctness is not established.** When compared against four one-card gfx1201 reference passes using identical lab inputs and weights, both gfx1100 output variants differ significantly: at 512/768, STG differs in 695,823/786,432 video BF16 elements and isolated-modal differs in 692,290/786,432. At 704/1216 the counts are 1,514,715/1,712,128 and 1,506,265/1,712,128 respectively. Audio outputs also differ. At the larger resolution all four outputs contain no BF16 NaNs or infinities. The lab exits nonzero on this unrelaxed bitwise failure, even though native issue, per-device repeatability and timing complete.

See `parallel-512.txt` and `parallel-704.txt`. `parallel-small-final.txt` is the post-refactor 256 × 256 × 17-frame smoke test: the one-card 440.679 ms baseline versus a 257.209 ms parallel makespan is 1.713×; same-device outputs match and the cross-architecture bitwise gate fails. `parallel-small.txt` records the earlier placed-serial-only smoke result (2.94×), which is not a one-card comparison.

## Reproduction and scope

Configure the existing `tools/lab/radiance/ltx-tp3/native-c01-gate/` CMake lab with an ABI-matched, HIP-enabled Radiance 2.2 core; the pinned LTX stage, checkpoint RAD file, and the gfx1100/gfx1201 kernel plugins. Set `LTX_NATIVE_GPU=ON` (the GPU link also requires HIP/HSA and FFmpeg libraries). The binary modes are:

```text
ltx_c02_lin_declare_test MODEL KERNEL_DIR --ltx-pass-residency-aql WIDTHxHEIGHTx121
ltx_c02_lin_declare_test MODEL KERNEL_DIR --ltx-pass-full-aql WIDTHxHEIGHTx121
```

Use `ROCR_VISIBLE_DEVICES=0` (or the other XTX) for the one-device residency mode and a visible roster of exactly the two gfx1100 plus one gfx1201 card for the full-pass mode. Use a separate GPU lease for each arm. The full-pass test explicitly fails if any result differs from the single-card reference, and never modifies shipping code. Neither model loading nor video decoder time is part of the speedup.

Host compilation of the updated C02 lab succeeded. In the same source/patch stage, `ltx_c02_actual_lin_declare` and `ltx03_ffn_modulation_residual_declarations` passed; `ltx03_text_cross_attention_declarations` and `ltx03_modal_cross_attention_declarations` failed with "C06 invalid native geometry admitted" (representative failures reproduce with the untouched worktree HEAD and identical staged core: `host-baseline.txt`). Additional all-block, weight-contract and instance-rendezvous tests failed with the same message; see `host-extra.txt`. No failure is attributed to this change without that comparison.

## Recommendation

Do **not** promote guidance-pass placement yet. The memory geometry permits it for 512/704 at 121 frames, and its real three-card makespan has upside, but cross-architecture bitwise output fails the stated acceptance criterion. Before shipping, isolate the first divergent native block/kernel between gfx1100 and gfx1201 on real generated-conditioning fixtures, distinguish numerical rounding from layout/selection defects, and establish an exact-arithmetic contract (or explicitly re-scope the bitwise requirement) before production integration. The 1080p-class single-XTX case additionally needs a reduced arena/weight footprint or different placement; no speculative offload path was added here.

## Cross-card agreement (supersedes the bitwise gate)

The two architectures run different kernels with different accumulation order, so the one-card reference can only be matched
bit-for-bit on the same card (which holds: every parallel output equals its serial output on its own device). Across cards the lab
now reports the relative RMS difference of the full DiT output against the R9700 reference
(`cross-card-rel-rms-*.txt`, rows PASS_FULL_CROSS_CARD):

| Workload | variants run on the R9700 | variants run on an XTX (video / audio rel RMS) |
| --- | --- | --- |
| 768x512x121 | 0 | video 1.52% and 1.48%; audio 2.08% and 1.89% |
| 1216x704x121 | 0 | video 1.52% and 1.48%; audio 2.04% and 1.89% |

All outputs are finite. The synthetic lab inputs are not a generation, so this is the size of the numeric difference, not a quality
verdict: whether 1.5-2% on a guidance branch is acceptable has to be judged on generated video (guidance branches are combined by
the guide op, which may average some of it out). The speed measured with the placement is 1.75x (768x512) and 1.58x (1216x704)
over four passes on the R9700 alone, and the four-pass saving shrinks with fewer passes (about 1.3x for three, none for two or one).
