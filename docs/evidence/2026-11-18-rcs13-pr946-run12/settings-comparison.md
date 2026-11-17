# Run 12 production-configuration isolation

This table contrasts the Run 11 direct `ServerRunner` manifest with the actual `flashnext-tp2xtx-r9700` production launcher and recent known-good VTP44 Flash-Next evidence. Values come from the preserved Run11 manifest and source, `tools/lab/radiance/launch-flashnext-best.sh`, `engines/radiance/layouts/flashnext.toml`, `tools/lab/radiance/lib/plan-check.sh`, the 2026-11-16 VTP44 Flash-Next helper A/B evidence, and Run12 measured v2 manifests. The host-pool value **is the measured/final** pool, not the 4,096 MiB layout initial setting.

| Setting | Run 11 direct serving | Production layout/VTP44 and Run12 measured | Gate relevance |
| --- | --- | --- | --- |
| Model and hardware | Flash-Next hcbf16; XTX×2 gfx1100; R9700 gfx1201 | Same | matched |
| Context maximum | 8,192 tokens | 65,536 tokens | different |
| Session maximum | 1 | 6 | different |
| Prefill `max_num_batched_tokens` | 256 | 2,048 | different |
| Helper `max_rows` | 64 | 4,096 | different |
| Helper `small_rows` | 8 | 8 | matched |
| Helper RAM budget in VRAM | 30,555 MiB | 29,656 MiB | different |
| Host pool | 32,005→32,301 MiB | 4,096→33,219 MiB | different measuring configuration |
| Host KV | 0 MiB | 0 MiB | matched |
| KV dtype | fp8 | fp8 | matched |
| Speculation | MTP, 3 tokens | MTP, 3 tokens | matched |
| N-gram storage | disk | disk (override of launcher's RAM default) | matched |
| Checkpoint interval | 4,096 tokens | 4,096 tokens | matched |
| Prefix resume | nearest | nearest | matched |
| TP / wire | 2 gfx1100 ranks, exact wire, P2P off | Same | matched |
| Heat | same tracked perf51 hot-first ranking | same tracked perf51 hot-first ranking | matched |
| Live route histogram | `/dev/null` | active rank histograms, every second | different |
| CPU/GPU admission, plan and serve | bespoke Python manifest + sidecar and `ServerRunner` | actual shell production launcher + strict v2 measured manifest + startup canary | materially different |
| GPU power | no production launcher power apply | VTP44 applied production power settings; Run12 explicitly disabled on currently unconfigured host | performance qualification caveat |
| Environment | direct `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `RADIANCE_HOME` and explicit work paths | launcher assigns BDF-visible devices; builds/helpers read manifest-owned settings; external `BC_WORK_ROOT`, `BC_MODEL_ROOT`, `BIGCHERRY_ENVIRONMENT`; no extra `R3_*` tuning | different orchestration |
| Live placement policy | no comparable calibration | layout head-only; production launcher records not_applicable and serves without live policy | recorded, not silently assumed |

When production power auto failed with `empty explicit request: no settings for any selected device`, the launcher had not started any GPU child and did not apply power. Run12 used its official `--no-gpu-power` flag consistently for A0/B1/B2/A3. The resource/manifest path was unchanged by this flag. Reproduction is **functional**, not proof that clocks or perf equal older power-controlled VTP44 runs.

Production faithful A0 passed the exact seed17 1,066-token prompt that triggered Run11's AQL queue exceptions and subsequent `hipMalloc(16KiB)` refusal; this isolates a setup difference rather than a conclusive main regression. To determine the precise responsible knob, follow-up can bisect one manifest setting at a time under a short lease. This was not necessary to decide Run12's PR #946 identity gate, which remains open because of two-stream output hash variation.
