# Serving manifest coverage plan (RSC12)

Goal (owner, 2026-11-07): the manifest expresses every configuration the engine has, whether or not it is also available as a flag, so a start from a manifest loses nothing. Flags stay accepted.

## What the manifest covers today

- The engine has 118 command-line options (pinned table, commit 4a61d72); **26** are bound to a manifest field. The other **89** are listed below by group (`-v`, `-vv` and `--help` are not configuration).
- Configuration that is **not an argument**: the engine and its components read about 91 environment variables of their own (counted from the radiance source and this repository, leaving out HOME, PATH and the like: RADIANCE_* 34, R3_* 31, RAD_* 17, R4D_* 6, BC_* 1, CALIB_* 1, GPU_* 1), plus files (route-heat, calibration store, live-policy and experiment objects) the manifest already points at or embeds. The helper settings the engine reads from the environment are the part RSC08 S5 left open (`R3_MOE_WORKER=1` and its siblings).

## Options with no manifest field

### HTTP server and limits (13)

`--api-key` `--host` `--http-threads` `--keep-alive-timeout` `--max-body-mib` `--max-queued-requests` `--max-stop-bytes` `--max-stop-strings` `--no-cors` `--read-timeout` `--retry-after` `--served-model-name` `--write-timeout`

### placement, prefill and tensor-parallel tuning (22)

`--attn-prefill-fp8` `--busy-prefill-chunk` `--checkpoint-policy` `--draft-coupling` `--draft-expert-placement` `--draft-model` `--draft-short-from` `--draft-window` `--embedding-placement` `--expert-pin-top` `--expert-profile` `--expert-stats` `--live` `--p2p` `--prefill-interleave` `--prefill-span` `--prefix-cache-ttl` `--state-cache-dtype` `--tp-attn-split` `--tp-wire-min-kb` `--weight-prefetch` `--weights-disk-tier`

### chat template and reasoning (6)

`--chat-template-kwargs` `--override-chat-template` `--reasoning-effort` `--reasoning-effort-map` `--reasoning-format` `--thinking-token-budget`

### LoRA (6)

`--control-vector` `--lora` `--lora-dir` `--lora-target-modules` `--max-lora-rank` `--max-loras`

### debug and diagnostics (2)

`--debug-accept-reference-kernels` `--debug-placement`

### run mode (stay flags, not configuration) (8)

`--debug-graph` `--download-dir` `--kld-corpus` `--kld-lora` `--kld-lora-sources` `--kld-out` `--kld-record` `--kld-ref`

### sampling defaults (20)

`--default-max-tokens` `--dry-allowed-length` `--dry-base` `--dry-multiplier` `--dry-penalty-last-n` `--dry-sequence-breakers` `--frequency-penalty` `--generation-config` `--max-n` `--max-tokens-cap` `--min-p` `--presence-penalty` `--repeat-last-n` `--repetition-penalty` `--temp` `--top-k` `--top-p` `--typical-p` `--xtc-probability` `--xtc-threshold`

### vision and video (12)

`--image-max-pixels` `--image-min-pixels` `--image-tokens` `--max-source-pixels` `--mm-max-patches` `--video-fps` `--video-max-frame-tokens` `--video-max-frames` `--video-max-pixels` `--video-min-frames` `--video-min-pixels` `--vision-tower-placement`

## Plan

**Status 2026-11-07: slice 1 is done** (80 option fields, contract amendment 13, spec §11); rad0009 needed no change because it turns every registry-bound field into its flag. Slice 2 (environment settings) has started: amendment 14 adds `spec.processes.engine.kernels` and the `env_setting` binding the engine applies at start; further groups follow. Slice 3 remains.

This reopens the v1 contract (an additive amendment 13, as amendments 10-12 were: new optional fields, regenerated schema, types, codec and fingerprints). Slices, each its own PR:

1. **Option fields** (engine side as flags): server and limits, sampling defaults, chat template and reasoning, placement/prefill/tensor-parallel tuning, LoRA, vision and video. Each group becomes an optional section under `spec.processes.engine`, bound to its option by the registry (`radiance_option`), so `bc-serve --set` writes it and rad0009 turns it into the flag. rad0009 already turns any registry-bound field into its flag, so it needs no change and keeps its qualification.
2. **Environment settings as fields**: the engine- and component-side environment variables become typed manifest fields the engine applies at start (setenv before initialisation), starting with the helper-hook switch and its siblings (RSC08 S5), then the debug/diagnostic group. Also a rad0009 change.
3. **Retire the side channels**: `layouts.toml`/`recipes.toml` `server-args`, the lab scripts that still pass flags, and the engine refusing a direct flag that has a field (RSC11 C).

Not configuration, so they stay flags beside a manifest: the run-mode group above (KL run flags, `--download-dir`, `--debug-graph`), `--role`, `--plan-memory`, `-v`/`-vv`.

Each slice is gated on the patch requalification and the card time that needs; nothing here changes engine behaviour for a start without `--manifest`.
