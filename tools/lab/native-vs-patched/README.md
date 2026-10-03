# native-vs-patched

Reference-ladder comparison of a stock build against a BigCherry-patched build,
both `linux-multi` (gfx1100;gfx1201;gfx1030, RCCL on), with per-model
activation probes (`BIGCHERRY_PATCH_HIT`/`BIGCHERRY_PATCH_TRACE`, patched arm
only) recorded under `<out>/activation/`.

Run on Brutus: `run-ladder.sh <stock bin dir> <patched bin dir> <out dir>`.
Results are per (model, device set); each ladder is order-rotated. The activation
probe is a separate short llama-bench run, not the timed runs, so it shows a
patch path executes for that model/device, not that it executed in each timed run.
