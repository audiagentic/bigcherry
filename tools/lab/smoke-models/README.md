# smoke-models (MEN15)

Fetches and converts the tiny models the RAM-only engine smoke (`bigcherry --engine <name> smoke-cpu`) serves on the
CPU. Their sha256 digests are pinned in each `engines/<name>/engine.toml` `[smoke.cpu]`; re-pin when a model changes.

- `fetch-tiny.sh` - `<model-root>/smoke/tiny-llama.gguf` (llama.cpp: stories15M, llama.cpp's CI model) and
  `<model-root>/smoke/tiny.rad` (radiance: Llama-68M converted with `rad-convert`, fp8 linears in 128x128 blocks).

Disposal: keep while the smoke models are produced by hand; fold into a `bigcherry` command if they are fetched often.
