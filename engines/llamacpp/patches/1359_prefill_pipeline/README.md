# 1359_prefill_pipeline

Not promoted. Mechanism is in SUMMARY.md: with an MTP drafter, prompt batch k+1 is submitted before the drafter
hook collects batch k's hidden states, which it then takes through a backend event that waits for batch k only.
Off by default; `BIGCHERRY_PREFILL_PIPELINE=1` turns it on.

## Evidence

- Gate (Flash-Next 24K, production profile): each target card has nothing queued for about 34 ms a 512-token batch
  (kernel trace `dp1`, `tools/lab/flash-next/kernel-gap-stats.py`); the host's one wait a batch is the hook's
  `llama_get_embeddings_nextn` (synchronisation trace `ps1`); without a drafter prefill is 14.6% faster (`nomtp1`).
- Mechanism outside the engine: `tools/lab/hip-probes`, test `pipeline`, run `hp6`.

## Still to show

Listed in SUMMARY.md. A changed greedy text rejects the patch.
