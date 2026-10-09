# 1325_sched_split_timing

## Promotion record

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11474 / b9acf138, 2026-10-08). A diagnostic, promoted as a
neutral enabler: it prints only when its flag is set and changes nothing otherwise, so it can be used on the production
binary without a special build.

- Build and run: experiment build `b-metamem-mig-diag` compiled clean.
- Activation with `BIGCHERRY_SUBMIT_TIMING=1` (run `metamem-mig-diag`): 594 `BIGCHERRY_SCHED_SPLIT` lines (two splits
  per graph: the CPU split and the tensor-split backend, with input and compute host time each).
- Neutral: greedy text with the flag on equals the flag-off arm (md5 fe307bdfb7e1).
- Pre-bump use (b11402, runs `metamem-pd-mtp` / `metamem-pd-plain`): it showed where the wait for the GPUs sits - 285.5
  ms in the tensor split's input stage without a draft, 11.8 ms with MTP (the wait moves into the NextN fetch).

## Native llama.cpp comparison

Native llama.cpp has no counterpart: the patch only adds reporting behind its flag. With the flag unset the build behaves as without the patch.
