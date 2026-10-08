# rd33 (1241) fixed-work plain decode vs control (no MTP), Qwen3.8-27B-Q8_0, dual 7900 XTX, -sm tensor

Script: tools/lab/native-vs-patched/decode-divergence.py. Greedy (temp 0, top_k 1), 4 prompts x 2 reps, 256 tokens, sequential arms.
Builds: control 7f701510/80a35ed7, rd33 89722ca1/081c8574 (bigcherry:stock:linux-multi, --experiment rd33-only).

| | control | rd33 |
|---|---|---|
| mean plain decode tok/s (8 runs each) | 33.06 | 34.67 (+4.9%) |

Per-run speed ratio is stable (32.95-33.13 vs 34.59-34.72): rd33 is faster in every run.
Numerics: rd33 removes Q8_1 activation quantisation, so it is not bit-identical to control.
- Greedy output diverged in 2 of 4 prompts (first divergent token 4 and 197); the other two are identical over the full length.
- Max |dlogprob| of the sampled token over the common prefix: 0.017 .. 0.075. Far above the 5e-4 gate for a bit-close claim.
- Reps are identical to each other (deterministic).

Reading: real fixed-work speedup on plain decode. The divergence is expected from removing activation quantisation and is not by itself
evidence of degradation (rd33's activations are unquantised), but no quality measurement against a reference exists yet.
Not established: output quality vs a high-precision reference (needs perplexity/KL vs an unquantised-activation reference), MTP-lane effect
(acceptance changes 0.901 -> 0.956, see ../rd33-ab1), other models.
Promotion is blocked on the correctness/quality gate, not on performance.

## Quality check (added): perplexity at batch 1 (the only path rd33 touches)
`llama-perplexity -sm tensor -c 512 -b 1 -ub 1 --chunks 8` on 8 chunks of repo docs text (same corpus, same builds; logs ppl-*.log).
control PPL 11.6132 +/- 0.717; rd33 PPL 11.5989 +/- 0.716 (-0.12%). rd33 is lower in all 8 running-mean checkpoints.
Reading: the numerical change moves predictions slightly toward better, not worse (consistent with removing activation quantisation).
Scope: one corpus, 4096 tokens, one model. It is evidence of no quality regression, not a formal contract correctness gate (5e-4 logprob).
