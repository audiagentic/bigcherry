# rad0036_prefill_governor

## Summary

VTP42 puts the scheduler's prefill admission and ordering policy behind one engine option and one header,
`core/sched/prefill_governor.h`, so a policy can be swapped or added without new options or new scheduler code
paths. The spec is `kind=K;key=value;...`:

| key | meaning |
|---|---|
| `kind=arrival` | **default** (empty spec): priority, then arrival; no admission limit |
| `kind=cap` | at most `max_seqs` long prefills run at once |
| `kind=budget` | prompt tokens left across running prefills stay within `factor` (default 1.5, at least 1) x the largest request (the context limit), and admitting must fit free KV; a lone prompt always runs; a long prompt waiting past `age_ms` reserves: no other long prompt is admitted until it fits |
| `long_tokens=N` | a prompt is long while more than N of it is left (**0**: one shared step); a short prompt is never held back by any kind or reservation and holds no cap slot |
| `order=arrival\|remaining` | after priority, walk and admit by least prompt left; prefills behind the first keep one share between them |
| `age_ms=T` | a prompt still waiting to start after T goes back to arrival order (and, under `budget`, reserves); a running prefill never ages |

Serving v2 field: `prefill_governor` (string). It replaces rad0034's `max_prefill_seqs` and rad0035's
`long_prefill_tokens`, `prefill_order` and `prefill_age_ms`:

| before | spec |
|---|---|
| `max_prefill_seqs = 2` | `kind=cap;max_seqs=2` |
| + `long_prefill_tokens = 16384` | `;long_tokens=16384` |
| + `prefill_order = "remaining"` | `;order=remaining` |
| + `prefill_age_ms = 120000` | `;age_ms=120000` |

The scheduler keeps four hooks: the queue order (`ahead_of`), the step walk's prefill shares, an admission gate in
`admit_one` (after the prefix lookup, so a hit can make a prompt short; a refusal is a wait, and a budget
reservation also holds every later long prompt that step), and the overdue-recompute guard (no evicting decoders for a recompute
the governor cannot admit). The `radiance:prefilling_sequences` gauge and
`radiance:prefill_admissions_deferred_total` counter carry over from rad0034.

Motivation: VTP39 and VTP41 added the cap, threshold and ordering as separate options; the owner asked for them as
one alternative governor so no legacy mechanism stays in the main code path, with room for a token-budget and later
a backfill policy (plan item VTP42).

## Outcome

Untested. Card-free (`tests/sched_test.cpp`, `tests/config_test.cpp`): rad0034's and rad0035's tests ported to
spec strings; equivalent specs plan the same steps over 200 random schedules; the budget keeps a third 3000-token
prompt waiting while a 900-token one fits, and an aged prompt's reservation stops later long prompts passing it while short ones still pass; bad
specs are refused at parse; a short prompt beside two long prefills running past the age reaches its first token
in 5 steps (13 if running prefills aged).

Change from rad0035: there, `age_ms` also put running prefills back in arrival order. On the card (VTP41 confirm,
`-long-bf16`, eight 100k prompts plus mid-size waves at 30 s and 150 s) that made the 150 s wave's 4k / 8k / 16k
requests wait 43-69 / 128-151 / 139-160 s against 4-7 / 11-13 / 20-23 s without the age, once the 100k prompts passed
120 s, for no change in the 100k mean. A running prefill always keeps a share of the step, so it cannot starve; only
a prompt not yet started needs the bound.

Hardware (`docs/evidence/2026-11-16-vtp42-prefill-governor/README.md`, `-long-bf16`, eight 100k prompts plus
mid-size waves): the `-long` cap spec reproduces the VTP41 options (100k mean 299 s, 4k-16k 3-27 s);
`kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000` keeps those mid-size times and lowers the
100k mean to 228 s (-24%) and the last 100k prompt 477 -> 390 s; no preemptions.

## Detail

- Spec parse and `describe()` live in the header; `config.cpp` validates at parse and `--explain` shows the
  normalised spec; the scheduler logs it at init.
- `prefill_load(with_plan)` counts running prefills and, during admission, this step's prefill entries not yet in
  `running_`; a prompt that finishes in its admission step holds nothing.
- `max_request()` is the context limit (`ctx_limit_`, else `max_ctx`).

### Promotion record

Model: Qwen3.8-27B w8a8 int8 (.rad), qwen27b-tp3-int8 (TP3 over two RX 7900 XTX and an R9700)
Mechanics: tools/tests/patch/test_vtp37_int8_pipelined_prefill.py passes and patch-lint is clean on the promotion PR; the lane builds carrying it build and serve.
Activation: `--pipelined-prefill on` (the layout's setting) against `off` selects the pipelined step; VTP37.md shows the serial note gone and the pipelined path taken in the on arms.
Identity: not bit-identical to the serial step at every prompt length, as on every pipelined step (rad0020 at MXFP4 too): greedy text identical at 300, 508, 4016, 6027 and 6090 tokens and different at 1092, 2023 and 2955 near ties; first-token log-probabilities bit-identical under 1024 tokens and within 0.001 on the top token above (VTP37.md). Owner reordering quality gate (VTP37.md, 2026-11-13) instead of bit identity: KLD against the same VTP36 BF16 reference, 42,761 positions, build main 4942dad1, layout 6,6,4: serial 0.00315 mean / 0.01057 p99 / 0.0893 p99.9 / 96.86% top-1 / PPL +0.040% (repeat identical to every digit), pipelined 0.00337 / 0.01019 / 0.0838 / 96.84% / +0.014%: mean +7.1%, p99.9 -6.2%, top-1 -0.02 points, PPL -0.026% (relative gate of +10% / -0.2 points / +0.1% passes; prq-rad0030, 2026-11-16). Limit: serial at 1024-token chunks scores lower still (0.00244 mean, 0.0715 p99.9), so reordering noise is the size of the difference; VTP37 recorded the serial-reference noise-floor check as passed (0.00125 / 0.021 / 99.31% against floor 0.00197 / 0.059 / 97.81%). The original gate's absolute BF16 values (VTP14 N6 reference) could not be rerun, the reference no longer being on disk.
A/B: 2x ABBA against the build without it: one-prompt time to first token -17% to -21%, four 4096-token prompts -16%, 4/6/8 streams +7.4% / +7.8% / +8.4% (VTP37.md).

Model: Qwen3.8-27B int8 (.rad), qwen27b-tp3-int8-556-long-bf16 (TP3 over two RX 7900 XTX and an R9700, bf16 KV cache, 262,144-token context)
Mechanics: tools/tests/patch/test_vtp42_prefill_governor.py passes and patch-lint is clean on the promotion PR; card-free sched_test shows each rad0034/rad0035 setting planned step for step by its spec equivalent (README, VTP42.md).
Activation: the -long configs serve with `--prefill-governor kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000` (owner decision 2026-11-16); the arms of docs/evidence/2026-11-16-vtp42-prefill-governor/README.md differ only in that spec and show different admission behaviour (15/218 s waits for B, 3-5 s for X).
Identity: one-stream greedy text sha identical under the default budget spec (2 starts) and kind=arrival at 302, 1085 and 4019 prompt tokens, 128 new tokens (prq-rad0036/ident, 2026-11-16); the governor only orders and admits prefill work, so a lone stream is unchanged by construction (this is weak evidence under contention).
A/B: hardware, eight ~100k prompts plus waves of ~450/4k/8k/16k: 100k mean first token 299 -> 227 s (-24%), last 100k 477 -> 388 s, mid-size first-token times held at 3-28 s, every request 200, no preemptions (VTP42 evidence README, two starts an arm); 4 decode streams beside the mix: 16.6 chunks/s (~4.15/stream) against 10.35 (~2.59/stream) under the cap spec, +60%, gap p50 ~44 ms both, 100k mean 279 vs 330 s (prq-rad0036/mix, two starts an arm, 0 preemptions). Not measured: fp8 -long, a steady arrival stream.
