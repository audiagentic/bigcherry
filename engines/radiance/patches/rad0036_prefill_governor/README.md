# rad0036_prefill_governor

## Summary

VTP42 puts the scheduler's prefill admission and ordering policy behind one engine option and one header,
`core/sched/prefill_governor.h`, so a policy can be swapped or added without new options or new scheduler code
paths. The spec is `kind=K;key=value;...`:

| key | meaning |
|---|---|
| `kind=arrival` | **default** (empty spec): priority, then arrival; no admission limit |
| `kind=cap` | at most `max_seqs` prefills with more than `long_tokens` of prompt left run at once (`long_tokens=0`: one shared step); shorter prompts hold no slot |
| `kind=budget` | prompt tokens left across running prefills stay within `factor` (default 1.5, at least 1) x the largest request (the context limit), and admitting must fit free KV; a lone prompt always runs; a prompt waiting past `age_ms` reserves: nothing behind it is admitted until it fits |
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
`admit_one` (after the prefix lookup, so a hit can make a prompt short; a refusal is a wait, or under a budget
reservation the end of the step's admissions), and the overdue-recompute guard (no evicting decoders for a recompute
the governor cannot admit). The `radiance:prefilling_sequences` gauge and
`radiance:prefill_admissions_deferred_total` counter carry over from rad0034.

Motivation: VTP39 and VTP41 added the cap, threshold and ordering as separate options; the owner asked for them as
one alternative governor so no legacy mechanism stays in the main code path, with room for a token-budget and later
a backfill policy (plan item VTP42).

## Outcome

Untested. Card-free (`tests/sched_test.cpp`, `tests/config_test.cpp`): rad0034's and rad0035's tests ported to
spec strings; equivalent specs plan the same steps over 200 random schedules; the budget keeps a third 3000-token
prompt waiting while a 900-token one fits, and an aged prompt's reservation stops later prompts passing it; bad
specs are refused at parse; a short prompt beside two long prefills running past the age reaches its first token
in 5 steps (13 if running prefills aged).

Change from rad0035: there, `age_ms` also put running prefills back in arrival order. On the card (VTP41 confirm,
`-long-bf16`, eight 100k prompts plus mid-size waves at 30 s and 150 s) that made the 150 s wave's 4k / 8k / 16k
requests wait 43-69 / 128-151 / 139-160 s against 4-7 / 11-13 / 20-23 s without the age, once the 100k prompts passed
120 s, for no change in the 100k mean. A running prefill always keeps a share of the step, so it cannot starve; only
a prompt not yet started needs the bound. Hardware for this patch: see the VTP42 evidence.

## Detail

- Spec parse and `describe()` live in the header; `config.cpp` validates at parse and `--explain` shows the
  normalised spec; the scheduler logs it at init.
- `prefill_load(with_plan)` counts running prefills and, during admission, this step's prefill entries not yet in
  `running_`; a prompt that finishes in its admission step holds nothing.
- `max_request()` is the context limit (`ctx_limit_`, else `max_ctx`).
