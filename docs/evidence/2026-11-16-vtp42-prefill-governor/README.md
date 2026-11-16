# VTP42: prefill governor (rad0036) on the card

Hardware A/B, `qwen27b-tp3-int8-556-long-bf16` (624k-token bf16 cache, 262,144-token context), served argv plus
`--prefill-interleave 2` and the arm's `--prefill-governor`; one fresh engine start per run, two starts an arm.
Workload as VTP41 (`docs/evidence/2026-11-15-vtp41-prefill-order/README.md`): eight distinct ~100k-token prompts sent
together, then waves 30 s and 150 s in of one ~450, ~4k, ~8k and ~16k-token request. Every request returned 200;
no preemptions in any run.

| arm | spec | ~450 | ~4k | ~8k | ~16k | 100k mean | last 100k |
|---|---|---|---|---|---|---|---|
| G | `kind=cap;max_seqs=2;long_tokens=16384;order=remaining;age_ms=120000` (`-long`) | 3-5 s | 5-7 s | 11-15 s | 21-27 s | 299 s | 477 s |
| B | `kind=budget;factor=1.5;order=remaining;age_ms=120000` | 15 / 218 s | 16 / 219 s | 20 / 222 s | 26 / 228 s | 209 s | 367 s |
| N | `kind=budget;factor=1.5;order=remaining` | 15 / 39 s | 16 / 40 s | 19 / 44 s | 26 / 49 s | 215 s | 378 s |
| X | `kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000` | 3-5 s | 5-8 s | 11-16 s | 19-28 s | 228 s | 390 s |
| Y | `kind=budget;factor=1.5;long_tokens=16384;order=remaining` | 2-5 s | 4-8 s | 10-16 s | 14-28 s | 227 s | 387 s |

B and N give two values (30 s wave / 150 s wave); repeat starts agree within 1 s. B and N ran before short prompts were
exempt from the budget (their rows are the case the exemption fixes).

## Reading

- **G** reproduces the VTP41 `-long` options run as separate flags (U: 299 s, 4-28 s) through the one spec, and its
  150 s wave is no longer held back by the age limit (VTP41 V: 43-160 s), since only waiting prompts age now.
- **Budget 1.5 x the largest request** (393k tokens): with `order=remaining` the step goes to the prompt nearest its
  first token, so the 100k prompts finish one after another about 44 s apart (one 100k prompt's solo prefill), which
  is what lowers the mean (as cap 1 did in VTP39, 208 s). The budget admits up to four at once, so later ones are
  already partly computed.
- **B / N:** without an exemption, short and mid-size requests wait for budget to free, up to one 100k prompt's
  prefill (N, 39-49 s); with the 120 s reservation the waiting 100k prompts hold every later admission until the last
  starts (B, ~220 s). Hence `long_tokens` now exempts short prompts in every kind and from any reservation.
- **X / Y:** with mid-size prompts exempt, the budget keeps G's mid-size first-token times and lowers the 100k mean
  299 -> 227 s (-24%) and the last 100k 477 -> 388 s. The reservation (X) costs 0-6 s on the second wave's 16k and is
  the bound that stops a waiting long prompt being passed by later long ones indefinitely.
- Not measured: decode streams beside the mix, fp8 `-long`, a steady arrival stream.

Owner decision (2026-11-16): the `-long` configs (fp8 and bf16) use X,
`kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000`.
