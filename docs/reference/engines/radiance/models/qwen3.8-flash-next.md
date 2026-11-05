# Qwen3.8-Flash-Next

The main heterogeneous Radiance workload. Results below use the pinned
**Radiance v1.14.0** and the 113.6 GiB
`qwen3.8-next-flash-fp8-iq4r-moe.rad` container: 512 routed experts, top 10,
48 main layers plus an MTP layer. Expert weights use `w4nl64a8h`; the
47.7 GiB n-gram table resides in host RAM for these runs.

## Best recorded throughput

The four-card layout has an R9700 engine, two RX 7900 XTX expert helpers and
an RX 6900 XT holding a 14 GiB tail of colder experts. Rates are
**tokens/s**, not latency or estimated kernel throughput.

| Prefill workload | Best recorded |
| --- | ---: |
| Fresh ~44K-token prompt | **3,837** |
| Eight sessions, 1.04M prompt tokens in total | **2,513** |
| New 50K-token prompt after ~1M retained session tokens | **3,165** |
| ~340K-token extended-context prompt | **3,397** |
| ~650–690K-token extended-context prompt | **2,796** |

The eight-session figure uses chunked helper-row combine. The retained-session
result uses a 9 GiB host expert pool and a 12 GiB host KV tier. Extended-context
runs use the 1M-token setting with YaRN scaling; their prompt lengths and
cache state differ from the fresh 44K run.

| Concurrent streams | Aggregate engine decode, tokens/s |
| ---: | ---: |
| 1 | **144–146** |
| 2 | **176–201** |
| 4 | **249–255** |
| 6 | **330–342** |

Decode figures use the four-card layout, 512-token prompts, 512 generated
tokens, three speculative drafts and the W4H2 helper GEMV. These are
*engine decode* figures; actual client output varies with draft acceptance,
text and concurrent prefill.

**These decode figures are from an unpinned client.** The benchmark client that
produced them sent no temperature, so the text was sampled and the rate follows
the draft acceptance of whatever was sampled (the RAD23 record found 93% against 43% on
two builds with identical draft counts and no kernel change). With `temperature: 0`
on pinned prompts, launcher decode is 100 to 140 tokens/s at one stream
([RAD23](../../../../planning/active/patching-radiance-rdna3/RAD23.md), decode paragraph),
and temperature-0 runs on 2026-11-04 measured 91 to 110 at one stream and 211 to 249 at
six streams on three builds of the four-card layout (with the live placement policy on). Compare builds only with temperature 0
(or a seed) pinned.

Rerun of the launcher defaults on 2026-11-04 (`flashnext-4gpu`: 14 GiB tail, default heat file,
combine on, 12 GiB host KV tier, no policy), `temperature: 0`, seed 7, 512-token prompts, 512
generated tokens, three repetitions; aggregate engine decode and cold prefill:

| Streams | Decode, tokens/s | Step | Draft tokens a step |
|---:|---:|---:|---:|
| 1 | 121.4 | 24.9 ms | 3.38 |
| 2 | 154.8 | 29.3 ms | 5.38 |
| 4 | 224.0 | 39.6 ms | 10.58 |
| 6 | 269.8 | 47.6 ms | 16.03 |

Cold prefill of fresh 48K and 50K-token documentation text: 3,624 and 3,628 tokens/s (the best
recorded fresh ~44K figure above is 3,837). One run of each.

## What changed the result

| Same-model comparison | Baseline | Updated configuration |
| --- | ---: | ---: |
| Eight-session prefill, two XTX helpers versus four-card layout with chunked combine | 1,353 | 2,513 tokens/s |
| Eight-session prefill, four cards, unchunked versus chunked combine | 2,369 | 2,513 tokens/s |
| New 50K prefill after retained-session load, without versus with host KV tier | 1,901 | 3,165 tokens/s |

The first comparison spans several changes and is **not** a controlled
single-variable A/B. The other comparisons isolate the combine and memory
policy more closely. None is a stock Radiance comparison.

The helper workers compute selected experts on the two RDNA3 cards and the
RDNA2 tail. The engine sends activations and collects partial outputs through
a pinned-host mailbox. Heat-ranked placement keeps frequently used experts
on the faster cards. Chunked combine overlaps the reduction of helper
outputs with their arrival instead of serialising the whole exchange.

## Same layout, before and after the policy and kernel changes, measured 2026-11-05

Launcher defaults (`flashnext-4gpu`: 14 GiB tail, combine on, live policy off), `temperature: 0`, seed 7, 512-token
prompts, 512 generated tokens, three repetitions, two passes of each build alternating A, B, A, B. A is the main build
before the live placement policy and the RAD47/RAD48 kernel changes; B is the build with them. Cold prefill is a fresh
~48K-token documentation prompt, before and after the decode runs.

| | A, before | B, after |
|---|---:|---:|
| Cold prefill, tokens/s (four runs) | 3,650 / 3,697 / 3,721 / 3,745 | 3,676 / 3,741 / 3,742 / 3,753 |
| Step, 1 stream | 25.1 ms | 24.9-25.0 ms |
| Step, 2 streams | 29.7-29.9 ms | 29.2-29.7 ms |
| Step, 4 streams | 40.1 ms | 39.2-39.3 ms |
| Step, 6 streams | 45.1-45.3 ms | 45.5-46.8 ms |
| Draft tokens a step, 1 / 2 streams | 2.69 / 6.33 | 3.38 / 5.38-5.45 |

Prefill and step time agree to within about 3%, and a repeat of each build reproduces its first pass, so the changes
cost no speed. Aggregate tokens/s differs between the builds only through draft tokens accepted a step, which is
stable for a build (identical on repeat) but differs between builds even with the text pinned, because a different
placement changes the greedy text early (RHP04). Compare builds by step time, not by tokens/s. Fresh 44K prefill in
this layout is 3,650-3,753 tokens/s; the best recorded figure above is 3,837.

## Long context by tail size, measured 2026-11-04

The 1M-token setting (YaRN, two sequences), two XTX helpers with and without the RX 6900 XT tail, default mixed heat
file, live policy off, C++/HIP source text, a fresh server per arm, temperature 0. Fills of 362K and 710K real tokens.

| Tail | Prefill 362K / 710K, tokens/s | Decode 362K / 710K, tokens/s | Host-to-GPU expert traffic |
|---|---:|---:|---:|
| none (12,288 MiB host pool) | 2,375 / 1,244 | 111-131 / 70 | 1,392 GiB |
| 3 GiB | 2,515 / 2,143 | 112 / 58 | 1,199 GiB |
| 6 GiB | 2,572 / 2,169 | 111 / 71 | 1,056 GiB |
| 14 GiB | 2,634 / 2,307 | 115 / 80 | 613 GiB |

A larger tail halves the expert traffic and lifts the 710K prefill by 85% (1,244 to 2,307); it does not change the
share of promotions that re-admit a just-demoted expert (61-66% in every arm), and long-context decode stays inside
the run-to-run spread (58 to 80 at 710K), so that slowdown is mostly attention over the cache. Without a tail the
default 9,216 MiB host pool cannot serve 1M: one sequence's cache (14.24 GiB) leaves the experts 8.63 GiB, short of
the 9.41 GiB the pool cannot hold, and startup stops out of memory; the no-tail row used 12,288 MiB. One run per arm;
details and caveats in [RHP10](../../../../planning/active/patching-radiance-heat-placement/RHP10.md).

## RX 6900 XT tail size, measured 2026-11-04

Two RX 7900 XTX helpers behind the R9700 engine, with and without the RX 6900 XT as a
tail helper (`flashnext-2xtx` and `flashnext-2xtx-rdna2-tail --budget gfx1030=N`), the
default mixed heat file, no live policy, temperature 0. Three cold ~50K-token prompts, and
decode with 512-token prompts and 256 generated tokens; one run of each. KLD and top-1
against BF16 over 8,240 positions, in two contrasts against the same engine-only control
(the campaign tool takes at most three arms).

| Tail | Prefill, tokens/s | Decode, 4 streams | Decode, 8 streams | KLD mean | p99.9 | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| none (control row: KLD of engine-only) | 3,158-3,216 | 213 | 291 | 0.0678 | 2.376 | 88.57% |
| 2 GiB | 3,128-3,180 | 211 | 289 | 0.0631 | 2.348 | 88.34% |
| 4 GiB | 3,085-3,172 | 208 | 290 | 0.0733 | 3.320 | 87.99% |
| 8 GiB | 3,087-3,195 | 196 | 266 | 0.0605 | 1.761 | 88.83% |

On this heat file a tail of 2 to 8 GiB adds no prefill or decode speed, and 8 GiB costs about
8% of multi-stream decode: the slower card takes experts that carry traffic. The 4 GiB row
breaches the p99.9 gate (+0.94 against +0.5) while 2 and 8 GiB pass; p99.9 over 8,240
positions rests on about eight positions and placement alone has moved it by around one
between runs, so a repeat is needed before reading it as a trend. The recorded four-card
result above uses a 14 GiB tail, where the placement fits the workload.

## Limits and quality

- Current results were recorded on **31 October–2 November 2026**. No
  matched unmodified Radiance v1.14.0 benchmark exists for this four-card
  configuration.
- Throughput at long context depends on KV residency. Without the host KV
  tier, a new prefill after retained sessions fell to 1,901 tokens/s.
- Four-card W4H2 passed the recorded KLD gate against the default-kernel
  control. Draft acceptance and client output rates can still differ
  between prompts. The container-to-full-model reference check recorded
  92.85% top-1 agreement over all tokens and 94.57% over text and assistant
  turns; that is not the helper-kernel A/B result.
- A separate two-XTX helper sweep reached the 1M-token setting at around
  708 prefill tokens/s and 50.2 decode tokens/s at full used context.
  It is **not** the four-card throughput above.

## Evidence

The retained results and comparison conditions are in the
[dated Radiance lab record](../../../../../docs/archive/radiance/LAB_HISTORY.md),
[the RAD23 layout record](../../../../planning/active/patching-radiance-rdna3/RAD23.md),
[the RAD42 combine record](../../../../planning/active/patching-radiance-rdna3/RAD42.md)
and [the RAD37 current-pin check](../../../../planning/completed/patching-radiance-rdna3/RAD37.md).
The [Radiance overview](../README.md) lists the related components.
