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
