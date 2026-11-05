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
[dated Radiance lab record](../LAB_HISTORY.md),
[the RAD23 layout record](../../../planning/active/patching-radiance-rdna3/RAD23.md),
[the RAD42 combine record](../../../planning/active/patching-radiance-rdna3/RAD42.md)
and [the RAD37 current-pin check](../../../planning/completed/patching-radiance-rdna3/RAD37.md).
The [Radiance overview](../../../reference/engines/radiance/README.md) lists the related components.
