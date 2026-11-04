# Radiance performance limits

BigCherry currently targets **Radiance v1.16.0**. The results below were
recorded on **v1.14.0** and have not been reverified on the current pin.
Different models, topologies and prompts are not directly comparable.

## Last recorded workloads (v1.14.0)

| Model | Best recorded result | Main limitation |
| --- | --- | --- |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md) | Four-card prefill **3,837 tokens/s** (~44K fresh prompt); six-stream engine decode **330–342 tokens/s** | Host-memory traffic, helper balance and retained KV |
| [Qwen3.8-27B](models/qwen3.8-27b.md) | R9700 six-stream client output **245–248 tokens/s** | Prefill and two-XTX drafter throughput not recorded on v1.14.0 |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | R9700 single-stream output **171–219 tokens/s** across prompts | Multi-stream prefix-cache behaviour not qualified on v1.14.0 |

For Flash-Next, the R9700 engine and RDNA3/RDNA2 helper cards exchange
routed activations through pinned host memory. Its four-card 8-session
prefill increased from 2,369 to 2,513 tokens/s when chunked combine
removed serial reduction on large prefill requests. A 12 GiB host KV tier
restored a new 50K-token prefill after retained sessions from 1,901 to
3,165 tokens/s. These are changes within the BigCherry configuration,
**not** comparisons against unmodified Radiance.

## Stock comparison

There is no verified, workload-matched **stock Radiance v1.14.0** result
for these current model/layout combinations. No stock speedup is claimed.
The model pages and [dated lab record](../../../../docs/archive/radiance/LAB_HISTORY.md)
contain the measurements and quality limits behind the figures above.
