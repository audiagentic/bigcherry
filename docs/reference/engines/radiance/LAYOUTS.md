# Radiance GPU layouts

BigCherry can split one Radiance engine and its routed MoE experts across
different AMD GPU generations. The current four-card Flash-Next layout
uses the RDNA4 R9700 for the engine, two RDNA3 RX 7900 XTX cards for
frequently used experts and the RDNA2 RX 6900 XT for colder experts.

## Configuration sources

| File | Defines |
| --- | --- |
| `engines/radiance/archs.toml` | Architecture roles, kernel library, supported weight formats and VRAM reserve |
| `engines/radiance/layouts.toml` | Named card groups, expert budgets, ownership order and head/tail placement |
| `engines/radiance/recipes.toml` | Pinned engine, patch selection and build/experiment source |

The host inventory supplies the available GPU identities. Layouts do not
assume a fixed device index. The resolver rejects missing cards,
unsupported roles and budgets exceeding the available memory.

## Measured Flash-Next layout

| Card | Role | Expert budget |
| --- | --- | ---: |
| Radeon PRO R9700 (gfx1201) | Engine | Engine-local experts and KV |
| 2 × RX 7900 XTX (gfx1100) | Heat-ranked helper group | 22 GiB each |
| RX 6900 XT (gfx1030) | Cold-expert tail helper | 14 GiB |

The `flashnext-4gpu` layout uses helper-row combine. Chunked combine
overlaps partial-result reductions with the helpers' output. Its measured
eight-session prefill is 2,513 tokens/s for 1.04M prompt tokens;
see the [Flash-Next result](models/qwen3.8-flash-next.md).

The shared-memory protocol supports up to **four helper processes**;
combine accepts two to four. The current engine path uses one engine
rank. Not every card can perform every role or weight format; check
`archs.toml` and the component manifests.

## Manifest transition

The named TOML layouts describe the existing launcher and benchmark
configuration, not the final manifest-facing serving interface.
The [v1 manifest contract](../../../design/serving-manifest-v1/spec.md)
and its C++ reader are in place; process-role configuration and resource
planning are being integrated. Follow the schema and implementation,
rather than copying launcher environment variables into new interfaces.
See the [serving manifest guide](SERVING_MANIFEST.md) for usage.
