"""Upper-level dense full-attention KV arithmetic (card-free, not an allocator).

Only classic K/V tensors are counted. Recurrent/GDN state, sparse-attention
indexers, graph arenas, MTP/session copies and allocator padding are separate.
"""
from __future__ import annotations


def full_attention_kv_bytes(*, context_tokens: int, attention_layers: int,
                            kv_heads: int, head_dim: int, dtype_bytes: int,
                            tp_ranks: int = 1) -> dict:
    values = (context_tokens, attention_layers, kv_heads, head_dim, dtype_bytes, tp_ranks)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("all dimensions, byte widths and ranks must be positive integers")
    if kv_heads % tp_ranks:
        raise ValueError(
            f"{kv_heads} KV heads cannot split evenly over {tp_ranks} ranks: "
            "the actual engine must prove uneven splits or KV replication"
        )
    logical = 2 * context_tokens * attention_layers * kv_heads * head_dim * dtype_bytes
    return {
        "logical_total_bytes": logical,
        "per_rank_bytes_if_head_sharded": logical // tp_ranks,
        "per_rank_kv_heads": kv_heads // tp_ranks,
        "excludes": ["GDN/recurrent state", "sparse indexer/QSA", "MTP", "allocators",
                     "graph arenas", "sessions and padding"],
    }
