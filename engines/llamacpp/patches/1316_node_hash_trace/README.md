# 1316_node_hash_trace

## Usage

Set `BIGCHERRY_NODE_HASH=from:count`, where both values are decimal integers, `from >= 0` and `count > 0`.
Example: `BIGCHERRY_NODE_HASH=0:4` observes graph_compute calls 0 through 3 for each llama context.

For simple backend buffers, each observed node is copied to host and logged with an FNV-1a hash. For Meta-owned
logical tensors, the diagnostic logs `skip=meta` and does not call the generic tensor getter. This avoids the
tensor-split/partial Meta getter abort while preserving hashes for nodes whose storage is directly readable.

The callback is diagnostic and intrusive: per-node observation synchronizes work and disables graph replay. Use a
small `from:count` window selected from the unperturbed 1315 trace.

## Fix record

On the production tensor-split layout, `BIGCHERRY_NODE_HASH=0:4` previously aborted in
`ggml_backend_meta_buffer_get_tensor` from `bc_node_hash_cb`. The fix classifies buffer ownership through the public
buffer-type device API and skips `GGML_BACKEND_DEVICE_TYPE_META` before `ggml_backend_tensor_get()`.

## Promotion record

Promotion record (BPB01 lightweight promotion tier, pin b11474, Brutus 2026-10-08).

On the production tensor-split layout with `BIGCHERRY_NODE_HASH=0:4`, the fixed patch emitted 29367 `NODE_HASH` lines, produced 0 error lines, and did not abort. Greedy output md5 was `2571b60b1505`, identical to the arm without 1316 enabled.

The diagnostic remains inert unless `BIGCHERRY_NODE_HASH` is set. The callback is intentionally intrusive when enabled and makes no performance claim.
