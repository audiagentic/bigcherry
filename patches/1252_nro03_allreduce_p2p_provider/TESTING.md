# Testing — NRO03 P2P AllReduce provider

Static gates must assert: default-off flag, helper calls `ggml_cuda_set_device(src_device)` immediately before `cudaMemcpyPeerAsync`, no fixed `p2p_issuer`, and required dependency on `1001`.

Hardware gate before live dispatch: both directions copy asymmetric nonzero patterns for repeated sizes and compare every byte/element. A single mismatch disables P2P and records fallback. Test peer-access-already-enabled and unavailable cases.

Only after that probe is green should the AllReduce path compare corrected push-P2P versus host staging. Include actual collective sizes, not only large memcpy bandwidth. Retain failed/negative evidence if host staging wins.
