# HI85 heterogeneous-architecture guard

This package contains the optional NCCL/RCCL heterogeneous-architecture
fail-closed guard (HI85). The durable contract is: mixed-architecture
communicator initialization must reject before launch; homogeneous and
non-NCCL paths remain unchanged. Architecture rationale and profiling context
are summarized in `docs/reference/architecture/MULTI_GPU_DISPATCH.md`; patch
identity and lifecycle metadata are authoritative in `patch.toml`.
