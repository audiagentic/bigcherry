# Testing — NRO02 fused AllReduce residual

The current draft is kernel-only. Static tests must prove NRO01 dependency and that both residual kernels are present but no graph skip is reachable.

Before wiring Meta fusion, add graph fixtures for: direct ADD; reshape->ADD; ADD operand order; extra consumer; non-reshape intermediate; mismatched type/shape; non-mirrored residual/output; missing provider entry point; provider returns false; null residual. Only the two positive patterns may skip an ADD.

GPU comparison uses identical wire mode in control/subject. Exact mode should characterize bitwise operation order; Q8/BF16 inherit the representation's accepted tolerance. Activation evidence must prove the standalone ADD launch disappears and no other node is skipped.
