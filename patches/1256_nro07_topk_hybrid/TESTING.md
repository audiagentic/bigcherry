# Testing — NRO07 hybrid TOP_K

Before dispatch wiring, build a CPU/current-backend reference fixture over ncols/k/nrows boundaries, random values, all-negative values, +/-Inf, duplicate/tied values and NaN policy. Record exact index-set and ordering requirements used by MoE/QSA callers.

After kernel port, compare every fixture against stock bitonic, with forced fallback tests. Capture real TOP_K signatures and call counts before performance claims. Non-HIP builds must compile and remain byte/behavior equivalent at the dispatch site.

NRO07-only is the control for NRO08; do not qualify both together first.
