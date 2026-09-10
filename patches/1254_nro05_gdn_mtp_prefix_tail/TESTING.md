# Testing — NRO05 GDN MTP prefix/tail

Static: ensure NRO04 dependency, helper returns false for K<=1, n_seqs!=1, kda, and short sequences; returned prefix must be exactly `n_tokens-K`.

Before live routing, build state fixtures for K=2/3/5/8 where valid and prefix lengths around threshold. Compare all K snapshot slots, final state and continuation against fully sequential GDN. Inject chunked-launch rejection and prove full sequential fallback.

Model performance must record MTP acceptance statistics as well as latency/TPS so changed speculative work cannot masquerade as execution speedup.
