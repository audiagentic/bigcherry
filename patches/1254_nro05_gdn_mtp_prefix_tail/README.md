# 1254_nro05_gdn_mtp_prefix_tail

Plan `NRO05`; state `untested`; requires NRO04.

The source optimization accelerates `K>1` GDN prefill by running a chunked prefix of `n_tokens-K`, then retaining the existing sequential kernel for the last K tokens so snapshot slots remain defined by the stock recurrence. This first draft only materializes the fail-closed eligibility predicate and prefix calculation; runtime routing remains unchanged.

Initial predicate: non-KDA, K>1, single sequence, `S_v` supported by the parent chunked implementation, and `n_tokens > K + 64`. Multi-sequence and short inputs remain sequential.

Before wiring, tests must compare every snapshot slot and final state, not merely logits or absence of crashes.
