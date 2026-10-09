# Testing — NRO08 wave32 TOP_K

Require NRO07 first. Device fixture must compare wave32 reduction with CPU/current TOP-1 over random values, ties, negative values, invalid/sentinel indices and block sizes 32/64/128/256.

Before adding radix-half scan, test rank thresholds around 31/32/33 and 63/64/65. Before items/thread tuning, test ncols at block*items +/-1 boundaries.

Performance control is NRO07-only. Report LDS traffic/barriers/pass count where profiler data is available. Exact routing correctness remains mandatory.
