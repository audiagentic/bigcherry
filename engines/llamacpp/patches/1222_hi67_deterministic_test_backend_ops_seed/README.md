# 1222_hi67_deterministic_test_backend_ops_seed

## Summary

Adds an opt-in deterministic mode (BIGCHERRY_TEST_DETERMINISTIC_SEED) to test-backend-ops' init_tensor_uniform(): single-threaded, seeded generation with a monotonic per-call counter folded into the seed, plus a machine-parseable FNV-1a digest printed on stderr; unset, every existing caller is byte-for-byte unchanged.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
