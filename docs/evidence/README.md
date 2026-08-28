# Tracked evidence

This directory contains compact, immutable evidence bundles needed by tests or
for reproducible review from a fresh checkout. Large traces, databases, and
machine-local run products belong under the ignored `artifacts/` tree instead.

Each bundle must include provenance, source/tooling identity, and checksums for
the files it contains. Do not use this directory for plan status or patch
metadata; those remain under `docs/planning/` and `patches/<patch-id>/`.
