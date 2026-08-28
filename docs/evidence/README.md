# Tracked evidence

This directory contains compact, immutable evidence bundles needed by tests or
for reproducible review from a fresh checkout. Large traces, databases, and
machine-local run products belong under the ignored `artifacts/` tree instead.

Each bundle must include provenance, source/tooling identity, and checksums for
the files it contains. Do not use this directory for plan status or patch
metadata; those remain under `docs/planning/` and `patches/<patch-id>/`.

## TR00 tooling rationalisation

`tooling-rationalisation/TR00/` contains the immutable 383-row
implementation-start disposition evidence. The maintained 385-row registry
is [`../reference/tooling/TOOL_DISPOSITION.md`](../reference/tooling/TOOL_DISPOSITION.md);
the two RA39 lab rows are current-only and are intentionally absent from the
historical bundle.
