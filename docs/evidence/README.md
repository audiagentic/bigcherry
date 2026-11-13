# Tracked evidence

This directory contains compact, immutable evidence bundles needed by tests or for reproducible review from a fresh checkout. Large traces, databases, and machine-local run products belong under the ignored `artifacts/` tree instead.

Each bundle must include provenance, source/tooling identity, and checksums for the files it contains. Do not use this directory for plan status or patch metadata; those remain under `docs/planning/` and `patches/<patch-id>/`.

## Non-patch real-hardware verification

Feature/toolchain/hardware checks that are not patch-acceptance campaigns use this same store. Do not create a second evidence engine and do not force these records through patch-specific validation code.

Use one directory per immutable run, normally `docs/evidence/<YYYY-MM-DD>-<plan-id>-<slug>/`. Keep large/raw outputs under `artifacts/<run-id>/`; commit only the compact proof needed to review the result.

A non-patch bundle should contain:

- `README.md`: what was tested, exact command/procedure, result, and limits of the claim.
- `manifest.json`: structured identity. Include `schema_version`, `kind = "non-patch-hardware-verification"`, `plan_item_ids`, observation time, host role, hardware identity, result/verdict, and the canonical provenance-v2 document shape from `tools/bigcherry/core/provenance.py` where the producing workflow can provide it. Do not invent missing provenance for historical runs.
- `SHA256SUMS`: checksums for every committed evidence payload other than `SHA256SUMS` itself. If a large raw artifact remains under ignored `artifacts/`, record its relative/host path and checksum in `manifest.json` or the README so the compact bundle cannot be mistaken for the raw data.

Minimum capture procedure for a new ad-hoc hardware check:

```bash
RUN_ID=<YYYY-MM-DD>-<plan-id>-<slug>
mkdir -p "docs/evidence/$RUN_ID" "artifacts/$RUN_ID"
git rev-parse HEAD
# Run the documented check, writing raw output under artifacts/$RUN_ID/.
# Copy only compact review evidence into docs/evidence/$RUN_ID/.
sha256sum docs/evidence/$RUN_ID/<payload> > "docs/evidence/$RUN_ID/SHA256SUMS"
```

Before committing, verify that the README/manifest name the exact BigCherry revision, source revision/build identity when applicable, command, host/device identity, toolchain identity, result, and limitations. A prose-only historical note without recoverable source/toolchain identity or checksummed output is context, not a reconstructed evidence bundle; recover the original artifacts or rerun the check instead of manufacturing missing fields.

Patch validation remains owned by `tools/bigcherry/patch/evidence.py` and the patch-validation workflow. Campaigns should keep using their existing artifact/provenance writers. This convention is only the durable tracked envelope for otherwise ad-hoc non-patch verification.

## TR00 tooling rationalisation

`tooling-rationalisation/TR00/` contains the immutable 383-row implementation-start disposition evidence. The maintained 385-row registry is [`../reference/tooling/TOOL_DISPOSITION.md`](../reference/tooling/TOOL_DISPOSITION.md); the two RA39 lab rows are current-only and are intentionally absent from the historical bundle.

## Recovered local-only investigation state

- [5 October 2026 QFP22 b11402 bisect](2026-10-05-qfp22-b11402-local-bisect/README.md): the exact eight unpublished, temporary pin-selection and release-record commits, captured as a patch series and provenance ledger. The authoritative measured findings remain in QFP22/BCOP27; the old commits must not be replayed onto the active pin.
- [13 November 2026 RAD01 Radiance v1.3.0 prototype capture](2026-11-13-rad01-v13-remote-prototype/README.md): the previously uncommitted tracked diff, two untracked source files and historical build output, with links to the current managed successors. This is source-history preservation, not a production engine patch or new hardware qualification.

These bundles close the local-only **source and Git-history** gaps identified in those two checkouts. They do not claim that all GPU run artifacts in the work root have been archived; the evidence rules above continue to govern other experiments.
