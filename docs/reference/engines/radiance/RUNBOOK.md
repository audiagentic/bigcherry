# Radiance build and serving runbook

Radiance production builds use the pinned recipes in `engines/radiance/recipes.toml`, with component binaries installed in `radiance_home`. The engine release is a **Python-free runtime**, separate from the platform tooling.

## Build for development

From a configured checkout:

```sh
PYTHONPATH=tools python3 -m bigcherry build --lane radiance:stock:linux-radiance
```

For a helper-offload build, require `bin/radiance` and `radiance_home/helpers/r3_moe_worker`; `radiance_home/bin/bc-serve` is the native process owner. Lab-side `lab-build.sh` is retained for developer builds; no shell launcher owns production serving.

## Install a verified release

Choose an **exact** `bc-radiance-X.Y.Z` tag and acquire its three assets (`bigcherryrad-PLAN.tar.gz`, `.sha256`, `.receipt.json`) from that **same tagged GitHub release**. Verify the publisher/source of the receipt; `latest` is not an immutable trust root. Use the native manager supplied with the engine:

```sh
radiance_home/bin/bc-release install --tag bc-radiance-X.Y.Z \
  --record /path/to/tag-bound.receipt.json \
  --archive /path/to/bigcherryrad-PLAN.tar.gz \
  --root /path/to/install-root --sdk /path/to/rocm-sdk
radiance_home/bin/bc-release verify --root /path/to/install-root --sdk /path/to/rocm-sdk
radiance_home/bin/bc-release verify --root /path/to/install-root --sdk /path/to/rocm-sdk
```

The native installer verifies tag, manifest, SDK, every runtime file digest and rollback candidates. It does not create source worktrees or use Python. See [RELEASES.md](../../build/RELEASES.md#installing-a-release) for the refusal matrix and rollback.

## Start, inspect and stop

Use `bc-serve` **from the current verified installation prefix**. The binary automatically chooses that prefix's `configs/`, `bin/radiance` and runtime home unless overridden; it refuses a mismatch or unverified/non-current prefix before spawning. Examples (replace inputs with actual installed paths and reviewed site/model settings):

```sh
/path/to/install-root/releases/bc-radiance-X.Y.Z/radiance_home/bin/bc-serve explain flashnext-r9700 --model-root /path/to/models
/path/to/install-root/releases/bc-radiance-X.Y.Z/radiance_home/bin/bc-serve start flashnext-r9700 \
  --site /path/to/site.toml --model-root /path/to/models --work-root /path/to/work-root --run-dir /path/to/run
/path/to/install-root/releases/bc-radiance-X.Y.Z/radiance_home/bin/bc-serve status --run-dir /path/to/run --json
/path/to/install-root/releases/bc-radiance-X.Y.Z/radiance_home/bin/bc-serve stop --run-dir /path/to/run
```

For a foreground process supervised by the caller, use `bc-serve serve CONFIG` with the same config/site/model/run-dir arguments. For a developer build outside an installed release, pass explicit `--configs`, `--engine-binary`, site, model root and run directory; that is not equivalent to a tagged release.

Obtain the exclusive `gpu` lease **before** a hardware start, measure, calibration, or benchmarking operation, and keep it through verified child exit and power restoration. Native start performs helper ordering, measured-plan admission, SDK checks, health/canary, power ownership and stop. Failure is a typed refusal, never a fallback to an old launcher. Full dense/helper hardware parity remains blocked on [S15](RSC02_S15_ACCEPTANCE.md).

## Configs and quality

The packaged serving configs are one native YAML per config (`configs/NAME.yaml`; the roster is the YAML files in `engines/radiance/configs`). Use `bc-serve explain CONFIG --model-root DIR` to inspect effective intent (it reads `NAME.yaml` from the verified bundle, else from `--configs DIR`, or from `--native-config YAML`, and runs the same admission as a start, offline) and `bc-serve status` to inspect the owned run. See [SERVING_MANIFEST.md](SERVING_MANIFEST.md) for manifest semantics, [LIVE_PLACEMENT.md](LIVE_PLACEMENT.md) for heat policy, and [enhancements](ENHANCEMENTS.md) for patch validation state.

For paired hardware experiments prefer `bigcherry ab-benchmark`; preserve the exact build/model/SDK hashes, immutable manifest, observed helper roles, quality digests, temperatures and power receipts. The RDNA2 helper tail has a documented worst-case quality penalty (RAD46); performance alone is not an acceptance result.
