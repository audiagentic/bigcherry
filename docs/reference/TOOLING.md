# BigCherry tooling

This guide is the normative map for maintained tooling. Before creating a tool, search the existing command/API and the owning domain. If classification is unclear, start in `tools/lab/<plan-topic>/`.

## Core BigCherry Product Workflows

### 1. Pin and release

Use `bigcherry repin`, `pin-status`, `pull`, `audit`, and release validation/records. `pin-status` is non-mutating; `repin` owns pin transition state.

### 2. Source and build

Source audit and identity live in the current `bigcherry` modules. Use `generate` and `build` for content-addressed source/build artifacts. Real compilation is explicit and is not part of `check`.

### 3. Patches

Use `patches`, `patch-status`, `patch-explain`, `patch-lint`, `patch-validate`, and evidence verification. Flat patches remain supported. Packaged patches use `patch.toml` as metadata authority and `validation.toml` for validation configuration.

### 4. Campaigns

Use the canonical campaign planner/lane/build path. Campaign runs own source, build, smoke, comparison, and benchmark orchestration. Hardware campaigns are explicit.

### 5. Tuning and replay

Use the catalog, journal, promotion, correctness, ranking, and replay tooling. Never treat a measured result as evidence without its provenance and identity bindings.

### 6. Experiment Contracts

Experiment Contracts and bundles are scientific-authority features. Keep contract identity, inputs, outputs, and state transitions explicit.

### 7. Diagnostics and check

`bigcherry check` and `doctor` are deterministic inspection workflows. Check is non-mutating, hardware-free, and does not launch ROCm builds, models, or campaigns.

## Maintained analysis

Reusable offline reports belong under the future `bigcherry.analysis` domain and must document inputs, outputs, and mutation behavior. Product workflows must not depend on analysis by default.

## Lab and environment

Temporary investigations belong in `tools/lab/<plan-topic>/`; environment setup belongs in `tools/env/`. Lab is intentionally not a Python package and is never evidence authority. See the corresponding READMEs.

## Tests and safety

Permanent tests remain under `tools/tests` until the test-domain migration phase. Hardware tests are explicit. Do not report hardware or benchmark results that were not actually observed.

## Migration note

The canonical destination package directories are being established incrementally. This guide distinguishes intended destinations from current supported import paths; no implementation has moved as part of TR01.
