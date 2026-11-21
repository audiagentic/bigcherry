# Radiance bump coupling: live preflight

Radiance's pin is `engines/radiance/recipes.toml` (`pinned`,
`pinned-version`). This is a **preflight checklist**, not an independent
inventory snapshot or a semantic acceptance certificate. Read
[PIN_BUMP.md](../../build/PIN_BUMP.md) and
`.claude/skills/bump-radiance/SKILL.md` for the procedure.

## Resolve the current selection

- `engines/radiance/engine.toml` lists the six owned components:
  `serving`, `moe-remote`, `libr3`, `libr2`, `ngram-helper`,
  `bc-gfx12`. Each `component.toml` declares `built-against`,
  `[coupling]` source paths, header/ABI, flag/env literals, and (where
  relevant) helper wire identities. The parser and gate in
  `tools/bigcherry/core/component_coupling.py` check the *declared*
  surface against both source and consumer.
- The recipe's `[source.*]` selections, patch `requires` graphs and
  `patch.py` targets are authoritative. The patch catalog and the generated
  [footprint](../../../evidence/radiance-patch-footprint/footprint.md)
  describe active attachments; the unchanged retired packages and their
  evidence are indexed in `engines/radiance/patches/archive/ARCHIVE.md`.
  Do not use a copied patch-count or path table to decide applicability.
- The active wire is serving **edition 3** from the one flat
  `engines/radiance/components/serving/contract/serving.toml`.
  `bc_contract_check` must regenerate the v3 schema, typed codec and
  `generated/v3/contract.fingerprint` without a difference. Its schema
  and semantic fingerprints are distinct; check both. Namespace names in
  generated C++ are not a second wire edition.
- The deployment authority is each
  `engines/radiance/configs/NAME.yaml`: upstream native settings plus
  a typed `x-bigcherry` subtree on the same sealed bytes. The engine
  inspector and `bc-serve explain NAME --native-config ... --json` must
  agree. Resolution root, YAML digest, parser/build identity, ordered
  devices and fresh room admission belong to the measured native v3
  owner; neither an old catalogue nor a second native-option registry
  supplies values.

## Run the gates, in this order

1. **Candidate identity:** record the exact upstream commit and
   `RAD_ABI_VERSION`; compare the exported layout fingerprint, CMake
   sources, plugin signatures and registered kernels on the same SDK.
   `pin-triage --json` is advisory. A missing ABI, broken coupling
   declaration or native rewrite is a refusal, not a clean result.
2. **Exact attachment proof:** apply `patch-rebase-check --source NAME`
   separately for each current `[source.*]` (especially the production
   `radiance` composition and diagnostic variants), then
   `patch-rebase-check --all` for any otherwise unselected registered
   package. Resolve `--all` failures by porting a still-needed patch or
   archiving an upstream-superseded package **unchanged** with an
   `ARCHIVE.md` disposition. Re-run both. Component declarations do
   not prove patch semantics or remote wire behavior.
3. **Footprint and native rows:** run `patch-footprint --check` before
   changes, regenerate after composition changes and record the
   before/after number of upstream anchors. Use
   `libr4d/bc_kv_rows.h` as the common declaration/row/chain
   registration site: new geometry needs a matching kernel capability,
   not copied row blocks or arbitrary local-head pins.
4. **Compiled contract and bundle:** run the `pin-bump` coupling drift
   gate, `libr3/r3_rewrite.py --check`, component builds and CTests,
   `component-plugin-abi`, `bc_contract_check`, lane build, CPU
   smoke and exact-source patch tests. All six component builds must
   match the candidate `built-against`, plugin ABI **and layout**.
   Keep the runtime Python-free; build-time generators are allowed.
5. **Model and runtime acceptance:** check every deployed YAML against
   retired weight formats and new quant scale/KV geometry. An IQ8R
   256-column scale group makes odd V-head TP splits invalid even
   though the rotation is FWHT128; the admission rule must use the
   container's declared group. Explain/preflight all YAMLs offline
   before GPU allocation. Then qualify canary, helper and mixed-rank
   start/stop, fresh/named/stored replay, KLD and interleaved paired
   performance on the *installed* exact bundle. The shared model
   identity cache under `${BC_WORK_ROOT}/cache/model-identities/`
   caches full digest identity only, never fresh resource admission.

A **DECLARED** coupling gate means the corresponding source/header/option
or wire literal is checked. It does not establish end-to-end safety,
model format support, replay equivalence or performance. RUM plans own
remaining semantic acceptance; preserve prior-pin evidence as immutable
history and record newly qualified facts under the new build identity.
