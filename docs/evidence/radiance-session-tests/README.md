# RUM04 step (iii)(b): session tests, coverage map and contract identity test

Date: 2026-11-22. Branch `node/rum04-session-tests`, base `origin/main` after the Radiance 2.2 merge (#1167).

## Result

1. **No session test file is deleted.** Each of `session_test.cpp`, `session_guards_test.cpp` and `session_stored_test.cpp`
   reaches production lines that no other test in the serving component reaches (below), so deleting any of them would
   remove coverage rather than a duplicate. `session_stored_test` is also the replay binary `release_install_test` runs
   against an installed prefix (`BC_SESSION_STORED_TEST`), so it cannot be removed without changing that test.
2. **New hard test `bc_contract_identity_test`** (`engines/radiance/components/serving/contract/identity_test.cpp`, a declared
   component test): the registry, the committed fingerprint file, the compiled codec and the generated schema have one identity.

## Method

All 29 card-free ctests of the serving component were built with `--coverage -O0` (one build, `nice -n 19`) and run one at a
time, clearing the `.gcda` files before each. After each test, `gcov --json-format` gave the set of executed lines in
`engines/radiance/components/serving` (the test sources, the test fixture header and the fake engine excluded). For a test T,
"unique lines" are the lines T executes that no other test executes. `bc_session_host_start_test` is `bc_session_test
--host-start-only` and is a subset of it. `bc_release_install_test` runs the stored-replay binary as a subprocess, so for
`session_stored_test` it is excluded from the other tests (otherwise the result is circular).

Limit: `bc_serving_test` aborted under the instrumented build (`stopped while native v3 measured writer was running`, a
timing failure of the slow build, not a logic failure), so its coverage is partial. That can only overstate the unique lines
of the other tests; it cannot hide a duplicate.

## Unique production lines, by function

`session_test` (47 lines, `session.cpp` 46, `callback_boundary.h` 1): `child_environment` 7, `read_helper_readiness` 10,
`run_to_completion` 2, `hooks_for_power_transaction` 5, `group_has_live_member` 2, `stop_child` 3, `read_record` 7,
`start_time` 1, `read_until` 2, `alive_recorded` 1, `record_then_release` 6, `invoke_backend` 1.

`session_stored_test` (139 lines): `session.cpp` 54 (`read_helper_readiness` 28, `heat_launch_nonce` 16, `plan_engine_only` 6,
`wait_measuring_engine` 3, power hooks 1), `v2/s10_resolver.cpp` 36, `v2/s10_inputs.cpp` 24, `s8_replay.cpp` 14 (calibration
comparison), `main.cpp` 5, `manifest_writer.cpp` 5, `serving.cpp` 1.

`session_guards_test` (337 lines): `calibration_lab.cpp` 179 (`read_calibration_drift` 95, prompt reading 24, marker parsing and
completion 30, checked arithmetic 8), `runtime_guards.cpp` 152 (`read_heat_proof` 111, `draft_replayed` 23, `replay_verdict` 6,
`startup` 6, `poll` 3), `v2/manifest.cpp` 5, `runtime_guards.h` 1.

The remaining cases inside each file overlap with other cases of the same file on lines; line coverage cannot show that two
cases assert the same behaviour, so no single case is removed on that evidence.

## Contract identity test

`bc_contract_identity_test` computes every expectation from `contract/serving.toml` in the process (no recorded value):

1. `fingerprint(r)` and `semantic_fingerprint(r)` equal `kFingerprint` and `kSemanticFingerprint` compiled into the codec the
   owner's writer and reader link, and `generated/v3/contract.fingerprint` and the generator's `contract.fingerprint` are
   exactly those two lines. The engine and the MoE worker generate their codec from the same registry and the build
   compares their fingerprint file with the committed one (`bc_generate_serving_v3`), which the release packer re-checks for
   engine and owner.
2. The registry flattened (every documented, non-patch node: path, kind, provenance class, required, nullable, Radiance
   binding, shared-type reference) equals the compiled `kFields` table, element by element and in order.
3. Every flattened path resolves in the generated JSON Schema (through `$ref`, `items` and map values), and the schema's
   `required` agrees with the registry for each object member.

It passes on the current registry (443 fields) and fails, naming the file, when the committed fingerprint file is altered.
