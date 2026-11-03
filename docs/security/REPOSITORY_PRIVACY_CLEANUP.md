# Repository privacy cleanup

BigCherry is public. The November 2026 audit found a credential-shaped value
in seven benchmark captures and machine-specific paths and identifiers in
committed files and historical Git objects. This checklist separates fixes
that can be merged normally from work that changes published history.

Do not put the actual credential, machine address or host-specific terms in
a PR description, issue, test fixture or replacement documentation.

## Current-tree remediation

- [x] Redact the authentication value from the seven RHA10/RHA11
  `bench.stdout` files. Keep timings and benchmark output intact.
- [x] Remove the local build directory from `releases/v1.14.0.json`;
  retain the build plan ID and binary digest.
- [x] Replace RSC03's local run-file references in the serving-manifest
  specification with a sanitised, tracked evidence summary.
- [x] Remove unnecessary absolute run-directory references from current
  reference pages, while keeping run identifiers and measurements.
- [x] Add credential and local build/run path detection to the existing
  changed-file, staged-file and commit-message hygiene scanner. Add
  regression tests.
- [ ] Review other captured environment sections, benchmark JSON,
  release notes and archived documents for host-specific details before
  deciding whether additional current-tree redactions are needed.

## Credential and evidence handling

- [ ] Determine whether the token used by the local benchmark client was
  a generated test secret or a live service credential. If live or uncertain,
  revoke or rotate it using the issuing service; verify the old token is
  refused. Never paste it into the tracking record.
- [ ] Change the external benchmark harness or its capture wrapper so
  `API_KEY`, bearer credentials and other authentication values are
  redacted before logs are written. The BigCherry scanner rejects new
  leaked values at review time but cannot prevent a local log from being
  created.
- [ ] Decide whether raw evidence should be published at all. Keep
  performance measurements and reproducibility metadata, but exclude
  authentication values, hostnames, IPs, absolute user paths and
  machine-unique identifiers unless required and deliberately approved.
- [ ] Check any GitHub release assets, PR attachments and external copies
  separately; the Git object scan alone cannot inspect them.

## Published-history remediation (requires explicit coordination)

- [ ] Inventory all affected Git blobs, commits, branches and tags, including
  historical host configuration, local network addresses, the benchmark
  credential, and machine-named archive tags. Record object IDs privately,
  not leaked values in this public checklist.
- [ ] Agree the rewrite scope, freeze pushes and coordinate with owners of
  local clones, CI mirrors, forks and release tags. Preserve a restricted
  backup before any destructive operation.
- [ ] After credential rotation, use a reviewed history rewrite that removes
  the sensitive values from every affected published ref and tag. Do not
  rewrite history in an ordinary PR or force-push as part of routine CI.
- [ ] Verify the rewritten remote refs and tags, arrange for old clones to
  be recloned or reset, and request hosting-platform cache cleanup where
  appropriate. History rewriting cannot erase copies already distributed.

## Acceptance

- [ ] The owner confirms credential status and revocation, if needed.
- [x] Updated evidence logs contain no original credential; the test suite
  checks redaction without embedding that credential in source.
- [x] Local paths removed by this PR are replaced by stable artefact
  identifiers or links to tracked evidence.
- [x] Changed-file hygiene checks reject representative synthetic secrets.
- [ ] A separate full-history audit confirms the agreed published-ref
  cleanup, including tags and historical commits.

The audit and cleanup scripts used locally are not part of the public
repository; only sanitised findings and regression tests are committed.
