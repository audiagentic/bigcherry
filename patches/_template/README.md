# Packaged patch template

Every new production patch is a package. Keep the package self-contained and
deterministic; add validation files when the patch is ready to make a
correctness or performance claim.

## Do

- Keep `patch.toml` authoritative for patch identity, state, composition, and
  contract bindings.
- Keep `SUMMARY.md`'s Status/Group/Plan item header synchronized with
  `patch.toml`.
- Keep `validation.toml` authoritative for validator configuration only.
- Put validator-specific code under `validation/` and bind every artifact by digest.
- Keep Experiment Contract fields in the contract, not adapter configuration.
- Prove source-tree identity before removing any legacy flat module.
- Add focused tests for positive, negative, idempotence, path-safety, and
  tamper paths.

## Do not

- Do not hard-code a patch ID or marker in generic campaign code.
- Do not claim PASS from caller-supplied metadata without verifying artifacts.
- Do not silently overwrite a built-in validator.
- Do not put thresholds, workloads, controls, or hypotheses in adapter config.
- Do not fabricate hardware evidence or grandfather records.
- Do not hand-edit evidence to turn BLOCKED/FAIL/ERROR into PASS.

The registry's flat-module reader exists only for compatibility fixtures and
should not be used for new patches. See
`docs/reference/patches/PATCH_AUTHORING.md` and
`docs/reference/patches/PATCH_VALIDATION.md` for the complete workflow.
