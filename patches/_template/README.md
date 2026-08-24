# Packaged patch template

Use a package when a patch has implementation-owned validation or composition
metadata. Keep the package self-contained and deterministic.

## Do

- Keep `patch.toml` authoritative for patch identity and implementation files.
- Keep `validation.toml` authoritative for validator configuration only.
- Put validator-specific code under `validation/` and bind every artifact by digest.
- Keep Experiment Contract fields in the contract, not adapter configuration.
- Prove source-tree identity before removing any legacy flat module.
- Add focused tests for positive, negative, and tamper paths.

## Do not

- Do not hard-code a patch ID or marker in generic campaign code.
- Do not claim PASS from caller-supplied metadata without verifying artifacts.
- Do not silently overwrite a built-in validator.
- Do not put thresholds, workloads, controls, or hypotheses in adapter config.
- Do not fabricate hardware evidence or grandfather records.

A deliberately simple patch may remain flat under `patches/`; packaging is not
required merely to preserve first-class support for small patches.
