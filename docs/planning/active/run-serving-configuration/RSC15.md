---
id: RSC15
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-26T05:41:50.993519+00:00'
breadth: ''
skill: advanced
created-by: bigcherry-2
priority: P3
work: M
---

# Deferred: resolver input from native YAML and site YAML instead of site/profile TOML and --set

## Description

DEFERRED (owner/lead decision 2026-11-26, split out of RSC11 slice E so RSC11 can complete). bc-serve's resolver (resolve(opts, facts, &settings) in manifest_writer.cpp, config_loader, serving.cpp) still reads the site and profile TOML (--site, --profile) and --set key=value to derive the values the manifest writer uses, beside the native YAML. The end state is native YAML plus one site YAML as the only authored inputs. Not needed for the cutover criterion (no process starts from a manifest file, no manifest file is read as a launch input, no manifest-shaped input crosses to the engine or helpers) and not started.

## Steps

1. Migrate the callers first: tools/bigcherry/engine_adapters/radiance_inventory.py (`bc-serve --explain --json --set layout.name=NAME`) and tools/lab/radiance/heat-placement/heat-watch.py (`serve ... --set heat=...`) to a native YAML copy (--native-config) or an explicit explain input. 2. Move the remaining resolver inputs (site machine facts, calibration cache path, model root) to one site YAML read through the engine's YAML parser, as power-site.yaml already is (PWR08 S3, read from <work-root>/power-site.yaml). 3. Delete --site/--profile/--set, config_loader's TOML readers and the Options fields they fill; keep --detect and --schema.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free: serving_test resolver cases re-pointed to YAML inputs; bc-serve explain of every shipped config unchanged byte for byte; the Python inventory adapter and heat-watch tests pass on the new inputs. No GPU run beyond the shared acceptance.

## Effort & Risk



## Standards



## Acceptance Criteria

bc-serve accepts no --site, --profile or --set; `explain` and every start resolve from the native YAML plus the site YAML; the callers named in step 1 no longer pass --set; the existing named-config behaviour (measured-plan reuse, power-site, calibration) is unchanged.

## Notes

Created 2026-11-26 from RSC11 slice E (RSC01 note). Callers named: radiance_inventory.py --explain --set; heat-watch.py --set; resolve() inputs in manifest_writer.cpp. First site-YAML consumer already merged: power-site.yaml (#1630).

## Change Log

- 2026-11-26T05:41:50.993519+00:00 (created-by): Created by bigcherry-2
- 2026-11-26T05:42:15.528279+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
