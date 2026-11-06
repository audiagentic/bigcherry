# Archived engine-only campaign jobs

Retired by RSC11 (one config method). These 19 campaign job files (the qwen35b engine MTP-depth, isolation, selection and
native variants, the qwen27b and flashnext engine MTP3 cache pairs, the bcgfx12 A/B and KLD, and the 27B engine fix A/B) and
the 2 drivers that ran them (`qwen35b-run.sh`, `qwen35b-bcgfx12-run.sh`) pass the engine its configuration as raw command-line
flags (`--tp`, `--max-model-len`, `--host-pool-mib`, ...). The engine now starts from a serving manifest, and the
campaign names one per arm; none of these was referenced by current work, the acceptance gate or a plan, and none was changed
after the October lab move.

They stay as the record of those A/Bs. To run one again, describe the arms as serving configurations (a layout and its settings,
see `docs/reference/engines/radiance/SERVING_MANIFEST.md`) instead of restoring the flags.

Kept in `tools/lab/radiance/` and migrated to manifests: `helper-ab.json`, `helper-kld.json`, `helper-combine-ab.json`,
`helper-combine-kld.json`, `rad20-burst-ab.json`, `rad20-burst-streams-ab.json` (current work or the post-bump acceptance
gate), `27b-native-kld.json` (a named post-bump acceptance quality config) and `qwen35b-engine-isolation.json` (the isolation
probe).
