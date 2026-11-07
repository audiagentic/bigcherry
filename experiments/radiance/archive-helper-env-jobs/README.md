# Archived helper-environment campaign jobs

Retired by RSC11 (one config method). These 36 campaign job files and the 7 driver scripts that ran them configure the
libr3 MoE helpers through `R3_MOE_WORKER_*` environment settings (`MIB`, `MAX_ROWS`, `SMALL_ROWS`, `HEAT`,
`ENGINE_RANKS`, `MOE_FORM` and the rest) plus raw `server_args`. Since RSC06 the engine and the helpers read their whole
configuration from a serving manifest (`--manifest M --role ...`) and no longer read those variables, and the radiance
adapter refuses to start helpers without a manifest. None of these jobs can run as written; their last campaign cells
are from 1-2 November, before that change.

They are kept as the record of the A/Bs behind the RAD06/RAD10/RAD12/RAD18/RAD20/RAD23 evidence. To run one again, port it
to a manifest (see `docs/reference/engines/radiance/SERVING_MANIFEST.md`) rather than restoring the environment settings.

Not archived, because current work or the post-bump acceptance gate uses them: `helper-ab.json`, `helper-kld.json`,
`helper-combine-ab.json`, `helper-combine-kld.json`, `rad20-burst-ab.json`, `rad20-burst-streams-ab.json`
(in `tools/lab/radiance/`; they migrate to manifests in the next RSC11 step).
