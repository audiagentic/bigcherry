# N1 — native Radiance rank-aware device-code eligibility (inactive)

This **managed source patch** modifies Radiance's existing C++ kernel
registry/selection and native builder only. The registry can be given
an **explicit, ordered** set of physical device IDs before loading
plugins. Each kernel row retains a per-rank compatibility mask
queried from its plugin's actual code-object registration through
\`dev_code_runs\`, rather than trusting a text build-target claim.
The native \`RadBuildCtx.rank\` is passed through the builder to the
selector without mutating a global rank field across worker threads.
An incompatible row is rejected with a named rank/plugin reason.

**Not activated:** No engine startup code calls the new API, and
\`usable_devices\` still restricts GPUs to one architecture. Ordinary
TP1/TP2 and the production recipe keep the existing single-device
plugin filtering. Do not bypass that guard or claim mixed gfx1100/
gfx1201 inference until the next N1 slice addresses per-device
facts, per-rank graph traces, fatbinary code coverage and full
startup admission.

Acceptance in this sub-slice: managed patch applied/idempotent to
the pinned v1.17.0 source, C++23 syntax build for modified units,
fake rank/device invalid-ID tests and no-production-recipe tests.
No GPU usage, model serving, or hot-path dispatch changes.
