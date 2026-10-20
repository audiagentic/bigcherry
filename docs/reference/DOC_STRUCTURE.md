# Per-change doc structure: patches and components

This defines the one documentation shape that a generation script can parse
reliably, for both llama.cpp patch packages (`engines/llamacpp/patches/<id>/`)
and engine components (`engines/<engine>/components/<name>/`). It extends the
conventions already used by `engines/llamacpp/patches/_template/README.md` and
`SUMMARY.md` and by radiance's per-model pages
(`docs/reference/engines/radiance/models/*.md`) rather than inventing a
parallel format.

## GitHub cannot transclude; this is a generation step

GitHub's markdown renderer has no live `#include`/transclusion mechanism: a
link renders as a link, not inlined content. "Keep the engine README's
enhancement list in sync with 150 patch docs" can only mean one of two things:
hand-maintained drift, or a script that reads the per-change docs and
regenerates a delimited section. This project chooses the latter — see
`tools/docs/generate_engine_enhancement_summaries.py` — and that script output
is committed like any other generated artifact (manifests, candidate
catalogs); it is not computed at render time.

## Required section shape

Every patch or component doc (the patch's `README.md`, or the component's
`README.md`/`SUMMARY.md`) must carry three headings, in order, somewhere in
the file. They do not need to be the only headings — existing patches such as
`0840_hybrid_allreduce_dispatch/README.md` carry a lot of narrative detail
beyond them — but these three must exist with exactly this text so a script
can find them by heading match:

```
## Summary

One or two sentences: what this changes and where.

## Outcome

One line: current state in the patch.toml/component.toml state vocabulary
(see below), plus the one-line measured result or reason, or `TODO:` if
genuinely not yet known. Never fabricate a result here.

## Detail

Everything else: mechanism, evidence, limitations, links. This section can be
as long as the existing SUMMARY.md/README.md content already is.
```

A patch's existing `SUMMARY.md` "Status" header and its `patch.toml` `state`
field remain authoritative for lifecycle state; `## Outcome` restates it in
prose for the generated index, it does not replace it.

## State vocabulary

Matches `patch.toml`'s `state` field (and `component.toml`'s `[setting.*]`
`status` field, which uses an analogous but distinct set — see below). Use
exactly these words in `## Outcome`:

- `untested` — package exists, not yet run.
- `validated` — evidence supports the claim; in a production build.
- `rejected` — measured negative or unsafe; kept on main per the "shelved work
  is kept, never deleted" rule, default off.
- `evaluated` — shelved: evidence gathered, did not pan out or was paused
  (not necessarily negative), kept on main, default off, outside the
  production set — the same "shelved"/"evaluated" language as patch state.
- `superseded` — replaced by a later patch/mechanism; kept for history.

Component settings (`component.toml` `[setting.*].status`) use: `production`,
`experimental`, `evaluated`, `diagnostic`, `tuning` — see
`docs/reference/engines/radiance/README.md`'s "Promoting a component setting"
section. A component's own top-level doc still uses the five-word patch
vocabulary above when describing the component package as a whole (e.g. "the
ngram-helper component is `evaluated`, shelved: the R9700 is compute").

## Where the doc lives

- Patch: `engines/llamacpp/patches/<id>/README.md`. (`SUMMARY.md` keeps its
  existing Status/Plan item header role per the template; it is not required
  to carry the three headings itself as long as `README.md` does.)
- Component: `engines/<engine>/components/<name>/README.md` if present,
  else `SUMMARY.md` playing that role — follow whichever file already exists
  for that component rather than creating a second one.

## What the generation script does with this

`tools/docs/generate_engine_enhancement_summaries.py` scans both doc sets,
extracts slug + the `## Outcome` one-liner + a link, and writes a delimited,
regenerated section into each engine's primary reference doc (for example
`docs/reference/engines/radiance/README.md`) between:

```
<!-- BEGIN GENERATED ENHANCEMENTS -->
...
<!-- END GENERATED ENHANCEMENTS -->
```

It warns (and can fail the run) on a patch/component directory with no doc at
all, or whose doc is missing the three headings, or whose only link target is
missing. Running it is a normal, committed step — not wired into a pre-commit
hook or CI by this change (see `docs/releases/README.md` for where it is
proposed to run in the release pipeline).
