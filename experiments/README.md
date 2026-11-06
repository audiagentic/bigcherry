# Experiments

Archived experiments are inert: nothing here is built, tested or shipped by default. Each folder has a README explaining the branch, purpose, outcome, evidence, and how to resume.

For changes **not** present on main, `change.patch` preserves the relevant delta against an explicitly named base commit. Already-shipped or superseded-only branches have a README and register entry but **no** redundant patch. Reusable, opt-in layouts can instead be placed in `engines/radiance/layouts.toml`, provided no current layout or launcher default changes.

The outcome index is `docs/research/radiance-history/EXPERIMENT-REGISTER.md`; the nine November branch dispositions are summarised in `docs/research/radiance-history/abandoned-branches-2026-11.md`. To revive a patch, create a fresh branch from the recorded base or carefully port to current main, apply with `git apply --3way`, then run the normal review, correctness, KLD and hardware gates. Do not assume historical patches apply unchanged to the current protocol.
