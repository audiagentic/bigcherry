# Experiments

Archived experiments are inert: nothing here is built, tested or shipped by default. Each folder has a README explaining the branch, purpose, outcome, evidence, and how to resume.

For changes **not** present on main, `change.patch` preserves the relevant delta against an explicitly named base commit. Already-shipped or superseded-only branches have a README and register entry but **no** redundant patch. Reusable, opt-in layouts can instead be placed in `engines/radiance/layouts.toml`, provided no current layout or launcher default changes.

The outcome index is `docs/research/radiance-history/EXPERIMENT-REGISTER.md`; the nine November branch dispositions are summarised in `docs/research/radiance-history/abandoned-branches-2026-11.md`. To revive a patch, create a fresh branch from the recorded base or carefully port to current main, apply with `git apply --3way`, then run the normal review, correctness, KLD and hardware gates. Do not assume historical patches apply unchanged to the current protocol.

## Branch and experiment closeout

Do not discard completed or rejected work. Before removing a branch or its worktree, merge a discoverable record of its purpose, exact revision and base, outcome, rationale, qualification limits and retained code/results. Reuse the shipped implementation when it already contains the work; otherwise retain an inert source delta or a verified durable archive reference, along with reproducibility inputs, logs or receipts needed to interpret the result. Preserve failed and negative results as well as wins. Review uncommitted and untracked material separately; an archive of commits alone is not sufficient. Keep archival code out of default recipes and builds, and move reusable diagnostics into their existing lab owner only when justified. Do not remove source refs or scratch until preservation and the archive merge are confirmed.

## Host experiments

- [GFXOFF Linux 7.0.13 reconstruction](host-kernel-gfxoff-7.0.13/README.md) — BHK01–BHK04; archived source/patch/config/build recipes and observed power evidence. No module installation, no exact installed-binary claim; generated runs stay outside Git.
