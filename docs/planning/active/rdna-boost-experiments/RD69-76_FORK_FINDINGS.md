# RD69-RD76 (FORK-* cluster): real findings from diffing MrLordCat/llama.cpp-rdna-lab

Recorded 2026-08-20. A dedicated investigation fork bare-cloned
`MrLordCat/llama.cpp-rdna-lab` (~17,500 commits) and keyword-searched it
against each of the 8 FORK-* items' own hypotheses. These findings kept
getting silently dropped from the individual RD69-RD76 plan items' `notes`
fields by a live, repeated concurrent-write race in the planning tool (a
second process was re-saving these items from a stale snapshot faster than
this session could persist the real content -- confirmed by re-reading the
live item state immediately after each write attempt and finding the
original stale content still in place, multiple times in a row). Recording
here, in a git-tracked file, as the durable copy until the plan items can be
safely updated (or until this doc's content is manually merged into them
once the racing process's sweep stops).

External-sources.toml's `mrlordcat-rdna-lab` `[[sources.tracked]]` entries
were also added by the same fork -- check `config/external-sources.toml`
directly for those rather than relying on this doc, since that file did not
hit the same clobbering issue during the fork's run (its edits landed and
are visible in `git diff`).

## Found real matching commits (5 of 8)

- **RD69** (FORK-VK-001, large cooperative-matrix Vulkan route) -- found
  `c399423c`. BUT: the fork's own subsequent experiments on this exact
  hypothesis found the variant either corrupts output, or has no real win
  once that corruption is fixed. Real negative evidence from the source
  itself. **Recommend closing/rejecting**, not porting.
- **RD71** -- found `41a8ca78`. Real, measured: +44.1% in one lane, +31.83%
  in another. Genuine candidate for a future patch-writing pass.
- **RD72** -- found `1fcc05da`. Real, measured: +24.53% decode. Committed
  only 5 days apart from RD71's commit and covers overlapping territory --
  check whether RD71/RD72 should be ported together as one coupled change.
- **RD73** -- found `7f2e7e4a`. Real, but the actual mechanism is a stable
  graph-cache KEY, not "prebuild widths" as the item's title/hypothesis
  describes. Re-scope the item's hypothesis to match reality before treating
  it as ready to port.
- **RD76** -- found `5aa2f049`. Real, small, low-risk build-system fix.
  Lowest-effort candidate of the 8 for a future patch-writing pass.

## Not found (3 of 8) -- reported honestly, not fabricated

- **RD70** -- no matching commit after a thorough search. Closest real
  work in the fork is RD72's territory (NEXTN-specific, not the general
  output-tensor placement RD70 was filed as). Recommend re-scoping to match
  RD72's real content, or closing as unverifiable against the actual source.
- **RD74** -- the fork actually TESTED this exact hypothesis (pinned host
  staging) and REJECTED it with real negative data. The investigation that
  led to the rejection surfaced a different, genuinely useful finding: HIP
  peer-copy corrupts data on Windows/RDNA4. Recommend re-scoping around that
  real finding instead of the original hypothesis, which the source itself
  already disproved.
- **RD75** -- no matching commit found; only unrelated upstream cherry-picks
  turned up. Recommend closing/re-scoping as unverifiable as filed.

## Process note

This is the second time in this session the planning tool's concurrent-write
race caused real research findings to be silently lost (see RE30's notes
for the first instance and the root-cause writeup). Root cause: the tool's
per-item write lock is in-process only (a `threading.RLock` in a module-level
dict), and each MCP client (each parallel fork/session) gets its own
separate server subprocess -- the lock provides no protection across
processes, and `update_item()`'s read-modify-write has no optimistic
concurrency check before its final `write_text()`. Root-caused and handed
off externally (AUDiaGentic, `audiagentic.components.planning`); not fixed
in this repo.
