# Rewritten commit hashes

On 2026-11-02 the commit messages of every branch and tag were rewritten to remove tool attribution trailers. Trees,
authors and dates are unchanged; only the messages differ, so every commit received a new hash. Commits and plan
items, ledger events, evidence notes and code comments written before that date may cite a hash that no longer
exists in the repository.

[`git-history-rewrite-map.txt`](git-history-rewrite-map.txt) maps each old hash to its new one (`old new`, one commit
per line, full 40-character hashes; 4,801 commits, of which 13 needed no change and kept their hash). To resolve a
cited hash, which may be abbreviated:

```
grep "^<old-hash-prefix>" docs/reference/git-history-rewrite-map.txt
```

The rewrite was verified before it was pushed: for every commit, the tree, author and date of the new commit equal
those of the old one.
