# Integrity index added 20 November 2026

This index covers the retained, path-sanitized results in this historical
work-root recovery bundle. The original raw work-root inputs were not
preserved, and machine-specific paths were replaced when the archive was
created. These digests identify **the retained checked-in bytes**, not the
original pre-sanitization files or independently validated GPU outcomes.

`integrity-digests.tsv` stores two lowercase SHA-256 digests per line:
`SHA256(relative-POSIX-path) TAB SHA256(file-contents)`, sorted by the
first digest. Relative paths are hashed rather than copied into a new text
file because some historical directory names still identify a physical
machine. This avoids weakening repository forbidden-terms rules. The
`integrity-index.json` file identifies the version, entry count and digest
of the index. Both index files exclude themselves from content coverage.

For a full verification, run the checked-in validator:

`PYTHONPATH=tools python -m unittest tools.tests.core.test_recovery_bundle_integrity`

This is a **legacy recovery integrity record**, not a new platform run
record or non-patch hardware-verification manifest. Do not infer original
toolchain, source hashes, pass/fail outcomes or byte-exact raw data beyond
what each preserved file itself proves. If retained files are changed,
regenerate and review the digest index in the same change.
