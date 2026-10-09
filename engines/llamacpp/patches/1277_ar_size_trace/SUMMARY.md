# 1277_ar_size_trace

**Status:** untested
**Plan item:** PGC09

Lab-only diagnostic over 0840's adaptive AllReduce dispatcher. With
`BIGCHERRY_AR_SIZE_TRACE=<n>`, the first `<n>` calls log
`BIGCHERRY_AR_SIZE bytes=... ne0=... ne1=... type=... provider=... switch=...`
at WARN level. Unset adds one predictable branch per call. Used to find which
AllReduce sizes cause adaptive's pp1024 -3.9% (plain decode, ubatch 512) and
whether a switch threshold can separate them from the decode AllReduces.
Never part of a production recipe.
