# rad0063 — vocabulary-head builder for any tensor-parallel world (VTP57)

**Status:** untested
**Plan item:** VTP57

The vocabulary-weighted head tool builds its width classes for two ranks only (it splits the vocabulary in halves). The engine
shards a class by rows over the rank count and needs every rank's slice of a class to be the same number of rows, a multiple
of 256, holding exactly the ids of that rank's vocabulary share; a row whose id lies outside the rank's share is dropped by the
head kernel (it is never written, never read back).

This patch gives the tool a `--world N` parameter (default 2, byte-identical output to the old tool when the halves already hold
equal, 256-multiple row counts). For N ranks it cuts the vocabulary by the engine's own share rule (an even share when N divides the
vocabulary, otherwise ceil(V/N) rounded up to 128 rows, the last rank short), keeps each rank's rows of a class in ascending id,
pads the shorter ranks' slices of a class to the longest slice rounded up to 256 rows with rows whose id is -1 (outside every
share) and whose weights repeat that rank's own rows, and writes `vrow` as the class width in the top four bits and the row
rank-major (rank * rows per rank + position) below, as the staging gather reads it.

Status: untested; the unit test applies the patch to a copy of the pinned tool and builds synthetic heads.
