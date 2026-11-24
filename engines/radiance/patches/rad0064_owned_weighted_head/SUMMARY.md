# rad0064 — uneven tensor-parallel ownership with a vocabulary-weighted head (VTP57)

**Status:** untested
**Plan item:** VTP57

The uneven ownership split of the 27B layout (two XTX and one R9700) refused any container whose output head is stored as width classes: it read only a plain `output.weight`. The three-rank vocabulary-weighted head is built so that every class has the same number of rows on every rank (a multiple of 256 x 3, padded with dropped rows), so the classes shard by rows exactly as in the even split; only the vocabulary-sized tensors need the explicit span of the rank's vocabulary share: the token table and the id-to-row map. This patch lets the ownership path declare those two spans instead of refusing, and leaves the plain-head path unchanged.
