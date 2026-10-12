# 1286_draft_local_shared_tensors

**Status:** untested (default off; not production-qualified)
**Plan item:** QFP43 (PNRO11 target-wide replication proposal closed; BCOP159)

## Implemented mechanism

For a single-device DFlash/DSpark draft borrowing `tok_embd`, `output` or `output_s` from a tensor-split target, copy only borrowed tensors whose buffer device is `Meta()` to a draft-local backend buffer at context creation. The draft context owns the copies; setup gathers each full tensor through temporary host memory and uploads it once. Draft-owned tensors and non-Meta sources are unchanged. No P2P, per-token cache, target-wide mirroring or verified performance gain is implied.

## Unresolved correctness fence (2026-10-12)

`patch.py`'s `dflash-output-copy` replacement in **both** graph builders chooses `output_s` based on `cparams.other_output`, not `cparams.other_output_s`. Independent weight/scale Meta residency can therefore drop a non-Meta scale or ignore a copied Meta scale. A 12-case host pointer-selection model found two conditional mismatches; no deployed mixed-residency GPU failure has been reproduced. `tools/tests/patch/test_1286_draft_local_shared_tensors.py` currently asserts the replacement string and composition/idempotence, not this semantic contract.

QFP43 owns the bounded selector truth table, per-tensor backend accessibility, full-copy/host-memory accounting, graph lifetime, repeated-request and logits/acceptance qualification. Keep the patch untested until those gates pass. The distinct DFlash2 in-graph `TOP_K` on axis-0 split logits when the **draft itself** is Meta-split is not fixed by this single-device-only patch; upstream llama.cpp #27858 is an open CPU-selector proposal. See PNRO11 and BCOP159.
