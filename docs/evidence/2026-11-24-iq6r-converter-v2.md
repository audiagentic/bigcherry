# IQ6R-V4 converter v2 — step receipts

## Step 1: managed converter and declaration gate

- Built the converter, information tool and quantizer at the intended accelerator target and low CPU priority.
- Verified the quantizer's dynamic link to the available optimized linear-algebra runtime.
- The queued, short-leased plan-only probe returned a failure **before declaration**. The supplied draft checkpoint is block-FP8 but the draft importer accepts only a different four-bit checkpoint representation.
- No converter output artifact was written. Neither the protected checkpoint nor its archived qualification state was changed.
- Gate: **blocked**. Do not interpret the plan-only failure as a GPU crash or mislabel FP8 data to bypass validation.

## Step 2: read-only source-head view

The DFlash2 draft-head copy is taken from the target's original unsplit vocabulary head, not from the drafter's head or the target's classified output matrices.

The original head physically remains in an existing checkpoint shard. The view helper creates a separate linked checkpoint directory and adds only the missing source-head index entry; it retains the existing 2-bit draft-head recipe without rewriting weights or modifying the protected checkpoint.

The helper validates the source shard header, prohibits writing into either input, rejects inconsistent existing views, and supports idempotent rechecks. Four card-free unit tests pass. The format-import blocker in Step 1 remains.
