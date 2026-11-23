# IQ6R-V4 converter v2 — step receipts

## Step 1: managed converter and declaration gate

- Built the converter, information tool and quantizer at the intended accelerator target and low CPU priority.
- Verified the quantizer's dynamic link to the available optimized linear-algebra runtime.
- The queued, short-leased plan-only probe returned a failure **before declaration**. The supplied draft checkpoint is block-FP8 but the draft importer accepts only a different four-bit checkpoint representation.
- No converter output artifact was written. Neither the protected checkpoint nor its archived qualification state was changed.
- Gate: **blocked**. Do not interpret the plan-only failure as a GPU crash or mislabel FP8 data to bypass validation.
