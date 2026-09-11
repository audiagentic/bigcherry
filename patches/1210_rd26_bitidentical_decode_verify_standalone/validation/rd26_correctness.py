"""RD26 correctness producer -- proves patches/1210_rd26_bitidentical_decode_verify_standalone.py's
kernel-routing changes (MMVF batch-size threshold in ggml-cuda.cu, sgemm
batch gate in llamafile/sgemm.cpp) do not change model output quality for
ordinary decode, via real perplexity (PPL) equality between subject
(patch applied) and control (patch absent) on real hardware.

IMPORTANT SCOPE LIMIT, stated honestly rather than overclaimed: this
producer does NOT prove RD26's actual determinism claim (decode (n_q=1)
and speculative-verify (n_q=n_draft+1, up to 8) batches producing
BIT-IDENTICAL logits against EACH OTHER). That claim needs a real
cross-batch-size comparison WITHIN one binary (n_q=1 output vs. the
corresponding row of an n_q=8 output), which is a materially different
test structure this producer does not implement -- and per this patch's
own module docstring, the determinism property only holds once the full
five-commit cluster lands (this patch ports only 2 of 5; the flash-attn
and RDNA4/RDNA3-specific hunks are composition-gated on other patches
not yet retained). What THIS producer proves: the two ported hunks
(which change WHICH kernel gets selected for small batches) do not
regress ordinary single-token decode output -- a real, narrower, but
still genuine data point, using the same subject/control PPL-equality
pattern as RD13/RD19/RD43 (reusing tools/bigcherry/experiment/
perplexity.py, its fifth real caller).

No bespoke control-variant worktree needed (same reasoning as RD13/
RD43): both hunks are self-contained edits with no other-file plumbing
left compiled-but-inert under a partial revert.
"""

from __future__ import annotations

import importlib

perplexity = importlib.import_module("bigcherry.experiment.perplexity")

PerplexityError = perplexity.PerplexityError
PerplexityRun = perplexity.PerplexityRun
PerplexityComparison = perplexity.PerplexityComparison
run_perplexity = perplexity.run_perplexity
require_ppl_equality = perplexity.require_ppl_equality
comparison_to_dict = perplexity.comparison_to_dict


class Rd26CorrectnessError(PerplexityError):
    """RD26 correctness evidence could not be produced, or failed to validate."""
