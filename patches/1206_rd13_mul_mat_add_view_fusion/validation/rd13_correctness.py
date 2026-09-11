"""RD13 correctness producer -- proves patches/1206_rd13_mul_mat_add_view_fusion.py's
RESHAPE-mediated mul_mat+add fusion does not change model output quality,
via real perplexity (PPL) equality between subject (patch applied) and
control (patch absent) on a real SSM/MoE model.

Unlike RD08/RD17, RD13 needs NO bespoke control-variant worktree: its
patch.py has exactly one Edit replacing an entire self-contained
ggml-cuda.cu block (`_OLD` -> `_NEW`) with no separate struct/kernel-
plumbing edits elsewhere that would stay compiled-but-inert if only
partially reverted (contrast RD08's VDR=2 struct fields, RD17's
common.cuh/mmvq.cu additions). Reverting the whole patch IS the correct
control -- which is exactly what this project's existing generic
validation-domain control_src (the source composition with this patch
excluded) already is. So this producer takes already-built control/
subject llama-perplexity binaries as arguments rather than materializing
its own variant pair; the caller (validation_campaign.py's generic
campaign flow) already builds both from the same immutable base.

Uses tools/bigcherry/experiment/perplexity.py (extracted from RD17's
producer, this being its second real caller) rather than duplicating the
run/compare logic.
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


class Rd13CorrectnessError(PerplexityError):
    """RD13 correctness evidence could not be produced, or failed to validate."""
