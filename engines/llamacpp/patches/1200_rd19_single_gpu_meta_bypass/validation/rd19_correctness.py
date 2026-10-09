"""RD19 correctness producer -- proves patches/1200_rd19_single_gpu_meta_bypass.py's
Meta-device-wrapper bypass (single GPU only) does not change model output,
via real perplexity (PPL) equality between subject (patch applied) and
control (patch absent) on real hardware.

This patch changes device-SELECTION logic only (which device object
llama_prepare_model_devices hands back for n_devices==1), never any
numerical kernel path -- so unlike RD08/RD17's genuine floating-point-
reordering claims, an exact PPL match here is the EXPECTED, not merely
hoped-for, result. A real prior promotion evidence session (2026-08-23,
gpt-dev-agent PROMOTE verdict, see this patch's own README.md) already
established this via a dedicated bench; this producer exists to let a
fresh run reproduce that conclusion through the project's now-standard
harness (reusing tools/bigcherry/experiment/perplexity.py, its fourth
real caller) rather than resting on that one historical session alone.

No bespoke control-variant worktree needed (same reasoning as RD13/
RD43): both hunks are self-contained edits to llama_prepare_model_devices
with no other-file plumbing that would stay compiled-but-inert under a
partial revert -- the project's normal baseline composition with RD19
excluded IS the correct control.
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


class Rd19CorrectnessError(PerplexityError):
    """RD19 correctness evidence could not be produced, or failed to validate."""
