"""RD43 correctness producer -- proves patches/1216_rd43_concurrent_join_fusion_guard.py's
op-fusion horizon cap does not change model output quality, via real
perplexity (PPL) equality between subject (1215+1216 applied) and control
(1215 alone) on a real MoE-with-shared-expert model, with
GGML_CUDA_GRAPH_OPT=1 set so HIP graph capture (the mechanism RD43's own
bug report concerns) is actually exercised.

RD43 exists only to fix a real crash ("capturing stream has unjoined
work" at cudaStreamEndCapture) that RD42's shared-expert concurrency
(patch 1215) can trigger during HIP graph capture -- so unlike RD08/RD13/
RD17, the PRIMARY proof here is that the SUBJECT run completes at all
under GGML_CUDA_GRAPH_OPT=1 (no capture abort); ppl_equality against the
control is the secondary "and it didn't change output" proof, reusing
tools/bigcherry/experiment/perplexity.py (its third real caller).

Like RD13, no bespoke control-variant worktree is needed: RD43 is one
self-contained ggml-cuda.cu edit with no other-file plumbing to leave
compiled-but-inert under a partial revert, so control = the project's
normal baseline composition WITH 1215 (RD43's own hard prerequisite,
patch.toml's REQUIRES) but WITHOUT 1216.
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

# The exact env var this patch's own docstring names as the real-hardware
# reproduction condition: "Needs a real decode run on a MoE-with-shared-
# expert model ... with GGML_CUDA_GRAPH_OPT=1 to prove no capture abort
# and output parity against the =0 baseline."
GRAPH_OPT_ENV: dict[str, str] = {"GGML_CUDA_GRAPH_OPT": "1"}


class Rd43CorrectnessError(PerplexityError):
    """RD43 correctness evidence could not be produced, or failed to validate."""
