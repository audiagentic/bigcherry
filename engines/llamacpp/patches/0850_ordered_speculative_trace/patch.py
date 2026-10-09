"""HI166: ordered per-verify-step speculative-decode acceptance trace.

behavioral_gate.py's pre-promotion regression gate currently compares only
AGGREGATE (draft_n, draft_n_accepted) scalars, which cannot distinguish two
genuinely different per-step work schedules that happen to sum to the same
totals -- e.g. native accepting [4,4] then [4,0] across two verify steps vs.
a candidate accepting [4,2] then [4,2]: both aggregate to draft_n=8/
accepted=4 and would currently be called identical. This is exactly the
kind of gap that let a real regression (HI141: draft acceptance
93.5%->62.1%) through a "numerically perfect" candidate once.

This patch adds a per-request ordered (draft_n, accepted_n) trace to the
existing speculative-decode verify loop and exposes it in the /completion
response's timings object, alongside the existing draft_n/draft_n_accepted
scalars it decomposes.

Design (dev-gpt-agent review req_afbaf0f27c6d4511, verified against source
before implementation, not assumed):

- Recorded directly on server_slot_stats (the per-task-result struct),
  the same place n_draft_tokens/n_draft_accepted/n_draft_verif_steps
  already live -- NOT server_slot's n_accepted_per_pos, which that struct's
  own comment explicitly says is deliberately excluded from task results.
- Acceptance in this codebase's accept-and-verify design
  (common_sampler_sample_and_accept_n / server_sample_and_accept_synth) is
  ALWAYS a prefix accept (both functions stop at the first rejected
  position) -- confirmed by GPT against the actual sampler implementations,
  not assumed. So a bare (draft_n, accepted_n) pair per step is sufficient;
  no bitmap/accepted-index vector is needed for this pin.
- Recording point: immediately after `accepted` is computed, BEFORE the
  checkpoint-rollback early return, guarded by `!slot.spec_is_replay`. This
  is the one place that captures the TRUE logical verification decision
  (original draft width, final accepted count after any rollback
  determination) exactly once -- recording at the later cumulative-
  accounting site instead would use the wrong (replay-truncated) n_draft
  on a rollback's resumed iteration, and double-count the same logical
  step. Verified against the checkpoint-rollback code path directly
  (slot.spec_is_replay = true; ...; return;) before choosing this point,
  not assumed from the aggregate accounting alone.
- The digest that makes the gate's equality check cheap
  (ordered_trace_digest) is computed CLIENT-SIDE (behavioral_gate.py), not
  here: the server cannot know in advance whether a native/candidate pair
  will match, so there is no server-side "common case" to optimize for.
  The wire format always includes the full trace (bounded by the number of
  verify steps in one completion -- not large); Python decides what to
  persist.
"""

GROUP = "core"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch  # type: ignore[import-not-found]


COMMON_H_PATCH = FilePatch(
    path="tools/server/server-common.h",
    description="add the ordered per-verify-step trace field to server_slot_stats",
    edits=(
        Edit(
            id="ordered-trace-include-utility",
            anchor=r"^#include <vector>$",
            rationale="standard-library include block",
            text="\n#include <utility>",
            guard=r"^#include <utility>$",
        ),
        Edit(
            id="ordered-trace-field",
            anchor=(
                r"    uint64_t n_draft_tokens      = 0;\n"
                r"    uint64_t n_draft_accepted    = 0;\n"
                r"    uint64_t n_draft_verif_steps = 0;"
            ),
            rationale="server_slot_stats's speculative-decoding stat fields "
                       "(no trailing comment -- distinguishes this from server_metrics's "
                       "same-named, differently-commented fields elsewhere in this file)",
            text=(
                "\n\n    // HI166: ordered per-verify-step (draft_n, accepted_n) pairs -- the\n"
                "    // aggregate scalars above cannot distinguish two different per-step\n"
                "    // work schedules that sum to the same totals. See patches/\n"
                "    // 0850_ordered_speculative_trace for why this is recorded here and not\n"
                "    // on server_slot alongside n_accepted_per_pos.\n"
                "    std::vector<std::pair<uint32_t, uint32_t>> draft_trace;"
            ),
            guard=r"draft_trace;",
        ),
    ),
)


COMMON_CPP_PATCH = FilePatch(
    path="tools/server/server-common.cpp",
    description="serialize the ordered trace into the timings response object",
    edits=(
        Edit(
            id="ordered-trace-serialize",
            anchor=(
                # String literal contents ("draft_n", "draft_n_accepted") are
                # blanked to spaces by the patch engine's noise-stripping
                # before anchor matching (see this repo's Edit docs) -- match
                # on surrounding structure only, never on the literal keys.
                r"    if \(n_draft_tokens > 0\) \{\n"
                r"        base\[.*?\]\s*= n_draft_tokens;\n"
                r"        base\[.*?\]\s*= n_draft_accepted;\n"
                r"    \}"
            ),
            rationale="the existing opt-in (speculative-decoding-only) draft-stats block "
                       "this trace decomposes",
            text=(
                "\n\n    if (!draft_trace.empty()) {\n"
                '        json trace = json::array();\n'
                "        for (const auto & step : draft_trace) {\n"
                "            trace.push_back(json::array({step.first, step.second}));\n"
                "        }\n"
                '        base["draft_trace"] = std::move(trace);\n'
                "    }"
            ),
            guard=r'base\["draft_trace"\]',
        ),
    ),
)


CONTEXT_CPP_PATCH = FilePatch(
    path="tools/server/server-context.cpp",
    description="record one ordered trace entry per true logical verify decision",
    edits=(
        Edit(
            id="ordered-trace-record",
            anchor=(
                r"                slot\.spec_i_batch\.clear\(\);\n"
                r"\n"
                r"                GGML_ASSERT\(accepted\.size\(\) >= 1\);"
            ),
            rationale="right after `accepted` is computed for this verify step, before "
                       "the checkpoint-rollback early return -- see this patch's module "
                       "docstring for why this exact point and not the later cumulative-"
                       "accounting site",
            text=(
                "\n\n                // HI166: record exactly one logical verification\n"
                "                // decision per step, using the ORIGINAL draft width\n"
                "                // (n_draft) and the post-rollback-determination accepted\n"
                "                // count. Guarded on !spec_is_replay so the resumed\n"
                "                // iteration after a checkpoint rollback (whose n_draft is\n"
                "                // the truncated replay draft, not the original) does not\n"
                "                // record a second, spurious entry for the same decision.\n"
                "                if (!slot.spec_is_replay) {\n"
                "                    slot.stats.draft_trace.emplace_back(\n"
                "                        (uint32_t) n_draft, (uint32_t) (accepted.size() - 1));\n"
                "                }"
            ),
            guard=r"slot\.stats\.draft_trace\.emplace_back",
        ),
    ),
)


PATCHES = [COMMON_H_PATCH, COMMON_CPP_PATCH, CONTEXT_CPP_PATCH]
