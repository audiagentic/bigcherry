"""Stage advisories: what a campaign's result means, and what NOT to conclude.

WHY THIS EXISTS. Over one session an agent independently rediscovered three
things this repository already knew:

- HI130 records "Any FUTURE benchmark ... with speculative decoding MUST
  verify draft_acceptance is IDENTICAL across compared configs before
  trusting tg128/tg512 deltas -- this was the root cause of every false
  conclusion in this investigation." A benchmark was nonetheless run and
  reported before that check, twice.
- `bigcherry ab-benchmark` already provides paired, interleaved A/B with
  build-composition verification. A worse one was written from scratch in
  bash, and its first result had to be retracted for a confound
  ab-benchmark already handles.
- The behavioural corpus covered a single generation length, so a gate
  built to catch acceptance regressions certified a cache that regressed
  acceptance at every longer length.

None of that was hidden. It was written down in plan items, docstrings and
a CLI the agent did not read. Knowledge that is only discoverable by
already knowing where to look does not get looked at, so this module puts
it where the result appears, at the moment it becomes relevant.

DESIGN CONSTRAINTS, each learned the hard way:

1. Not the C++ runtime. llama-server installs a log callback that swallows
   the library's GGML_LOG_INFO lines entirely -- dispatch counter reports
   were invisible for hours because of it. An advisory emitted from the
   dispatch layer would go nowhere.
2. Therefore this lives in the campaign tooling, which never ships. The
   "must be disabled in production builds" requirement is satisfied
   structurally rather than by a compile flag that could be got wrong.
3. Conditional, never blanket. An advisory that always prints is banner
   text, and banner text is not read. Each one below is gated on a
   condition actually present in the receipt, so seeing one means it
   applies to the run in front of you.
4. stderr, not stdout, and suppressed entirely under --json, so machine
   consumers are unaffected.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Advisory:
    """One conditional note tied to something observed in the receipt."""

    tag: str          # short stable identifier, e.g. "mtp-acceptance"
    headline: str     # what was observed
    body: tuple[str, ...]   # what to do or not conclude


def _coverage_families(coverage: Any) -> dict[str, dict]:
    if not isinstance(coverage, dict):
        return {}
    fams = coverage.get("families")
    return fams if isinstance(fams, dict) else {}


def _replay(coverage: Any) -> dict:
    if not isinstance(coverage, dict):
        return {}
    r = coverage.get("replay")
    return r if isinstance(r, dict) else {}


def advisories_for_campaign(
    *,
    replay_coverage: Any = None,
    recovery_result: Any = None,
    corpus_vectors: Iterable[Any] = (),
    inventory: Any = None,
) -> list[Advisory]:
    """Build the advisories that apply to one completed campaign.

    Every argument is optional: a caller that cannot supply one simply gets
    fewer advisories, rather than a crash at the end of a campaign that
    already spent real GPU hours.
    """
    out: list[Advisory] = []
    replay = _replay(replay_coverage)

    # --- proving the winners actually run -------------------------------
    if replay:
        out.append(Advisory(
            tag="activation-evidence",
            headline="A promoted cache does not prove tuned kernels will run.",
            body=(
                "exact > 0 is NOT sufficient: the resolver revalidates a cached",
                "candidate after an exact hit (can_execute, arch support, blacklist,",
                "transform applicability) and can substitute native. A run can report",
                "exact hits and launch native for every one of them.",
                "The only sufficient evidence is final_tuned_launches > 0, counted at",
                "the executor. Build with GGML_HIP_DISPATCH_DIAGNOSTICS=ON and read",
                "the coverage JSON (GGML_HIP_DISPATCH_COVERAGE=<path>) -- the log",
                "channel is swallowed by llama-server and will appear silent.",
            ),
        ))

    # --- MTP acceptance: HI130's rule -----------------------------------
    scenarios = []
    lengths = []
    for v in corpus_vectors or ():
        n = getattr(v, "n_predict", None)
        if n is not None:
            lengths.append(int(n))
        s = getattr(v, "scenario", None)
        if s:
            scenarios.append(str(s))
    if any("mtp" in s.lower() for s in scenarios):
        out.append(Advisory(
            tag="mtp-acceptance",
            headline="This workload uses MTP speculative decode.",
            body=(
                "HI130: verify draft acceptance is IDENTICAL across the arms being",
                "compared BEFORE trusting any tg delta. Differing acceptance means the",
                "arms did different amounts of work and the throughput comparison is",
                "void, however clean the numbers look.",
                "This is not hypothetical: a promoted cache measured 0.94734 -> 0.86391",
                "acceptance, bit-identical across 16 cells, which read as an 9% tg512",
                "regression that was really a change in work done.",
            ),
        ))

    # --- corpus coverage vs what will be served -------------------------
    if lengths:
        out.append(Advisory(
            tag="corpus-coverage",
            headline=(
                "The behavioural gate certified generation length(s): "
                + ", ".join(str(n) for n in sorted(set(lengths)))
                + "."
            ),
            body=(
                "Behaviour at any OTHER generation length is uncertified. An earlier",
                "corpus held a single 128-token vector; the cache it passed regressed",
                "acceptance at 512 and 2048, and 128 was the only length that improved.",
                "If production serves lengths outside this set, publish a new corpus",
                "edition covering them (editions are immutable -- add, never edit).",
            ),
        ))

    # --- recovery exhausted its options ---------------------------------
    recs = []
    if isinstance(recovery_result, dict):
        recs = recovery_result.get("retune_recommendations") or []
    if recs:
        sigs = ", ".join(
            str(r.get("signature_dispatch", "?"))[:12] for r in recs if isinstance(r, dict)
        )
        out.append(Advisory(
            tag="alternatives-exhausted",
            headline=f"Recovery exhausted its candidate alternatives for: {sigs}.",
            body=(
                "Those signatures fell back to native, so the cache is SAFE but leaves",
                "measured performance on the table.",
                "HTR04's governing rule: recovery failure != retune. The correct chain",
                "is failure -> native fallback -> quantify the real loss -> only if",
                "material, decide whether the measurements are stale or the candidate",
                "search space is genuinely exhausted -> only then recommend a retune.",
                "Never retune directly on this signal: it risks respending GPU time",
                "remeasuring a pool already known to be undeployable.",
                "Repeated exhaustion on one candidate family is evidence about the",
                "FAMILY, not about unlucky instances.",
            ),
        ))

    # --- families carrying real work with nothing tuned -----------------
    fams = _coverage_families(replay_coverage)
    tuned_types = {}
    if isinstance(inventory, dict):
        for key in ("mmq_types", "mmvq_types", "mmvf_types", "mmf_types"):
            tuned_types[key.replace("_types", "")] = inventory.get(key) or []
    if tuned_types:
        untuned = [
            name for name, stats in fams.items()
            if isinstance(stats, dict)
            and int(stats.get("executed") or 0) > 0
            and not tuned_types.get(name, [])
        ]
        if untuned:
            out.append(Advisory(
                tag="untuned-families",
                headline=(
                    "Families executing real work with NO tuned candidates: "
                    + ", ".join(sorted(untuned)) + "."
                ),
                body=(
                    "Tuning cannot improve these at all, however good the dispatch path",
                    "becomes. Before optimising lookup overhead, check what share of",
                    "dispatches these families carry -- effort belongs where the work is.",
                    "Use `bigcherry kernel-fraction` over a rocprofv3 kernel trace to get",
                    "the time share, and `bigcherry impact` to predict the saving a tuned",
                    "candidate set would actually deliver.",
                ),
            ))

    # --- how to benchmark this cache ------------------------------------
    out.append(Advisory(
        tag="use-the-harness",
        headline="Benchmarking this cache: use the maintained tooling.",
        body=(
            "`bigcherry ab-benchmark` is paired and interleaved and verifies build",
            "composition (--stock-cmake-cache / --patched-cmake-cache). Do not write",
            "a harness: one written from scratch here reimplemented order balancing",
            "badly and its first result was retracted for a confound ab-benchmark",
            "already handles.",
            "Measure on a build with GGML_HIP_DISPATCH_DIAGNOSTICS=OFF; take activation",
            "evidence from a separate diagnostics-ON build of the SAME revision.",
        ),
    ))
    return out


def render(advisories: list[Advisory]) -> str:
    if not advisories:
        return ""
    lines = ["", "-- advisories " + "-" * 62]
    for a in advisories:
        lines.append(f"[{a.tag}] {a.headline}")
        lines.extend("    " + b for b in a.body)
        lines.append("")
    lines.append("Suppressed under --json. Tooling only; never present in a shipped build.")
    return "\n".join(lines)


def emit(advisories: list[Advisory], stream=None) -> None:
    """Write to stderr so machine-readable stdout stays clean."""
    text = render(advisories)
    if text:
        print(text, file=stream or sys.stderr)
