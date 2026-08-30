"""HI143: pre-promotion real-workload behavioral regression gate.

HI141 found a real production regression (mmvq:q8_0:w4:nw8:rpb1:sk0:v1
collapsing MTP draft acceptance from 93.5% to 62.1%) caused by a candidate
that measured as essentially numerically perfect against HI67's synthetic
correctness-evidence gate (nmse=5.47e-15, machine-epsilon) and a genuine
23% real latency win -- a clean, obviously-safe promotion by every metric
the tuning pipeline checked at the time. It only failed against real model
data during actual MTP speculative decoding.

This module is the converged design from an adversarially-negotiated
session with gpt-dev-agent (2026-08-29, 3 rounds, session
ses_330ae3c055084f38) -- see HI143's plan-item notes for the full
negotiation record. Summary of what was rejected and why:

- An arbitrary tolerance band on acceptance-rate delta (e.g. "<3 points is
  fine"): rejected. Under a deterministic (temp=0, fixed seed) reproducer
  there is no measurement noise to build a tolerance band around -- a real
  delta is either explained (see below) or it isn't.
- A strict per-decode-step "candidate acceptance must never decrease"
  rule: rejected. It would false-positive-reject a candidate whose
  legitimate floating-point-reassociation-driven local reordering nets out
  neutral or positive in real throughput.

Converged three-state contract, evaluated per real-workload vector:
  - generated token IDs differ from native -> HARD FAIL (model-visible
    output changed; never acceptable regardless of throughput)
  - same output, same accepted/generated draft trace -> exact PASS (cheap
    path, no further adjudication needed)
  - same output, different accepted/generated draft trace -> BEHAVIOR
    CHANGED (not an automatic reject; needs a separate real E2E throughput
    non-inferiority check against a calibrated native<->native noise
    floor before promotion, which this module does not itself perform --
    see HI143's notes for the full adjudication step)

Isolation (automatically identifying WHICH candidate in a cache caused a
HARD FAIL) is explicitly out of scope here -- see HI143's own notes for
why that is deferred as a separate follow-up (v2) rather than bundled in.

KNOWN LIMITATION, not yet fixed (gpt code review, 2026-08-29): comparing
aggregate ``(draft_n, draft_n_accepted)`` scalars cannot distinguish two
genuinely different ORDERED per-decode-step acceptance patterns that
happen to sum to the same totals (e.g. native accepting [0, 4] across two
verify steps vs. a candidate accepting [2, 2] -- both aggregate to
draft_n=8/draft_n_accepted=4 and would currently be called exact_pass).
llama-server's own ``n_accepted_per_pos`` metric does not fully solve this
either -- it is a cumulative histogram by draft position, not an ordered
per-verify-step sequence, so [1,3] and [3,1] would still collide. Closing
this gap needs new llama-server-side instrumentation exposing an ordered
per-step trace, which does not exist in the response API today -- tracked
as required follow-up work before this gate is fully sound for the
"exact_pass must mean truly identical" contract; not a reason to leave the
gate unbuilt while that instrumentation lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .server_runner import ServerRunner

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: HTR03 (GPT review round 2, 2026-08-30): the comparison SEMANTICS
#: identity -- the hard_fail/exact_pass/behavior_changed three-state
#: contract itself -- kept deliberately separate from corpus_schema_
#: version (the manifest FILE FORMAT) and a corpus edition's own content
#: digest (the exact curated CONTENTS). Persisted into every gate report
#: so a future reviewer knows exactly which comparison rules produced a
#: given verdict, independent of which corpus/profile were involved.
BEHAVIORAL_GATE_CONTRACT_VERSION = "hi143-v1"


def load_hi141_regression_vector() -> BehavioralVector:
    """The one pinned, repository-owned real-workload regression vector
    for the HI141 defect (mmvq:q8_0:w4:nw8:rpb1:sk0:v1 collapsing MTP
    draft acceptance 93.5%->62.1% after a 4096-token prefill).

    gpt code/design review (2026-08-29, session ses_330ae3c055084f38) was
    explicit that this fixture must be a repository-owned file, not a
    runtime import of the Brutus bench harness's own generate_prompt() --
    the harness is an external dependency the tuning pipeline should not
    take on just to reproduce one known regression. The text below is the
    exact, verbatim output of that harness's generate_prompt(4096) (fixed
    seed 1_000_003 + 4096, per lib/command.py's own deterministic
    construction), captured once and frozen here.

    This is intentionally NOT general corpus coverage -- it demonstrably
    catches the one known defect and establishes a real mandatory
    behavioral gate today. Broader corpus curation (real production
    traffic sampling, quartile stratification) is tracked as HI143
    follow-up work, not blocked on by this vector's existence."""
    prompt = (_FIXTURES_DIR / "hi141_qwen38_27b_mtp_4096_v1.txt").read_text(encoding="utf-8")
    return BehavioralVector(name="hi141-qwen38-27b-mtp-4096-v1", prompt=prompt, n_predict=128)


@dataclass(frozen=True)
class BehavioralVector:
    """One real-workload regression vector. ``prompt`` and ``n_predict``
    should be taken from an actual real (or production-parity) request,
    not synthetic filler text -- see HI143's notes on why a single
    synthetic/repeated prompt is not sufficient corpus coverage."""
    name: str
    prompt: str
    n_predict: int
    seed: int = 42
    # HTR03 (2026-08-30): per-vector, replacing the previous single
    # runtime-profile-wide boolean (require_mtp inferred by string-
    # matching '--spec-type' in server_args). Defaults True to preserve
    # every existing caller's behavior (including load_hi141_regression_
    # vector() below) that never set this explicitly.
    requires_mtp: bool = True


@dataclass(frozen=True)
class BehavioralTrace:
    """What one (native or candidate) run of one vector produced."""
    generated_token_ids: tuple[int, ...]
    draft_n: int
    draft_n_accepted: int


class BehavioralGateError(RuntimeError):
    pass


def run_vector(runner: ServerRunner, vector: BehavioralVector, *, require_mtp: bool = True) -> BehavioralTrace:
    """Drive one real completion request and extract the exact generated
    token sequence plus the real MTP draft/accepted counts from the
    server's own ``timings`` response field (server-common.cpp's
    ``n_draft_tokens``/``n_draft_accepted``).

    Uses ``return_tokens: true`` for the generated token IDs rather than
    reading ``completion_probabilities[i]["id"]`` -- gpt code review
    (2026-08-29, session ses_330ae3c055084f38) confirmed the latter is
    presentation/probability machinery (trimmed for stop-word handling)
    and not the canonical generated-token channel; ``return_tokens``
    exposes the raw generated sequence directly.

    Fails CLOSED, not open, when MTP telemetry is missing or zero on a
    vector that requires it (``require_mtp=True``, the default): llama-
    server only populates draft_n/draft_n_accepted when speculative
    decoding actually ran, so an absent or zero value on BOTH legs could
    otherwise silently pass as "identical" even though MTP never engaged
    at all (e.g. a misconfigured launch, or a future response-schema
    change) -- a real gap gpt's review caught in the first version of
    this function, which defaulted missing fields to 0 and let that
    become a false exact_pass."""
    response = runner.post_json(
        "/completion",
        {
            "prompt": vector.prompt, "n_predict": vector.n_predict,
            "seed": vector.seed, "temperature": 0.0, "return_tokens": True,
        },
    )
    tokens = response.get("tokens")
    if not tokens:
        raise BehavioralGateError(
            f"vector {vector.name!r}: no 'tokens' in response (requires "
            f"return_tokens=true support) -- cannot recover exact generated token IDs"
        )
    token_ids = tuple(int(t) for t in tokens)
    timings = response.get("timings")
    if timings is None:
        raise BehavioralGateError(f"vector {vector.name!r}: response has no 'timings' field at all")
    if "draft_n" not in timings or "draft_n_accepted" not in timings:
        if require_mtp:
            raise BehavioralGateError(
                f"vector {vector.name!r}: 'timings' is missing draft_n/draft_n_accepted -- "
                f"MTP did not engage or the response schema changed; refusing to silently "
                f"treat this as draft_n=0 (fail-open was a real gap in an earlier version)"
            )
        draft_n, draft_n_accepted = 0, 0
    else:
        draft_n = int(timings["draft_n"])
        draft_n_accepted = int(timings["draft_n_accepted"])
        if require_mtp and draft_n <= 0:
            raise BehavioralGateError(
                f"vector {vector.name!r}: draft_n={draft_n} <= 0 on a vector that requires "
                f"MTP -- speculative decoding did not actually engage for this request"
            )
        if not (0 <= draft_n_accepted <= draft_n):
            raise BehavioralGateError(
                f"vector {vector.name!r}: draft_n_accepted={draft_n_accepted} out of range "
                f"for draft_n={draft_n}"
            )
    return BehavioralTrace(
        generated_token_ids=token_ids, draft_n=draft_n, draft_n_accepted=draft_n_accepted,
    )


@dataclass(frozen=True)
class VectorVerdict:
    vector_name: str
    verdict: str  # "exact_pass" | "hard_fail" | "behavior_changed"
    native: BehavioralTrace
    candidate: BehavioralTrace
    first_output_divergence: int | None = None  # token index, only set for hard_fail


def token_digest(token_ids: tuple[int, ...]) -> str:
    """A stable content digest over a generated token-id sequence (HTR03
    provenance point C: persist enough per-vector detail in the gate
    report that a future reviewer never needs to re-derive it)."""
    import hashlib
    return hashlib.sha256(",".join(str(t) for t in token_ids).encode("utf-8")).hexdigest()


def compare_traces(vector_name: str, native: BehavioralTrace, candidate: BehavioralTrace) -> VectorVerdict:
    """Pure comparison logic -- no I/O, fully offline-testable. This is the
    exact three-state contract described in this module's docstring."""
    if native.generated_token_ids != candidate.generated_token_ids:
        first_divergence = next(
            (i for i, (a, b) in enumerate(zip(native.generated_token_ids, candidate.generated_token_ids)) if a != b),
            min(len(native.generated_token_ids), len(candidate.generated_token_ids)),
        )
        return VectorVerdict(
            vector_name=vector_name, verdict="hard_fail",
            native=native, candidate=candidate, first_output_divergence=first_divergence,
        )
    if (native.draft_n, native.draft_n_accepted) == (candidate.draft_n, candidate.draft_n_accepted):
        return VectorVerdict(vector_name=vector_name, verdict="exact_pass", native=native, candidate=candidate)
    return VectorVerdict(vector_name=vector_name, verdict="behavior_changed", native=native, candidate=candidate)


@dataclass
class BehavioralGateReport:
    verdicts: list[VectorVerdict] = field(default_factory=list)

    @property
    def hard_fail(self) -> bool:
        return any(v.verdict == "hard_fail" for v in self.verdicts)

    @property
    def needs_throughput_adjudication(self) -> bool:
        return any(v.verdict == "behavior_changed" for v in self.verdicts)

    def summary(self) -> dict[str, Any]:
        return {
            "hard_fail": self.hard_fail,
            "needs_throughput_adjudication": self.needs_throughput_adjudication,
            "vectors": [
                {
                    "name": v.vector_name, "verdict": v.verdict,
                    "native_draft": [v.native.draft_n, v.native.draft_n_accepted],
                    "candidate_draft": [v.candidate.draft_n, v.candidate.draft_n_accepted],
                    "first_output_divergence": v.first_output_divergence,
                }
                for v in self.verdicts
            ],
        }


def run_gate(
    *, native_runner: ServerRunner, candidate_runner: ServerRunner,
    vectors: list[BehavioralVector],
) -> BehavioralGateReport:
    """Run every vector against both a native-dispatch server and a
    candidate-cache server (both already launched/healthy -- this function
    does not manage server lifecycle) and produce the full report.

    Callers needing per-vector candidate/signature dispatch-hit coverage
    (HI143's "behavior_uncovered" requirement -- a provisional candidate
    never exercised by any vector must not silently count as PASS) should
    cross-reference GGML_HIP_DISPATCH_HIT_LOG output for the candidate
    run against the cache's own candidate set; that is deliberately kept
    out of this function since it is a property of the cache/coverage
    setup, not of any single vector comparison."""
    # HTR03 (GPT review round 2, 2026-08-30): this shared evaluator and
    # workflow.py's own _stage_replay_validate had drifted -- workflow.py
    # correctly used each vector's own requires_mtp, but this function
    # still hardcoded True, silently defeating per-vector applicability
    # for any future caller of this "shared" seam. Both must honor the
    # SAME per-vector field.
    report = BehavioralGateReport()
    for vector in vectors:
        native_trace = run_vector(native_runner, vector, require_mtp=vector.requires_mtp)
        candidate_trace = run_vector(candidate_runner, vector, require_mtp=vector.requires_mtp)
        report.verdicts.append(compare_traces(vector.name, native_trace, candidate_trace))
    return report
