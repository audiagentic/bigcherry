"""Per-promoted-key execution audit: does a tuned kernel actually launch?

WHY THIS EXISTS. HI160 established that an exact replay cache hit is NOT
evidence a tuned kernel ran: the resolver revalidates a cached candidate
after an exact hit (can_execute, arch support, blacklist, transform
applicability) and can substitute native. A run can therefore report exact
hits and launch native for every one of them. Separately, the HI168
production-shaped baseline found that GPU0, GPU1 and gfx1030 replay
generally failed to beat BC native end-to-end, while the dual-XTX 27B and
gfx1201 results remain exploratory (`performance_admitted=false` throughout,
and the gfx1201 prompt-processing deltas are large enough to be an execution-
identity warning rather than a performance claim).

That combination -- "cache hit proves nothing" plus "replay generally did
not win" -- means the honest next question is NOT "do we need more/better
candidates?". It is "did the candidates we already promoted actually launch,
and if they did, did they help?". Widening the candidate search before
answering that risks the exact failure this project has already hit twice:
a larger search space finds more candidates that look like winners for
reasons that have nothing to do with being faster (HI166's speculative
work-equivalence problem is the sibling case).

This module builds ``hip-tuning-execution-audit.jsonl``: one row per
PROMOTED key (dispatch signatures where the tuner's own promotion decision
picked something other than native), joining:

  - promoted.jsonl        -- tune_promotion.py's own record: native/winner
                              candidate identity, promotion_status,
                              improvement_pct (the TUNER's isolated
                              measurement, not an in-situ or E2E number).
  - the replay hit log     -- GGML_HIP_DISPATCH_HIT_LOG output from a
                              GGML_HIP_REPLAY_DIAGNOSTICS build. Recorded at
                              ggml_hip_replay_record_hit(), which fires at
                              exactly the point a cached candidate survives
                              revalidation and becomes the binding that later
                              increments HI160's final_tuned_launches -- so a
                              hit-log entry for a dispatch digest IS actual
                              launch evidence, not merely a cache lookup.
  - an optional e2e verdict -- supplied by the caller (e.g. transcribed from
                              a HI168-style baseline doc), never invented
                              here. Absence is recorded explicitly rather
                              than defaulting to a claim of improvement.

Deliberately NOT attempted: per-key GPU-time-in-production. Nothing in this
project's current artifacts measures how long an individual promoted
candidate actually runs inside the real serving graph (see HI168's runbook,
section 16, for what that would require: rocprofv3 kernel-fraction data
joined per dispatch digest). A row's ``launch_count`` is real; its
``isolated_improvement_pct`` is the tuner's own microbenchmark claim; there
is no fabricated GPU-time-saving field here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Classification(str, Enum):
    """What actually happened to a promoted key, in production terms.

    Named to match the runbook's section 7/13 vocabulary exactly, so a reader
    moving between the audit output and the runbook does not have to
    translate.
    """

    # Promoted "winner" resolves to the same implementation identity as
    # native -- an alias, not a real override. No performance claim is even
    # possible; the fix is to eliminate the duplicate identity.
    SAME_NATIVE = "SAME_NATIVE"

    # Actual launch evidence confirms the promoted candidate ran, it is
    # genuinely different from native, and the ONLY evidence available (the
    # tuner's isolated measurement) says it is faster. This is NOT an
    # end-to-end claim -- see `e2e_verdict`.
    DIFFERENT_FASTER = "DIFFERENT_FASTER"

    # Actual launch evidence confirms the promoted candidate ran, it is
    # genuinely different from native, and a supplied E2E verdict says it
    # regressed production. Only reachable when the caller supplies e2e
    # evidence; never inferred from the isolated measurement alone.
    DIFFERENT_SLOWER = "DIFFERENT_SLOWER"

    # The hit log's actual launched candidate does not match the promoted
    # winner (a miss, a fallback after failed revalidation, or a downgrade to
    # native at launch time). The cache resolved; the intended kernel did not
    # run.
    FALLBACK = "FALLBACK"

    # No hit-log entry exists for this dispatch digest at all. Either the key
    # was never exercised by the workload the hit log was captured against,
    # or no diagnostic build/run was ever performed for it. Absence of
    # evidence, not evidence of failure -- but it blocks any performance
    # claim for this key just the same.
    NOT_EXECUTED = "NOT_EXECUTED"


@dataclass(frozen=True)
class HitRecord:
    dispatch: str
    signature: str
    candidate: str
    calls: int


@dataclass(frozen=True)
class AuditRow:
    dispatch: str
    signature: str
    native_candidate: str
    replay_candidate: str  # the tuner's promoted "winner"
    actual_launched_candidate: str | None
    launch_count: int
    classification: Classification
    isolated_improvement_pct: float | None
    e2e_verdict: str  # "improved" | "regressed" | "not_measured" (default)
    e2e_source: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "dispatch": self.dispatch,
            "signature": self.signature,
            "native_candidate": self.native_candidate,
            "replay_candidate": self.replay_candidate,
            "actual_launched_candidate": self.actual_launched_candidate,
            "launch_count": self.launch_count,
            "classification": self.classification.value,
            "isolated_improvement_pct": self.isolated_improvement_pct,
            "e2e_verdict": self.e2e_verdict,
            "e2e_source": self.e2e_source,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_promoted(path: str | Path) -> list[dict[str, Any]]:
    """Only PROMOTED rows -- tune_promotion.py's own decision that the winner
    differs from native. A row where the tuner itself kept native is not an
    execution question; there is nothing to audit."""
    rows = _read_jsonl(Path(path))
    out = []
    for row in rows:
        if row.get("kind") == "header":
            continue
        if row.get("promotion_status") != "promoted":
            continue
        if not row.get("dispatch") or not row.get("winner"):
            continue
        out.append(row)
    return out


def load_hit_log(path: str | Path | None) -> dict[str, HitRecord]:
    """Keyed by dispatch digest. Missing/empty path is a legitimate input --
    it means no diagnostic run was captured -- not an error."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, HitRecord] = {}
    for row in _read_jsonl(p):
        dispatch = row.get("dispatch")
        if not dispatch:
            continue
        out[dispatch] = HitRecord(
            dispatch=dispatch,
            signature=row.get("signature", ""),
            candidate=row.get("candidate", ""),
            calls=int(row.get("calls", 0)),
        )
    return out


def load_e2e_verdicts(path: str | Path | None) -> dict[str, str]:
    """Optional caller-supplied per-dispatch E2E verdicts.

    Format: {"<dispatch-digest>": "improved" | "regressed", ...}. This
    function does not compute or infer verdicts -- see the module docstring
    on why per-key production GPU time is not available from current
    artifacts. A caller with a HI168-style baseline maps its findings onto
    dispatch digests explicitly, or leaves this empty.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected a JSON object of dispatch -> verdict")
    for k, v in data.items():
        if v not in ("improved", "regressed"):
            raise ValueError(f"{p}: verdict for {k!r} must be 'improved' or 'regressed', got {v!r}")
    return dict(data)


def classify(
    *,
    native_candidate: str,
    replay_candidate: str,
    hit: HitRecord | None,
    e2e_verdict: str | None,
) -> Classification:
    if native_candidate == replay_candidate:
        return Classification.SAME_NATIVE
    if hit is None:
        return Classification.NOT_EXECUTED
    if hit.candidate != replay_candidate:
        return Classification.FALLBACK
    if e2e_verdict == "regressed":
        return Classification.DIFFERENT_SLOWER
    # e2e_verdict == "improved" or not supplied: the only evidence available
    # either way is the tuner's isolated measurement, which promotion already
    # required to be positive. Distinguishing a genuine production win from
    # "isolated win, E2E unknown" is the job of `e2e_verdict`, not of this
    # classification -- see the row's own e2e_verdict field.
    return Classification.DIFFERENT_FASTER


def build_audit(
    *,
    promoted_path: str | Path,
    hit_log_path: str | Path | None = None,
    e2e_verdicts_path: str | Path | None = None,
) -> list[AuditRow]:
    promoted = load_promoted(promoted_path)
    hits = load_hit_log(hit_log_path)
    e2e = load_e2e_verdicts(e2e_verdicts_path)

    rows: list[AuditRow] = []
    for row in promoted:
        dispatch = row["dispatch"]
        native_candidate = row.get("native", "")
        replay_candidate = row["winner"]
        hit = hits.get(dispatch)
        verdict = e2e.get(dispatch)
        classification = classify(
            native_candidate=native_candidate,
            replay_candidate=replay_candidate,
            hit=hit,
            e2e_verdict=verdict,
        )
        rows.append(AuditRow(
            dispatch=dispatch,
            signature=row.get("signature", ""),
            native_candidate=native_candidate,
            replay_candidate=replay_candidate,
            actual_launched_candidate=hit.candidate if hit else None,
            launch_count=hit.calls if hit else 0,
            classification=classification,
            isolated_improvement_pct=row.get("improvement_pct"),
            e2e_verdict=verdict or "not_measured",
            e2e_source=str(e2e_verdicts_path) if verdict else None,
        ))
    return rows


@dataclass(frozen=True)
class AuditSummary:
    total: int
    by_classification: dict[str, int]

    @property
    def not_executed(self) -> int:
        return self.by_classification.get(Classification.NOT_EXECUTED.value, 0)

    @property
    def fallback(self) -> int:
        return self.by_classification.get(Classification.FALLBACK.value, 0)

    @property
    def unproven_fraction(self) -> float:
        """Fraction of promoted keys with NO evidence they ever launched as
        intended -- NOT_EXECUTED or FALLBACK. This is the number the stop
        condition below is keyed on."""
        if self.total == 0:
            return 0.0
        return (self.not_executed + self.fallback) / self.total


def summarize(rows: Iterable[AuditRow]) -> AuditSummary:
    rows = list(rows)
    by_class: dict[str, int] = {}
    for r in rows:
        by_class[r.classification.value] = by_class.get(r.classification.value, 0) + 1
    return AuditSummary(total=len(rows), by_classification=by_class)


def write_audit(rows: Iterable[AuditRow], output_path: str | Path) -> None:
    p = Path(output_path)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_json(), sort_keys=True) + "\n")
