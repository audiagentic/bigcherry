"""HI67 slices 2c/3: CPU-reference correctness evidence generator.

Runs test-backend-ops twice per deterministic seed -- once forced to native,
once forced to a candidate -- and reduces both runs to the per-seed and
aggregated rows sql/dispatch-db.sql's correctness_evidence/correctness_
evidence_seed tables (schema 6) expect. Implements the RV49 contract as
adjudicated for its implementation details in RV77 (patch-management
review, 2026-08-20/21):

    E_N = NMSE(R, N) <= T
    E_C = NMSE(R, C) <= E_N + headroom_fraction * (T - E_N)
    max_abs(C, R) <= max_abs(N, R)

using the WORST result across >=3 deterministic seeds.

Dispatch-mode mechanism (source-verified 2026-08-21 against hip-autotune-
dispatch.cu, correcting an unverified assumption in RV77's own design --
GPT's suggestion of `GGML_HIP_FORCE_CANDIDATE=native` does not work):

- ``ggml_hip_dispatch_resolve()`` returns plain native immediately (line
  ~422, `if (mode == GGML_HIP_DISPATCH_MODE_NATIVE || !native.valid) return
  resolved;`) BEFORE the forced-candidate check is ever reached. No
  candidate is registered under the literal stable_name "native" either
  (native-wrapper candidates are found via ``ggml_hip_registry_native()``,
  by source_class, never by name) -- ``GGML_HIP_FORCE_CANDIDATE=native``
  would silently fail to match and fall back to ordinary resolution.
- With GGML_HIP_DISPATCH_MODE unset entirely, ggml_hip_parse_mode()
  defaults to NATIVE too -- so the candidate run must ALSO set
  GGML_HIP_DISPATCH_MODE to something other than "native" or the
  forced-candidate branch is never reached at all.

So: N = GGML_HIP_DISPATCH_MODE=native (no force var, and none is needed).
    C = GGML_HIP_DISPATCH_MODE=replay + GGML_HIP_FORCE_CANDIDATE=<name>.

Requires patches/1222_hi67_deterministic_test_backend_ops_seed.py and
patches/1223_hi67_machine_readable_correctness_metrics.py to be applied to
the binary under test -- BIGCHERRY_TEST_DETERMINISTIC_SEED, BIGCHERRY_REF_
DIGEST and BIGCHERRY_CORRECTNESS_METRIC do not exist without them.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

CONTRACT_VERSION = "hi67-rv49-v1"
DEFAULT_HEADROOM_FRACTION = 0.5
MIN_SEEDS = 3


class EvidenceError(RuntimeError):
    """Evidence generation failed closed -- never a partial or unproven row."""


_REF_DIGEST_RE = re.compile(
    r"BIGCHERRY_REF_DIGEST name=(?P<name>\S+) call_index=(?P<call_index>\d+) "
    r"digest=(?P<digest>[0-9a-fA-F]+) nels=(?P<nels>\d+)"
)

_METRIC_RE = re.compile(
    r"BIGCHERRY_CORRECTNESS_METRIC op=(?P<op>\S+) tensor=(?P<tensor>\S+) "
    r"backend1=(?P<backend1>\S+) backend2=(?P<backend2>\S+) "
    r"err=(?P<err>\S+) max_abs=(?P<max_abs>\S+) threshold=(?P<threshold>\S+) "
    r"n=(?P<n>\d+)"
    # HI83: added by patches/1223's backend1_digest/backend2_digest extension.
    # Optional so this parser stays usable against older builds that predate
    # that extension (e.g. HI67 evidence captured before HI83 landed).
    r"(?: backend1_digest=(?P<backend1_digest>[0-9a-fA-F]+)"
    r" backend2_digest=(?P<backend2_digest>[0-9a-fA-F]+))?"
)


@dataclass(frozen=True)
class RefDigest:
    name: str
    call_index: int
    digest: str
    nels: int


@dataclass(frozen=True)
class CorrectnessMetric:
    op: str
    tensor: str
    backend1: str
    backend2: str
    err: float
    max_abs: float
    threshold: float
    n: int
    # HI83: exact backend-output byte digests -- None against a build that
    # predates the 1223 digest extension.
    backend1_digest: str | None = None
    backend2_digest: str | None = None


def parse_ref_digests(stderr_text: str) -> list[RefDigest]:
    return [
        RefDigest(m["name"], int(m["call_index"]), m["digest"], int(m["nels"]))
        for m in _REF_DIGEST_RE.finditer(stderr_text)
    ]


def parse_correctness_metrics(stderr_text: str) -> list[CorrectnessMetric]:
    return [
        CorrectnessMetric(
            op=m["op"], tensor=m["tensor"], backend1=m["backend1"], backend2=m["backend2"],
            err=float(m["err"]), max_abs=float(m["max_abs"]), threshold=float(m["threshold"]),
            n=int(m["n"]),
            backend1_digest=m["backend1_digest"].lower() if m["backend1_digest"] is not None else None,
            backend2_digest=m["backend2_digest"].lower() if m["backend2_digest"] is not None else None,
        )
        for m in _METRIC_RE.finditer(stderr_text)
    ]


def run_test_backend_ops(
    binary: Path, *, op_filter: str, seed: int, dispatch_mode: str,
    forced_candidate: str | None, env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> subprocess.CompletedProcess:
    if seed == 0:
        raise EvidenceError(
            "seed must be nonzero -- 0 leaves BIGCHERRY_TEST_DETERMINISTIC_SEED "
            "unset in the patched test-backend-ops, disabling deterministic mode "
            "entirely (see patches/1222_hi67_deterministic_test_backend_ops_seed.py)"
        )
    run_env = dict(env or {})
    run_env["BIGCHERRY_TEST_DETERMINISTIC_SEED"] = str(seed)
    run_env["GGML_HIP_DISPATCH_MODE"] = dispatch_mode
    if forced_candidate is not None:
        run_env["GGML_HIP_FORCE_CANDIDATE"] = forced_candidate
    else:
        run_env.pop("GGML_HIP_FORCE_CANDIDATE", None)
    argv = [str(binary), "test", "-o", "MUL_MAT", "-p", op_filter]
    return runner(argv, capture_output=True, text=True, env=run_env)


def find_digest_for_tensor(stderr_text: str, tensor_name: str) -> RefDigest | None:
    matches = [d for d in parse_ref_digests(stderr_text) if d.name == tensor_name]
    return matches[0] if matches else None


def find_metric_for_tensor(stderr_text: str, tensor_name: str) -> CorrectnessMetric | None:
    matches = [m for m in parse_correctness_metrics(stderr_text) if m.tensor == tensor_name]
    return matches[0] if matches else None


@dataclass(frozen=True)
class SeedEvidence:
    seed: int
    reference_digest: str
    e_n_nmse: float
    e_c_nmse: float
    max_abs_native: float
    max_abs_candidate: float
    native_execution_status: str
    candidate_execution_status: str
    # HI67 threshold-authority fix: the upstream correctness threshold T, as
    # ACTUALLY EMITTED by test-backend-ops (BIGCHERRY_CORRECTNESS_METRIC's own
    # threshold=... field) -- never a caller-supplied Python float. Before
    # this field existed, threshold_t was an argument to
    # generate_correctness_evidence(), an unreviewed policy-injection surface
    # with no independent check that the value matched what the binary under
    # test actually used.
    threshold_t: float


def collect_seed_evidence(
    binary: Path, *, op_filter: str, target_tensor: str, candidate_stable_name: str,
    seed: int, env: dict[str, str] | None = None, runner=subprocess.run,
) -> SeedEvidence:
    """Run test-backend-ops twice for one seed (forced-native, forced-
    candidate) and reduce both runs to one SeedEvidence row, matched by
    tensor name.

    Fails closed: a missing digest/metric line, or a digest mismatch between
    the two runs, raises EvidenceError rather than silently producing a
    partial or unproven row (RV77 Q1's hard gate -- a forced-native and a
    forced-candidate invocation are only a valid joint measurement if both
    actually compared against the same CPU-reference input). A nonzero exit
    code alone does not raise -- it is recorded as execution_status='failed'
    so the caller (aggregate_seed_evidence) can fail closed on ANY failed
    seed rather than only ones that also happened to omit a digest line."""
    native_run = run_test_backend_ops(
        binary, op_filter=op_filter, seed=seed, dispatch_mode="native",
        forced_candidate=None, env=env, runner=runner,
    )
    candidate_run = run_test_backend_ops(
        binary, op_filter=op_filter, seed=seed, dispatch_mode="replay",
        forced_candidate=candidate_stable_name, env=env, runner=runner,
    )

    native_status = "ok" if native_run.returncode == 0 else "failed"
    candidate_status = "ok" if candidate_run.returncode == 0 else "failed"

    native_digest = find_digest_for_tensor(native_run.stderr, target_tensor)
    candidate_digest = find_digest_for_tensor(candidate_run.stderr, target_tensor)
    if native_digest is None or candidate_digest is None:
        raise EvidenceError(
            f"seed {seed}: missing BIGCHERRY_REF_DIGEST for tensor {target_tensor!r} "
            f"(native found={native_digest is not None}, "
            f"candidate found={candidate_digest is not None}) -- is the binary "
            f"built with patches/1222_hi67_deterministic_test_backend_ops_seed.py "
            f"and patches/1223_hi67_machine_readable_correctness_metrics.py applied?"
        )
    if native_digest.digest != candidate_digest.digest:
        raise EvidenceError(
            f"seed {seed}: native and candidate runs saw DIFFERENT CPU-reference "
            f"input for tensor {target_tensor!r} (native digest="
            f"{native_digest.digest}, candidate digest={candidate_digest.digest}) "
            f"-- the comparison is invalid, not just imprecise. Check that both "
            f"runs exercise the identical op filter in the identical operand "
            f"order (RV77 Q1's hard gate)."
        )

    native_metric = find_metric_for_tensor(native_run.stderr, target_tensor)
    candidate_metric = find_metric_for_tensor(candidate_run.stderr, target_tensor)
    if native_status == "ok" and native_metric is None:
        raise EvidenceError(
            f"seed {seed}: native run exited 0 but produced no "
            f"BIGCHERRY_CORRECTNESS_METRIC for tensor {target_tensor!r}"
        )
    if candidate_status == "ok" and candidate_metric is None:
        raise EvidenceError(
            f"seed {seed}: candidate run exited 0 but produced no "
            f"BIGCHERRY_CORRECTNESS_METRIC for tensor {target_tensor!r}"
        )
    if native_metric is not None and candidate_metric is not None \
            and native_metric.threshold != candidate_metric.threshold:
        raise EvidenceError(
            f"seed {seed}: native and candidate runs report DIFFERENT upstream "
            f"correctness thresholds for tensor {target_tensor!r} (native="
            f"{native_metric.threshold!r}, candidate={candidate_metric.threshold!r}) "
            f"-- they must be comparing the same operation under the same "
            f"upstream test-backend-ops tolerance table."
        )

    return SeedEvidence(
        seed=seed,
        reference_digest=native_digest.digest,
        e_n_nmse=native_metric.err if native_metric else float("nan"),
        e_c_nmse=candidate_metric.err if candidate_metric else float("nan"),
        max_abs_native=native_metric.max_abs if native_metric else float("nan"),
        max_abs_candidate=candidate_metric.max_abs if candidate_metric else float("nan"),
        native_execution_status=native_status,
        candidate_execution_status=candidate_status,
        threshold_t=native_metric.threshold if native_metric else float("nan"),
    )


@dataclass(frozen=True)
class EvidenceAggregate:
    seed_rows: tuple[SeedEvidence, ...]
    e_n_nmse: float
    e_c_nmse: float
    max_abs_native: float
    max_abs_candidate: float
    threshold_t: float
    headroom_fraction: float
    contract_version: str

    @property
    def native_passes(self) -> bool:
        return self.e_n_nmse <= self.threshold_t

    @property
    def candidate_passes_headroom(self) -> bool:
        return self.e_c_nmse <= self.e_n_nmse + self.headroom_fraction * (
            self.threshold_t - self.e_n_nmse
        )

    @property
    def candidate_max_abs_ok(self) -> bool:
        return self.max_abs_candidate <= self.max_abs_native

    @property
    def dispatchable(self) -> bool:
        """RV49's full acceptance rule: native itself must clear the
        threshold, the candidate must stay within its headroom budget of
        native's remaining NMSE room, and not be worse than native in
        max_abs."""
        return self.native_passes and self.candidate_passes_headroom and self.candidate_max_abs_ok


def aggregate_seed_evidence(
    seed_rows: list[SeedEvidence], *,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    contract_version: str = CONTRACT_VERSION,
) -> EvidenceAggregate:
    """Worst-of-seeds reduction (RV49: "use the WORST result across >=3
    deterministic seeds, not an average"). Fails closed on too few seeds or
    any seed that did not execute cleanly -- a failed seed is evidence of a
    problem, not something to average away.

    threshold_t is NOT a parameter here (HI67 threshold-authority fix): it is
    derived from the seed rows' own SeedEvidence.threshold_t, which in turn
    comes only from test-backend-ops' own emitted BIGCHERRY_CORRECTNESS_METRIC
    threshold=... field. Every seed row must agree on T -- disagreement means
    the seeds are not actually comparable evidence for the same operation."""
    if len(seed_rows) < MIN_SEEDS:
        raise EvidenceError(
            f"need >={MIN_SEEDS} deterministic seeds (RV49 contract), got {len(seed_rows)}"
        )
    failed = [
        row for row in seed_rows
        if row.native_execution_status != "ok" or row.candidate_execution_status != "ok"
    ]
    if failed:
        raise EvidenceError(
            "one or more seeds did not execute cleanly, refusing to aggregate: "
            + ", ".join(
                f"seed={row.seed} native={row.native_execution_status} "
                f"candidate={row.candidate_execution_status}"
                for row in failed
            )
        )
    thresholds = {row.threshold_t for row in seed_rows}
    if len(thresholds) != 1:
        raise EvidenceError(
            "seeds disagree on the upstream correctness threshold T -- RV49 "
            "requires a single authoritative T derived from test-backend-ops' "
            f"own emitted value for every seed of the same comparison, got: "
            + ", ".join(f"seed={row.seed} threshold_t={row.threshold_t!r}" for row in seed_rows)
        )
    threshold_t = next(iter(thresholds))
    return EvidenceAggregate(
        seed_rows=tuple(seed_rows),
        e_n_nmse=max(row.e_n_nmse for row in seed_rows),
        e_c_nmse=max(row.e_c_nmse for row in seed_rows),
        max_abs_native=max(row.max_abs_native for row in seed_rows),
        max_abs_candidate=max(row.max_abs_candidate for row in seed_rows),
        threshold_t=threshold_t,
        headroom_fraction=headroom_fraction,
        contract_version=contract_version,
    )


def generate_correctness_evidence(
    binary: Path, *, op_filter: str, target_tensor: str, candidate_stable_name: str,
    seeds: tuple[int, ...] = (1, 2, 3),
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    contract_version: str = CONTRACT_VERSION,
    env: dict[str, str] | None = None, runner=subprocess.run,
) -> EvidenceAggregate:
    """End-to-end: collect per-seed evidence for every seed, then aggregate.
    ``seeds`` should come from a versioned seed-set policy the caller keeps
    fixed per contract_version (RV77 Q2 change 6) -- picking seeds ad hoc per
    invocation would make repeated generation runs for the same identity a
    cherry-picking vector, which the schema's UNIQUE constraint on
    (build_id, hardware_id, signature_id, candidate_id, contract_version)
    additionally forecloses at the storage layer.

    No threshold_t parameter (HI67 threshold-authority fix): T is derived
    entirely from test-backend-ops' own emitted threshold, via
    aggregate_seed_evidence()."""
    seed_rows = [
        collect_seed_evidence(
            binary, op_filter=op_filter, target_tensor=target_tensor,
            candidate_stable_name=candidate_stable_name, seed=seed, env=env, runner=runner,
        )
        for seed in seeds
    ]
    return aggregate_seed_evidence(
        seed_rows, headroom_fraction=headroom_fraction, contract_version=contract_version,
    )


def write_correctness_evidence(
    connection: sqlite3.Connection, *, build_id: int, hardware_id: int, signature_id: int,
    candidate_id: int, native_candidate_id: int, aggregate: EvidenceAggregate,
    tool_version: str,
) -> int:
    """Insert one correctness_evidence row plus its correctness_evidence_seed
    children. Raises sqlite3.IntegrityError (uncaught -- a caller bug, not
    something to silently overwrite) if this exact identity+contract_version
    already has evidence, per the schema's UNIQUE constraint."""
    cursor = connection.execute(
        "INSERT INTO correctness_evidence "
        "(build_id, hardware_id, signature_id, candidate_id, native_candidate_id, "
        "contract_version, threshold_t, headroom_fraction, e_n_nmse, e_c_nmse, "
        "max_abs_native, max_abs_candidate, seed_count, tool_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            build_id, hardware_id, signature_id, candidate_id, native_candidate_id,
            aggregate.contract_version, aggregate.threshold_t, aggregate.headroom_fraction,
            aggregate.e_n_nmse, aggregate.e_c_nmse, aggregate.max_abs_native,
            aggregate.max_abs_candidate, len(aggregate.seed_rows), tool_version,
        ),
    )
    evidence_id = cursor.lastrowid
    for row in aggregate.seed_rows:
        connection.execute(
            "INSERT INTO correctness_evidence_seed "
            "(correctness_evidence_id, seed, reference_digest, e_n_nmse, e_c_nmse, "
            "max_abs_native, max_abs_candidate, native_execution_status, "
            "candidate_execution_status, threshold_t) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                evidence_id, row.seed, row.reference_digest, row.e_n_nmse, row.e_c_nmse,
                row.max_abs_native, row.max_abs_candidate, row.native_execution_status,
                row.candidate_execution_status, row.threshold_t,
            ),
        )
    connection.commit()
    return evidence_id
