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

Requires patches/1222_hi67_deterministic_test_backend_ops_seed/patch.py and
patches/1223_hi67_machine_readable_correctness_metrics/patch.py to be applied to
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

_REGISTRY_MISMATCH_RE = re.compile(
    r"GGML_HIP_FORCE_CANDIDATE=(?P<candidate>\S+) not found in registry"
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
    binary: Path, *, op_filter: str | None = None, test_file: Path | None = None,
    moe_glu_file: Path | None = None,
    seed: int, dispatch_mode: str,
    forced_candidate: str | None, env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> subprocess.CompletedProcess:
    """``op_filter`` selects a case from test-backend-ops' own fixed synthetic
    corpus via ``-p`` -- it only produces evidence when a real dispatch
    signature happens to coincide with one of those hardcoded shapes, which
    real production shapes generally do not (HI80, 2026-08-23 real-hardware
    finding: gpt-oss-20B's real m=32/n=21/k=2880 MUL_MAT matches none of
    them). ``test_file`` instead points at a test-backend-ops
    ``--test-file`` line (see signature_correctness_mapping.
    signature_to_test_file_line) describing the EXACT signature via
    test_generic_op, bypassing the fixed corpus entirely -- this is the
    real fix for that gap, not a new patch to test-backend-ops itself.
    ``moe_glu_file`` (HI119) is a third, real alternative for a fused
    MUL_MAT_ID(gate)+MUL_MAT_ID(up)+GLU dispatch -- test_generic_op can only
    build ONE op per line and cannot represent this fused compound at all
    (HI108's own investigation); points at a ``--moe-glu-file`` line (see
    signature_mapping.signature_to_moe_glu_file_line) instead, which drives
    the registered test_bigcherry_moe_glu_fusion class (patches 1239/1240).
    Exactly one of the three must be given."""
    given = [x is not None for x in (op_filter, test_file, moe_glu_file)]
    if sum(given) != 1:
        raise EvidenceError(
            "run_test_backend_ops requires exactly one of op_filter, test_file "
            "or moe_glu_file"
        )
    if seed == 0:
        raise EvidenceError(
            "seed must be nonzero -- 0 leaves BIGCHERRY_TEST_DETERMINISTIC_SEED "
            "unset in the patched test-backend-ops, disabling deterministic mode "
            "entirely (see patches/1222_hi67_deterministic_test_backend_ops_seed/patch.py)"
        )
    run_env = dict(env or {})
    run_env["BIGCHERRY_TEST_DETERMINISTIC_SEED"] = str(seed)
    run_env["GGML_HIP_DISPATCH_MODE"] = dispatch_mode
    if forced_candidate is not None:
        # HI105: GGML_HIP_FORCE_CANDIDATE_STRICT must accompany FORCE_CANDIDATE
        # here -- dispatch.cu's own comment names this exact file as the
        # reason STRICT exists: without it, an ineligible/unregistered
        # requested candidate silently falls back to ordinary resolution
        # instead of aborting, and this producer would then record a
        # plausible-looking "candidate" correctness result for an operation
        # that never actually ran under the named candidate at all.
        run_env["GGML_HIP_FORCE_CANDIDATE"] = forced_candidate
        run_env["GGML_HIP_FORCE_CANDIDATE_STRICT"] = "1"
    else:
        run_env.pop("GGML_HIP_FORCE_CANDIDATE", None)
        run_env.pop("GGML_HIP_FORCE_CANDIDATE_STRICT", None)
    if moe_glu_file is not None:
        argv = [str(binary), "test", "--moe-glu-file", str(moe_glu_file)]
    elif test_file is not None:
        argv = [str(binary), "test", "--test-file", str(test_file)]
    else:
        argv = [str(binary), "test", "-o", "MUL_MAT", "-p", op_filter]
    return runner(argv, capture_output=True, text=True, env=run_env)


def find_digest_for_tensor(stderr_text: str, tensor_name: str) -> RefDigest | None:
    matches = [d for d in parse_ref_digests(stderr_text) if d.name == tensor_name]
    return matches[0] if matches else None


def find_metric_for_tensor(stderr_text: str, tensor_name: str) -> CorrectnessMetric | None:
    matches = [m for m in parse_correctness_metrics(stderr_text) if m.tensor == tensor_name]
    return matches[0] if matches else None


@dataclass(frozen=True)
class NativeSeedEvidence:
    """HTR01 (2026-08-30, adversarially designed with GPT, session
    ses_330ae3c055084f38): the forced-native half of one seed's evidence,
    isolated so it can be REUSED across multiple candidates for the same
    (binary, signature, seed) -- recovery.py's whole cost-amortization
    story ("first alternative for a signature pays for native + candidate;
    every later alternative for the SAME signature pays candidate only")
    depends on this being a real, separate, cacheable unit rather than
    baked into one seed-evidence-per-candidate call."""
    seed: int
    reference_digest: str
    e_n_nmse: float
    max_abs_native: float
    threshold_t: float
    native_execution_status: str
    native_output_digest: str | None
    reference_output_digest: str | None
    output_nels: int | None


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
    # schema 9 / HTR01 (2026-08-30): exact backend OUTPUT byte digests
    # (HI83's backend1_digest/backend2_digest, already emitted by the
    # patched producer but previously discarded here) -- enables exact
    # numerical-family clustering ("these candidates produce IDENTICAL
    # bytes"), strictly stronger than matching NMSE alone. None when the
    # producer predates the HI83 digest extension.
    native_output_digest: str | None = None
    candidate_output_digest: str | None = None
    reference_output_digest: str | None = None
    output_nels: int | None = None


def collect_native_seed_evidence(
    binary: Path, *, op_filter: str | None = None, test_file: Path | None = None,
    moe_glu_file: Path | None = None,
    target_tensor: str, digest_tensor: str | None = None,
    seed: int, env: dict[str, str] | None = None, runner=subprocess.run,
) -> NativeSeedEvidence:
    """The forced-native half of collect_seed_evidence's original two-run
    logic, isolated for reuse. See collect_seed_evidence's docstring for
    the digest_tensor/target_tensor distinction, unchanged here."""
    digest_tensor = digest_tensor if digest_tensor is not None else target_tensor
    native_run = run_test_backend_ops(
        binary, op_filter=op_filter, test_file=test_file, moe_glu_file=moe_glu_file,
        seed=seed, dispatch_mode="native",
        forced_candidate=None, env=env, runner=runner,
    )
    native_status = "ok" if native_run.returncode == 0 else "failed"
    native_digest = find_digest_for_tensor(native_run.stderr, digest_tensor)
    if native_digest is None:
        raise EvidenceError(
            f"seed {seed}: missing BIGCHERRY_REF_DIGEST for tensor {digest_tensor!r} "
            f"(native run) -- is the binary built with patches/1222_hi67_"
            f"deterministic_test_backend_ops_seed/patch.py and patches/1223_hi67_"
            f"machine_readable_correctness_metrics/patch.py applied?"
        )
    native_metric = find_metric_for_tensor(native_run.stderr, target_tensor)
    if native_status == "ok" and native_metric is None:
        raise EvidenceError(
            f"seed {seed}: native run exited 0 but produced no "
            f"BIGCHERRY_CORRECTNESS_METRIC for tensor {target_tensor!r}"
        )
    return NativeSeedEvidence(
        seed=seed,
        reference_digest=native_digest.digest,
        e_n_nmse=native_metric.err if native_metric else float("nan"),
        max_abs_native=native_metric.max_abs if native_metric else float("nan"),
        threshold_t=native_metric.threshold if native_metric else float("nan"),
        native_execution_status=native_status,
        native_output_digest=native_metric.backend1_digest if native_metric else None,
        reference_output_digest=native_metric.backend2_digest if native_metric else None,
        output_nels=native_metric.n if native_metric else None,
    )


def collect_candidate_seed_evidence(
    binary: Path, *, op_filter: str | None = None, test_file: Path | None = None,
    moe_glu_file: Path | None = None,
    target_tensor: str, digest_tensor: str | None = None, candidate_stable_name: str,
    seed: int, native: NativeSeedEvidence, env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> SeedEvidence:
    """The forced-candidate half of collect_seed_evidence's original logic,
    reusing an ALREADY-COLLECTED ``native`` (HTR01: this is what lets
    recovery.py qualify a second, third, ... alternative for the SAME
    signature without re-running the native leg every time). Fails closed
    exactly as collect_seed_evidence always did: a missing digest/metric,
    a digest mismatch against the given native, or a threshold mismatch
    raises EvidenceError rather than silently producing a partial row."""
    digest_tensor = digest_tensor if digest_tensor is not None else target_tensor
    candidate_run = run_test_backend_ops(
        binary, op_filter=op_filter, test_file=test_file, moe_glu_file=moe_glu_file,
        seed=seed, dispatch_mode="replay",
        forced_candidate=candidate_stable_name, env=env, runner=runner,
    )
    candidate_status = "ok" if candidate_run.returncode == 0 else "failed"

    # HI106: GGML_HIP_FORCE_CANDIDATE_STRICT aborts (SIGABRT) when the named
    # candidate isn't in the binary's compiled registry -- e.g. the binary
    # was built from a different --inventory than the tune run being
    # evidenced. That failure looks identical to a real numerical
    # correctness bug once folded into a generic "failed" execution_status,
    # and was mistaken for one until traced back to this exact abort message.
    # Fail closed here with an unambiguous diagnosis instead of letting it
    # surface later as an opaque aggregate "did not execute cleanly" error.
    registry_mismatch = _REGISTRY_MISMATCH_RE.search(candidate_run.stderr)
    if registry_mismatch is not None:
        raise EvidenceError(
            f"seed {seed}: candidate {registry_mismatch['candidate']!r} is not in "
            f"the binary's compiled candidate registry -- this binary was built "
            f"from a different --inventory than the tune run being evidenced. "
            f"Rebuild test-backend-ops scoped to the SAME inventory as the "
            f"measurements being evidenced (this is not a correctness failure)."
        )

    candidate_digest = find_digest_for_tensor(candidate_run.stderr, digest_tensor)
    if candidate_digest is None:
        raise EvidenceError(
            f"seed {seed}: missing BIGCHERRY_REF_DIGEST for tensor {digest_tensor!r} "
            f"(candidate run) -- is the binary built with patches/1222_hi67_"
            f"deterministic_test_backend_ops_seed/patch.py and patches/1223_hi67_"
            f"machine_readable_correctness_metrics/patch.py applied?"
        )
    if native.reference_digest != candidate_digest.digest:
        raise EvidenceError(
            f"seed {seed}: native and candidate runs saw DIFFERENT CPU-reference "
            f"input for tensor {digest_tensor!r} (native digest="
            f"{native.reference_digest}, candidate digest={candidate_digest.digest}) "
            f"-- the comparison is invalid, not just imprecise. Check that both "
            f"runs exercise the identical op filter in the identical operand "
            f"order (RV77 Q1's hard gate)."
        )

    candidate_metric = find_metric_for_tensor(candidate_run.stderr, target_tensor)
    if candidate_status == "ok" and candidate_metric is None:
        raise EvidenceError(
            f"seed {seed}: candidate run exited 0 but produced no "
            f"BIGCHERRY_CORRECTNESS_METRIC for tensor {target_tensor!r}"
        )
    if candidate_metric is not None and native.threshold_t != candidate_metric.threshold:
        raise EvidenceError(
            f"seed {seed}: native and candidate runs report DIFFERENT upstream "
            f"correctness thresholds for tensor {target_tensor!r} (native="
            f"{native.threshold_t!r}, candidate={candidate_metric.threshold!r}) "
            f"-- they must be comparing the same operation under the same "
            f"upstream test-backend-ops tolerance table."
        )

    return SeedEvidence(
        seed=seed,
        reference_digest=native.reference_digest,
        e_n_nmse=native.e_n_nmse,
        e_c_nmse=candidate_metric.err if candidate_metric else float("nan"),
        max_abs_native=native.max_abs_native,
        max_abs_candidate=candidate_metric.max_abs if candidate_metric else float("nan"),
        native_execution_status=native.native_execution_status,
        candidate_execution_status=candidate_status,
        threshold_t=native.threshold_t,
        native_output_digest=native.native_output_digest,
        candidate_output_digest=candidate_metric.backend1_digest if candidate_metric else None,
        reference_output_digest=native.reference_output_digest,
        output_nels=native.output_nels,
    )


def collect_seed_evidence(
    binary: Path, *, op_filter: str | None = None, test_file: Path | None = None,
    moe_glu_file: Path | None = None,
    target_tensor: str, digest_tensor: str | None = None, candidate_stable_name: str,
    seed: int, env: dict[str, str] | None = None, runner=subprocess.run,
    native: NativeSeedEvidence | None = None,
) -> SeedEvidence:
    """Run test-backend-ops for one seed (forced-native unless ``native`` is
    already given, plus forced-candidate) and reduce to one SeedEvidence
    row. Thin wrapper over collect_native_seed_evidence +
    collect_candidate_seed_evidence preserving this function's original
    two-run-every-call behavior for existing callers; pass ``native``
    explicitly (HTR01's recovery.py does) to skip re-running the native
    leg when it was already collected for this exact (binary, signature,
    seed).

    See collect_native_seed_evidence/collect_candidate_seed_evidence for
    the ``digest_tensor``/fail-closed contract details, unchanged from
    this function's original single-pass implementation."""
    if native is None:
        native = collect_native_seed_evidence(
            binary, op_filter=op_filter, test_file=test_file, moe_glu_file=moe_glu_file,
            target_tensor=target_tensor, digest_tensor=digest_tensor,
            seed=seed, env=env, runner=runner,
        )
    return collect_candidate_seed_evidence(
        binary, op_filter=op_filter, test_file=test_file, moe_glu_file=moe_glu_file,
        target_tensor=target_tensor, digest_tensor=digest_tensor,
        candidate_stable_name=candidate_stable_name, seed=seed, native=native,
        env=env, runner=runner,
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
    binary: Path, *, op_filter: str | None = None, test_file: Path | None = None,
    moe_glu_file: Path | None = None,
    target_tensor: str, digest_tensor: str | None = None, candidate_stable_name: str,
    seeds: tuple[int, ...] = (1, 2, 3),
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    contract_version: str = CONTRACT_VERSION,
    env: dict[str, str] | None = None, runner=subprocess.run,
    native_seeds: dict[int, NativeSeedEvidence] | None = None,
) -> EvidenceAggregate:
    """End-to-end: collect per-seed evidence for every seed, then aggregate.
    ``seeds`` should come from a versioned seed-set policy the caller keeps
    fixed per contract_version (RV77 Q2 change 6) -- picking seeds ad hoc per
    invocation would make repeated generation runs for the same identity a
    cherry-picking vector, which the schema's UNIQUE constraint on
    (build_id, hardware_id, signature_id, candidate_id, contract_version)
    additionally forecloses at the storage layer.

    ``native_seeds`` (HTR01, 2026-08-30): an optional ``{seed:
    NativeSeedEvidence}`` map of ALREADY-COLLECTED native baselines for this
    exact (binary, signature) -- a caller evidencing a SECOND, THIRD, ...
    alternative candidate for the same signature (recovery.py's lazy
    qualification path) passes the map it built while evidencing the
    first, so this call only re-runs the candidate leg per seed, not
    native+candidate. A seed missing from the map still gets its native
    leg collected fresh.

    No threshold_t parameter (HI67 threshold-authority fix): T is derived
    entirely from test-backend-ops' own emitted threshold, via
    aggregate_seed_evidence()."""
    native_seeds = native_seeds or {}
    seed_rows = [
        collect_seed_evidence(
            binary, op_filter=op_filter, test_file=test_file, moe_glu_file=moe_glu_file,
            target_tensor=target_tensor,
            digest_tensor=digest_tensor,
            candidate_stable_name=candidate_stable_name, seed=seed, env=env, runner=runner,
            native=native_seeds.get(seed),
        )
        for seed in seeds
    ]
    return aggregate_seed_evidence(
        seed_rows, headroom_fraction=headroom_fraction, contract_version=contract_version,
    )


@dataclass(frozen=True)
class EvidenceOrigin:
    """HTR01 / schema 9 (2026-08-30): WHY a correctness_evidence row exists
    -- see correctness_evidence_origin's own schema comment. ``reason`` must
    be one of the CHECK constraint's known values; an unrecognized reason
    fails at INSERT time (sqlite3.IntegrityError), not silently."""
    reason: str  # "promotion_winner" | "recovery_alternative" | "manual_analysis"
    campaign_run_id: str | None = None
    recovery_run_id: str | None = None


def write_correctness_evidence(
    connection: sqlite3.Connection, *, build_id: int, hardware_id: int, signature_id: int,
    candidate_id: int, native_candidate_id: int, aggregate: EvidenceAggregate,
    tool_version: str, origin: EvidenceOrigin | None = None,
) -> int:
    """Insert one correctness_evidence row plus its correctness_evidence_seed
    children (and, if ``origin`` is given, one correctness_evidence_origin
    row -- schema 9/HTR01; omitted entirely, not NULL-reasoned, when the
    caller doesn't supply one, matching the migration's "no row means
    predates origin tracking" contract). Raises sqlite3.IntegrityError
    (uncaught -- a caller bug, not something to silently overwrite) if this
    exact identity+contract_version already has evidence, per the schema's
    UNIQUE constraint."""
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
            "candidate_execution_status, threshold_t, native_output_digest, "
            "candidate_output_digest, reference_output_digest, output_nels) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                evidence_id, row.seed, row.reference_digest, row.e_n_nmse, row.e_c_nmse,
                row.max_abs_native, row.max_abs_candidate, row.native_execution_status,
                row.candidate_execution_status, row.threshold_t, row.native_output_digest,
                row.candidate_output_digest, row.reference_output_digest, row.output_nels,
            ),
        )
    if origin is not None:
        connection.execute(
            "INSERT INTO correctness_evidence_origin "
            "(correctness_evidence_id, reason, campaign_run_id, recovery_run_id) "
            "VALUES (?, ?, ?, ?)",
            (evidence_id, origin.reason, origin.campaign_run_id, origin.recovery_run_id),
        )
    connection.commit()
    return evidence_id
