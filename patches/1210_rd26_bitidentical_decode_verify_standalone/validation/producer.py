"""PA36 migration #4 (dev-gpt-agent req_110d0729beb44d8b): patch-local
validation producer for 1210 (RD26-DECODE-VERIFY-BIT-IDENTITY).

Mechanically migrated off validation_campaign.py's
run_rd26_decode_verify_bit_identity_check() -- the real within-build
decode-vs-speculative-verify raw-F32-logit byte-identity oracle that
drives ``llama-results`` (a registered llama.cpp example tool) through
``llama_decode()`` and writes the raw ``llama_get_logits_ith()`` rows
to GGUF. The dead legacy run_rd26_ppl_check() PPL wrapper and this
patch's rd26_correctness.py module (its only home) are deleted too --
zero live call sites. Both functions and the --run-rd26-contract CLI
path are DELETED from shared code in the same change (no compatibility
layer, per the project's migrate-up doctrine).

The experiment is deliberately a within-build comparison first:
  control: ubatch=1 vs ubatch=n_draft+1
  subject: ubatch=1 vs ubatch=n_draft+1
The contract passes only when the subject pair is byte-identical and
the control pair is not -- non-vacuous, proving both the invariant
RD26 claims and that THIS patch (not an already-identical baseline)
removed the divergence.

Design rulings applied (req_110d0729beb44d8b):

- ONE named contract-correctness result: ``bit_identical`` only (the
  RD26 contract requires exactly that one check).
- The isolated llama-results pair is built ONCE as the fat multi-arch
  set via ``ctx.runtime.build_pair()`` (the PA36 build-once authority,
  primary_target="llama-results") and run per-device;
  require_parity=True demands assert_validation_subject_parity().
- Normal focal subject composition: build_pair()'s subject = baseline
  + focal (1210). 1210's patch.toml has requires=[] and the patch is
  ONLY the two base-standalone hunks, so focal does not expand to the
  deferred five-commit determinism cluster (no common_extra_patches).
- trace_probe='skip': RD26 declares no activation check (its expected
  effect is determinism/correctness only); the producer returns no
  activation/trace evidence.
- performance_evidence stays None: the contract's controls check
  remains unsatisfied exactly as the historical record shows it.
  THIS MIGRATION PROVES PRODUCER EQUIVALENCE, NOT RD26 QUALIFICATION --
  a real run against the current 2-of-5 base-standalone subset may
  legitimately FAIL, and that honest FAIL is the expected receipt.
  Never weaken this check to force a pass.

Every canonical identity field (patch digest, source trees, campaign
identity, root correctness.json) is owned by the shared binder, never
here.
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from bigcherry.patch import validation_producer as vp

# One contract architecture per run (the historical RD26 rule): the
# operator names it via --amdgpu-targets; the binary itself is built
# ONCE as the production-matching fat multi-arch set and the real
# device is selected at run time.
_CONTRACT_ARCHITECTURES: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030")

_CONTRACT_ID = "RD26-DECODE-VERIFY-BIT-IDENTITY"
_SUBJECT_PATCH = "1210_rd26_bitidentical_decode_verify_standalone"

# Keep verify width n_draft+1 inside RD26's <=8 scope (n_max in [1,7]).
_SPEC_DRAFT_N_MAX = 4
_VERIFY_WIDTH = _SPEC_DRAFT_N_MAX + 1
_CTX_SIZE = 256
_REPLICATES = 2
_TIMEOUT_S = 900
_PROMPT = (
    "RD26 determinism probe. The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump. Bright vixens jump; dozy fowl quack."
)

_ARTIFACT_NAME = "rd26-decode-verify-bit-identity.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _RunRecord:
    path: Path
    size: int
    sha256: str


def _first_diff_offset(left: Path, right: Path) -> int | None:
    offset = 0
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            lchunk = lhs.read(1024 * 1024)
            rchunk = rhs.read(1024 * 1024)
            if lchunk == rchunk:
                if not lchunk:
                    return None
                offset += len(lchunk)
                continue
            limit = min(len(lchunk), len(rchunk))
            for index in range(limit):
                if lchunk[index] != rchunk[index]:
                    return offset + index
            return offset + limit


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.patch import source as psi

    if (
        len(ctx.fat_targets.targets) != 1
        or ctx.fat_targets.targets[0] not in _CONTRACT_ARCHITECTURES
    ):
        raise vp.ValidationProducerError(
            f"rd26 bit identity: {_CONTRACT_ID} requires exactly one "
            "contract architecture per run (gfx1100, gfx1201, or gfx1030); "
            f"got targets={ctx.fat_targets.targets!r}"
        )
    architecture = ctx.fat_targets.targets[0]

    # The raw-logit oracle is a whole-MODEL run; it needs a real model
    # file and no PPL corpus. Fail closed before any build or
    # subprocess.
    if ctx.model is None:
        raise vp.ValidationProducerError(
            "rd26 bit identity requires a real whole model (--model); "
            "refusing to run without one"
        )
    model = ctx.model

    if not 1 <= _SPEC_DRAFT_N_MAX <= 7:
        raise vp.ValidationProducerError(
            "rd26 bit identity: spec_draft_n_max must be in [1, 7] "
            "so verify width n_draft+1 stays inside RD26's <=8 scope"
        )
    if _CTX_SIZE < _VERIFY_WIDTH * 2:
        raise vp.ValidationProducerError(
            "rd26 bit identity: ctx_size must be at least "
            "2 * (spec_draft_n_max + 1)"
        )
    if not _PROMPT.strip():
        raise vp.ValidationProducerError(
            "rd26 bit identity: prompt must be non-empty"
        )
    if _REPLICATES < 2:
        raise vp.ValidationProducerError(
            "rd26 bit identity: replicates must be >= 2 to prove "
            "same-configuration repeatability"
        )

    # The one sanctioned pair authority: control = baseline composition;
    # subject = baseline + focal (1210). 1210 is ONLY the two
    # base-standalone hunks (requires=[]), so focal does not expand to
    # the deferred five-commit cluster. Built ONCE as the fat multi-arch
    # set, parity asserted, run per-device.
    pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="llama-results",
        baseline_source="bigcherry",
        require_parity=True,
    )
    control_binary = pair.control_bin
    subject_binary = pair.subject_bin

    devices = ctx.runtime.device_contexts(device_map=ctx.device_map)
    matches = [d for d in devices if d.architecture == architecture]
    if len(matches) != 1:
        raise vp.ValidationProducerError(
            f"rd26 bit identity: expected exactly one device mapped for "
            f"run architecture {architecture!r}, got {len(matches)}; "
            f"device_map={ {k: tuple(v) for k, v in ctx.device_map.items()}!r} "
            "-- --device-map must select one real device for it"
        )
    device = matches[0]

    # Sanctioned HIP-only device selector env (the RD04 producer's
    # merge order): build env -> device overrides -> env_unset popped
    # LAST, so the sanctioned selector always wins.
    runtime_env = dict(ctx.build_env)
    runtime_env.update(dict(device.env_overrides))
    for key in device.env_unset:
        runtime_env.pop(key, None)

    scratch_dir = ctx.workdir / "scratch" / "rd26-bit-identity"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    created_outputs: list[Path] = []

    def _run_one(
        *, arm: str, binary: Path, mode: str, ubatch_size: int, replicate: int,
    ) -> _RunRecord:
        output = scratch_dir / f"{arm}-{mode}-rep{replicate}.gguf"
        created_outputs.append(output)

        argv = [
            str(binary), "--model", str(model), "--output", str(output),
            "--prompt", _PROMPT, "--ctx-size", str(_CTX_SIZE),
            "--batch-size", str(_CTX_SIZE), "--ubatch-size", str(ubatch_size),
            "-ngl", "99",
        ]
        completed = subprocess.run(
            argv, cwd=ctx.workdir, env=runtime_env, capture_output=True, text=True,
            timeout=_TIMEOUT_S, check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "<no output>"
            raise vp.ValidationProducerError(
                f"rd26 bit identity: {arm}/{mode}/rep{replicate} llama-results failed "
                f"with exit {completed.returncode}: {detail}"
            )
        if not output.is_file():
            raise vp.ValidationProducerError(
                f"rd26 bit identity: {arm}/{mode}/rep{replicate} llama-results did not "
                "create its GGUF output"
            )

        size = output.stat().st_size
        with output.open("rb") as handle:
            magic = handle.read(4)
        if size <= 4 or magic != b"GGUF":
            raise vp.ValidationProducerError(
                f"rd26 bit identity: {arm}/{mode}/rep{replicate} produced a malformed "
                "llama-results artifact"
            )

        return _RunRecord(path=output, size=size, sha256=_sha256_file(output))

    try:
        runs: dict[str, dict[str, list[_RunRecord]]] = {
            "control": {"decode": [], "verify": []},
            "subject": {"decode": [], "verify": []},
        }
        binaries = {"control": control_binary, "subject": subject_binary}
        widths = {"decode": 1, "verify": _VERIFY_WIDTH}

        # Alternate ordering between replicate pairs. Exact equality
        # should not depend on thermal state, but this avoids
        # systematically attaching any process-order effect to one
        # configuration.
        for arm in ("control", "subject"):
            for replicate in range(_REPLICATES):
                mode_order = ("decode", "verify") if replicate % 2 == 0 else ("verify", "decode")
                for mode in mode_order:
                    runs[arm][mode].append(_run_one(
                        arm=arm, binary=binaries[arm], mode=mode,
                        ubatch_size=widths[mode], replicate=replicate,
                    ))

        # Cross-configuration divergence is attributable only if each
        # individual configuration repeats exactly by itself.
        for arm in ("control", "subject"):
            for mode in ("decode", "verify"):
                records = runs[arm][mode]
                sizes = {record.size for record in records}
                digests = {record.sha256 for record in records}
                if len(sizes) != 1 or len(digests) != 1:
                    raise vp.ValidationProducerError(
                        f"rd26 bit identity: {arm}/{mode} is not repeatable across "
                        f"{_REPLICATES} identical process runs"
                    )

        arm_comparison: dict[str, dict[str, object]] = {}
        for arm in ("control", "subject"):
            decode = runs[arm]["decode"][0]
            verify = runs[arm]["verify"][0]

            if decode.size != verify.size:
                raise vp.ValidationProducerError(
                    f"rd26 bit identity: {arm} decode/verify llama-results artifacts have "
                    f"different sizes ({decode.size} != {verify.size}); model/prompt "
                    "output shape changed, so byte comparison is not a valid raw-logit "
                    "identity test"
                )

            identical = decode.sha256 == verify.sha256
            first_diff = None if identical else _first_diff_offset(decode.path, verify.path)

            arm_comparison[arm] = {
                "bit_identical": identical,
                "decode_sha256": decode.sha256, "verify_sha256": verify.sha256,
                "decode_repeat_sha256": [record.sha256 for record in runs[arm]["decode"]],
                "verify_repeat_sha256": [record.sha256 for record in runs[arm]["verify"]],
                "artifact_size": decode.size, "first_file_byte_mismatch": first_diff,
            }

        subject_identical = bool(arm_comparison["subject"]["bit_identical"])
        control_diverged = not bool(arm_comparison["control"]["bit_identical"])

        # Fail closed if the subject happens to be identical but the
        # control is also identical. That would establish the invariant
        # for this sample, but would not establish RD26's fixing effect.
        passed = subject_identical and control_diverged

        if not subject_identical:
            detail = (
                "subject decode/verify raw-logit artifacts differ; first_file_byte_mismatch="
                f"{arm_comparison['subject']['first_file_byte_mismatch']}"
            )
        elif not control_diverged:
            detail = (
                "subject decode/verify raw logits are bit-identical, but control is also "
                "bit-identical; RD26 fixing effect was not triggered and the result is "
                "non-authoritative"
            )
        else:
            detail = (
                f"subject decode (n_q=1) and verify-shaped (n_q={_VERIFY_WIDTH}) raw F32 "
                "logits are bit-identical; control diverges, proving a non-vacuous RD26 "
                "fixing effect"
            )

        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=passed, detail=detail,
        )

        comparison = {
            "method": "llama-results-raw-f32-gguf-cross-ubatch-byte-identity",
            "oracle": "llama_get_logits_ith raw F32 rows",
            "decode_ubatch": 1, "verify_ubatch": _VERIFY_WIDTH,
            "spec_draft_n_max": _SPEC_DRAFT_N_MAX, "replicates": _REPLICATES,
            "subject_bit_identical": subject_identical, "control_diverged": control_diverged,
            "arms": arm_comparison,
        }
        doc = {
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "check": bit_identical_result.check, "passed": bit_identical_result.passed,
            "detail": bit_identical_result.detail,
            "base_revision": ctx.base_revision, "architecture": architecture,
            "model": str(model), "prompt": _PROMPT,
            "prompt_sha256": hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest(),
            "ctx_size": _CTX_SIZE, "subject_patch": _SUBJECT_PATCH,
            "comparison": comparison,
            "control_source_tree": psi.git_worktree_tree(pair.control_source),
            "subject_source_tree": psi.git_worktree_tree(pair.subject_source),
            "control_build_identity": pair.validation_build_identities["control"],
            "subject_build_identity": pair.validation_build_identities["subject"],
        }
        ctx.runtime.write_artifact(name=_ARTIFACT_NAME, payload=doc)

        return vp.ProducerResult(
            correctness={
                "disposition": "passed" if passed else "failed",
                "mechanism": "rd26-decode-verify-bit-identity",
                "detail": detail,
            },
            validation_build_identities=pair.validation_build_identities,
            activation_evidence=None,
            performance_evidence=None,
            trace_evidence=None,
            check_results=(),
            lane_effects=(),
            contract_correctness_results=(bit_identical_result,),
            emitted_artifacts=frozenset({_ARTIFACT_NAME}),
        )
    finally:
        for path in created_outputs:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()
            ctx.workdir.joinpath("scratch").rmdir()
