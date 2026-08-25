"""HI78: fast single-model end-to-end smoke campaign.

record -> inventory -> tune -> dispatch-db -> promote -> [correctness-evidence
-> re-promote] -> export -> replay. The bracketed stage (S3b) only runs when
--correctness-binary is given; see HI80.

Exercises the whole HI29/HI30/HI31/HI67/HI74 chain against one small local
GGUF model in minutes, not the hours a full HI35/HI36-style production
campaign takes. Built from a scratchpad driver script that found 5 real bugs
(1 production-breaking compile bug, 4 correctness/usability bugs) in under an
hour on its first real run -- see ledger event
chg_20260821_121835_fixed-a-production-breaking-co_3880. Intended to run
before landing any change to the tuner/promotion/replay chain, as a much
cheaper alternative to a full production campaign for catching integration
bugs between the C++ emitter and the Python promotion/replay tooling.

Requires two pre-built llama-server binaries (this script does not build
them -- see docs/reference/BUILD.md for cmake invocations):
  --tune-server    built with GGML_HIP_AUTOTUNE=ON, GGML_HIP_AUTOTUNE_RECORD=ON
  --replay-server  built with GGML_HIP_DISPATCH_REPLAY=ON (GGML_HIP_AUTOTUNE off
                    -- these two options are cmake-mutually-exclusive, so this
                    is necessarily a second build tree, not a runtime switch)

Usage:
    python -m bigcherry.e2e_smoke_campaign \\
        --model G:/models/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf \\
        --tune-server C:/bcw/bin/llama-server.exe \\
        --replay-server C:/bcw-replay/bin/llama-server.exe \\
        --manifest H:/.../artifacts/<rev>/hip-autotune-manifest.json \\
        --workdir C:/scratch/e2e-smoke

Resumable: each stage's output is checked before rerunning it, so a fixed
bug can be validated by rerunning just the stages downstream of the fix
(delete that stage's output file(s) to force a rerun).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROGRESS_INTERVAL_S = 15

# HI82 item 9: campaign-identity-gated resume (design/implementation:
# gpt-auto-agent, req_527bff46e32e481c).
STATUS_SCHEMA_VERSION = 2
CAMPAIGN_IDENTITY_SCHEMA_VERSION = 2

# Manual semantic version for campaign execution logic. Bump this whenever a
# code change can alter stage meaning/output without already changing one of
# the explicit identity inputs below (model/build/patch/source identity).
CAMPAIGN_SCRIPT_VERSION = "hi82-item8-item9-v2"
COMPLETIONS = (
    {
        "prompt": "Explain the difference between a mutex and a semaphore in "
                  "three sentences.",
        "n_predict": 128,
        "temperature": 0,
    },
    {
        "prompt": "Write a short poem about the ocean, at least twenty lines "
                  "long, covering waves, tides, storms, and calm mornings.",
        "n_predict": 256,
        "temperature": 0,
    },
)


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class CampaignIdentityContext:
    """External provenance supplied by a build/source orchestrator.

    patch_validation_campaign.py supplies this richer identity (patch
    digest, patched source tree, per-build-mode compile-verified evidence
    from builds.capture_completed_build_evidence()). Direct
    e2e_smoke_campaign CLI use may omit it (identity_context=None on
    Campaign) -- executable file identities still protect resume in that
    case, there is just naturally no patch/source provenance to record.
    """

    patch_name: str
    patch_digest: str
    patched_source_tree: str
    gpu_architecture: str
    build_identities: dict[str, dict[str, object]]


# The exact keys builds.CompletedBuildEvidence.campaign_identity() returns.
# HI82 (GPT review, req_6b1466ee8369406c): the campaign only validated ROLE
# presence in build_identities, not the CONTENT of each role's identity --
# {"tune": {}, "replay": {}, "stock": {}} would structurally satisfy the
# type while carrying none of the real compile/runtime proof. Fail closed
# on the boundary instead, so a future caller can't silently regress to
# weaker evidence.
_COMPLETED_BUILD_IDENTITY_REQUIRED_KEYS = (
    "effective_build_id",
    "compile_verification_id",
    "compile_commands_digest",
    "hip_compile_commands_digest",
    "runtime_bundle_hash",
    "runtime_artifacts",
)


def _require_completed_build_identity_shape(role: str, identity: object) -> None:
    if not isinstance(identity, dict):
        raise CampaignError(f"build identity for role {role!r} is not an object")
    missing = [key for key in _COMPLETED_BUILD_IDENTITY_REQUIRED_KEYS if key not in identity]
    if missing:
        raise CampaignError(
            f"build identity for role {role!r} is missing required field(s) {missing!r} -- "
            "expected the shape of builds.CompletedBuildEvidence.campaign_identity()"
        )
    for key in _COMPLETED_BUILD_IDENTITY_REQUIRED_KEYS:
        if key == "runtime_artifacts":
            if not isinstance(identity[key], dict) or not identity[key]:
                raise CampaignError(
                    f"build identity for role {role!r}: runtime_artifacts must be a "
                    "non-empty object"
                )
        elif not isinstance(identity[key], str) or not identity[key]:
            raise CampaignError(
                f"build identity for role {role!r}: {key!r} must be a non-empty string"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_content_identity(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise CampaignError(f"campaign identity input does not exist: {path}")
    stat = path.stat()
    return {"size": stat.st_size, "sha256": _sha256_file(path)}


def _stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# HI82 (design: GPT, req_f2b3c8914ec0498d): the canonical artifacts-directory
# manifest (autotune_catalog.emit()'s hip-autotune-manifest.json) is an
# evidence artifact, not a compile input -- its top-level generated_at
# legitimately advances on every `bigcherry generate` invocation even when
# the candidate catalog is unchanged. Hashing the raw file bytes for
# campaign resume identity (found for real: two back-to-back identical
# campaign runs on gfx1100 refused to resume solely because of this field)
# conflates "when was this evidence emitted" with "does this manifest mean
# the same thing to inventory/export/replay". Fail closed by default: only
# this one known-volatile top-level field is excluded, so an unforeseen new
# manifest field participates in identity until someone deliberately
# classifies it otherwise.
_MANIFEST_IDENTITY_VOLATILE_TOP_LEVEL_FIELDS = frozenset({"generated_at"})


def _manifest_semantic_identity(path: Path) -> dict[str, object]:
    """Stable identity of the manifest's campaign-relevant substance.

    manifest["manifest_hash"] (autotune_catalog.manifest_hash()) and
    manifest["build_descriptor"]["descriptor_hash"] are both already-stable,
    narrower fingerprints (catalog-only, compiled-descriptor-only) -- neither
    covers the complete manifest (e.g. source_revision, coverage,
    supported_coverage are outside both). Recorded here as diagnostics only;
    the authoritative resume key is the canonical hash of the whole manifest
    minus the one known-volatile field.
    """
    path = Path(path)
    if not path.is_file():
        raise CampaignError(f"campaign identity input does not exist: {path}")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read campaign manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CampaignError(f"campaign manifest root is not a JSON object: {path}")

    semantic_manifest = {
        key: value for key, value in manifest.items()
        if key not in _MANIFEST_IDENTITY_VOLATILE_TOP_LEVEL_FIELDS
    }

    identity: dict[str, object] = {
        "canonicalization": "canonical-json-v1",
        "excluded_top_level_fields": sorted(_MANIFEST_IDENTITY_VOLATILE_TOP_LEVEL_FIELDS),
        "sha256": _stable_json_sha256(semantic_manifest),
    }

    manifest_hash_value = manifest.get("manifest_hash")
    if isinstance(manifest_hash_value, str) and manifest_hash_value:
        identity["manifest_hash"] = manifest_hash_value

    build_descriptor = manifest.get("build_descriptor")
    if isinstance(build_descriptor, dict):
        descriptor_hash = build_descriptor.get("descriptor_hash")
        if isinstance(descriptor_hash, str) and descriptor_hash:
            identity["build_descriptor_hash"] = descriptor_hash

    return identity


def _identity_differences(old: object, new: object, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        differences: list[str] = []
        for key in sorted(set(old) | set(new)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old:
                differences.append(f"{child} added")
            elif key not in new:
                differences.append(f"{child} removed")
            else:
                differences.extend(_identity_differences(old[key], new[key], child))
        return differences

    if old != new:
        return [prefix or "<root>"]
    return []


@dataclass
class Campaign:
    model: Path
    tune_server: Path
    replay_server: Path
    manifest: Path
    workdir: Path
    port: int = 42301
    host: str = "127.0.0.1"
    n_gpu_layers: int = 99
    ctx_size: int = 4096
    stock_bench: Path | None = None
    tune_bench: Path | None = None
    replay_bench: Path | None = None
    # HI80: optional, patched test-backend-ops (patches 1222+1223 applied).
    # When given, S3b generates RV49 correctness evidence for rows that
    # cleared every statistical promotion criterion but were rejected purely
    # for missing evidence, then re-runs promotion. When omitted, S3b is a
    # no-op and those rows stay rejected -- exactly today's behavior.
    correctness_binary: Path | None = None
    bench_prompt: int = 512
    bench_gen: int = 128
    bench_repetitions: int = 3
    identity_context: CampaignIdentityContext | None = None

    def __post_init__(self) -> None:
        self.model = Path(self.model)
        self.tune_server = Path(self.tune_server)
        self.replay_server = Path(self.replay_server)
        self.manifest = Path(self.manifest)
        self.workdir = Path(self.workdir)

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.logdir = self.workdir / "logs"
        self.logdir.mkdir(exist_ok=True)
        self.status_path = self.workdir / "status.json"
        self.progress_path = self.workdir / "progress.jsonl"
        if self.stock_bench is not None:
            self.stock_bench = Path(self.stock_bench)
        if self.tune_bench is not None:
            self.tune_bench = Path(self.tune_bench)
        if self.replay_bench is not None:
            self.replay_bench = Path(self.replay_bench)
        if self.correctness_binary is not None:
            self.correctness_binary = Path(self.correctness_binary)
        self.bench_prompt = int(self.bench_prompt)
        self.bench_gen = int(self.bench_gen)
        self.bench_repetitions = int(self.bench_repetitions)
        # Lazy: file hashing belongs in run()/ensure_campaign_identity(),
        # where CampaignError is already part of the execution contract,
        # not during bare dataclass construction.
        self._campaign_identity_document: dict[str, object] | None = None
        if self.bench_prompt <= 0:
            raise ValueError("bench_prompt must be > 0")
        if self.bench_gen <= 0:
            raise ValueError("bench_gen must be > 0")
        if self.bench_repetitions <= 0:
            raise ValueError("bench_repetitions must be > 0")

    # -- campaign identity (HI82 item 9) --------------------------------

    def _make_campaign_identity(self) -> dict[str, object]:
        bench_enabled = bool(self.stock_bench and self.tune_bench and self.replay_bench)

        executable_files: dict[str, object] = {
            "tune_server": {
                "path": str(self.tune_server.resolve()),
                "file_identity": _file_content_identity(self.tune_server),
            },
            "replay_server": {
                "path": str(self.replay_server.resolve()),
                "file_identity": _file_content_identity(self.replay_server),
            },
        }
        if self.stock_bench is not None:
            executable_files["stock_bench"] = {
                "path": str(self.stock_bench.resolve()),
                "file_identity": _file_content_identity(self.stock_bench),
            }
        if self.tune_bench is not None:
            executable_files["tune_bench"] = {
                "path": str(self.tune_bench.resolve()),
                "file_identity": _file_content_identity(self.tune_bench),
            }
        if self.replay_bench is not None:
            executable_files["replay_bench"] = {
                "path": str(self.replay_bench.resolve()),
                "file_identity": _file_content_identity(self.replay_bench),
            }
        if self.correctness_binary is not None:
            executable_files["correctness_binary"] = {
                "path": str(self.correctness_binary.resolve()),
                "file_identity": _file_content_identity(self.correctness_binary),
            }

        if self.identity_context is None:
            patch_identity: dict[str, object] | None = None
            patched_source_tree = None
            gpu_architecture = None
            # Standalone CLI fallback: still safe against binary
            # replacement, but cannot invent build provenance that was not
            # supplied by the build layer.
            build_identities: dict[str, object] = {"standalone_executables": executable_files}
        else:
            patch_identity = {
                "name": self.identity_context.patch_name,
                "digest": self.identity_context.patch_digest,
            }
            patched_source_tree = self.identity_context.patched_source_tree
            gpu_architecture = self.identity_context.gpu_architecture
            build_identities = {
                name: dict(value)
                for name, value in sorted(self.identity_context.build_identities.items())
            }

            required_builds = {"tune", "replay"}
            if bench_enabled:
                required_builds.add("stock")
            missing = required_builds - set(build_identities)
            if missing:
                raise CampaignError(
                    f"campaign identity context is missing build identities: {sorted(missing)}"
                )
            for role in required_builds:
                _require_completed_build_identity_shape(role, build_identities[role])

        return {
            "schema_version": CAMPAIGN_IDENTITY_SCHEMA_VERSION,
            "campaign_script_version": CAMPAIGN_SCRIPT_VERSION,
            # The path tells the operator which model was intended; the
            # independent file identity detects replacement in place.
            "model_identity": {"path": str(self.model.resolve())},
            "model_file_identity": _file_content_identity(self.model),
            # The manifest affects dispatch-db/export semantics and must not
            # be allowed to change underneath resumed measurements. Its
            # canonical evidence file has an intentionally-live generated_at
            # timestamp, so identity binds the complete semantic JSON
            # payload (generated_at excluded) rather than the raw file bytes.
            "manifest_identity": {
                "path": str(self.manifest.resolve()),
                "semantic_identity": _manifest_semantic_identity(self.manifest),
            },
            "patch_identity": patch_identity,
            "patched_source_tree": patched_source_tree,
            "gpu_architecture": gpu_architecture,
            "build_identities": build_identities,
            # Actual executable files are included even when richer build
            # identities exist -- catches a binary replacement occurring
            # after build evidence was captured but before Campaign starts.
            "executables": executable_files,
            "campaign_parameters": {
                "n_gpu_layers": self.n_gpu_layers,
                "ctx_size": self.ctx_size,
                "completions": COMPLETIONS,
                "bench_enabled": bench_enabled,
                "bench_prompt": self.bench_prompt,
                "bench_gen": self.bench_gen,
                "bench_repetitions": self.bench_repetitions,
            },
        }

    def _campaign_identity(self) -> dict[str, object]:
        if self._campaign_identity_document is None:
            self._campaign_identity_document = self._make_campaign_identity()
        return self._campaign_identity_document

    @property
    def campaign_identity_digest(self) -> str:
        return _stable_json_sha256(self._campaign_identity())

    def _known_stage_artifacts(self) -> tuple[Path, ...]:
        tune_db = self.workdir / "tune.jsonl"
        candidates = (
            self.workdir / "record.jsonl",
            self.workdir / "inventory.json",
            self.workdir / "inventory.sqlite",
            tune_db,
            Path(f"{tune_db}.measurements.jsonl"),
            Path(f"{tune_db}.journal.jsonl"),
            self.workdir / "dispatch.sqlite",
            self.workdir / "promoted.jsonl",
            self.workdir / "correctness_evidence_input.jsonl",
            self.workdir / "dispatch.cache",
            self.workdir / "coverage.json",
            self.workdir / "bench.json",
            self.workdir / "report.md",
            # HI82 item 4: patch_validation_campaign.py writes this before
            # campaign.run() -- an orphaned activation.json without an
            # identity-bound status.json must refuse resume exactly like
            # every other stage artifact.
            self.workdir / "activation.json",
        )
        return tuple(path for path in candidates if path.exists())

    def _read_status_document(self) -> dict[str, Any]:
        try:
            value = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"cannot trust existing status.json: {exc}") from exc
        if not isinstance(value, dict):
            raise CampaignError("cannot trust existing status.json: root is not an object")
        return value

    def ensure_campaign_identity(self) -> dict[str, Any]:
        """Bind this workdir to exactly one campaign identity.

        Called before any stage can resume. A mismatch never deletes or
        overwrites prior evidence; it refuses the run -- reusing another
        workdir explicitly, or deleting the old campaign directory, is a
        human-visible operation, not something this does automatically.
        """
        expected = self._campaign_identity()
        expected_digest = self.campaign_identity_digest

        if not self.status_path.exists():
            existing_artifacts = self._known_stage_artifacts()
            if existing_artifacts:
                names = ", ".join(path.name for path in existing_artifacts[:8])
                more = (
                    "" if len(existing_artifacts) <= 8
                    else f" (+{len(existing_artifacts) - 8} more)"
                )
                raise CampaignError(
                    "campaign workdir contains stage artifacts but has no identity-bound "
                    f"status.json; refusing legacy/unattributed resume: {names}{more}"
                )

            data: dict[str, Any] = {
                "schema_version": STATUS_SCHEMA_VERSION,
                "campaign_identity": expected,
                "campaign_identity_digest": expected_digest,
                "history": [],
            }
            self._atomic_write_json(self.status_path, data)
            return data

        data = self._read_status_document()

        if data.get("schema_version") != STATUS_SCHEMA_VERSION:
            raise CampaignError(
                "existing status.json uses an untrusted/legacy status schema; refusing resume"
            )

        recorded = data.get("campaign_identity")
        recorded_digest = data.get("campaign_identity_digest")
        if not isinstance(recorded, dict):
            raise CampaignError("existing status.json has no campaign identity; refusing resume")
        if not isinstance(recorded_digest, str):
            raise CampaignError(
                "existing status.json has no campaign identity digest; refusing resume"
            )

        if _stable_json_sha256(recorded) != recorded_digest:
            raise CampaignError(
                "existing status.json campaign identity does not recompute; refusing resume"
            )

        # `recorded` came back from json.loads() (tuples became lists);
        # `expected` is still the freshly-built native dict (e.g.
        # COMPLETIONS is a tuple there) -- round-trip `expected` through
        # JSON too before any raw dict comparison, or a semantically
        # identical value would spuriously "differ" by container type even
        # though campaign_identity_digest (itself JSON-based) already
        # proves equivalence.
        expected_canonical = json.loads(json.dumps(expected, sort_keys=True, ensure_ascii=False))

        if recorded_digest != expected_digest or recorded != expected_canonical:
            differences = _identity_differences(recorded, expected_canonical)
            summary = ", ".join(differences[:12])
            if len(differences) > 12:
                summary += f" (+{len(differences) - 12} more)"
            raise CampaignError(
                "campaign identity mismatch; refusing to resume stale stage artifacts"
                + (f": {summary}" if summary else "")
            )

        return data

    # -- status/progress -----------------------------------------------

    def write_status(self, stage: str, state: str, detail: str = "") -> None:
        # Every mutation re-validates the identity-bound document -- an
        # individual stage invoked directly still cannot reuse an
        # unbound/mismatched workdir, not just the top-level run() gate.
        data = self.ensure_campaign_identity()

        record = {
            "stage": stage,
            "state": state,
            "detail": detail,
            "campaign_identity_digest": self.campaign_identity_digest,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        data["current"] = record
        history = data.setdefault("history", [])
        if not isinstance(history, list):
            raise CampaignError("existing status.json history is invalid")
        history.append(record)

        self._atomic_write_json(self.status_path, data)
        print(f"[{record['ts']}] {stage} {state} {detail}", flush=True)

    def sample_progress(self, stage: str, journal_path: Path, t0: float) -> None:
        lines = 0
        if journal_path.exists():
            with journal_path.open(encoding="utf-8", errors="replace") as f:
                lines = sum(1 for _ in f)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "journal_lines": lines,
            "elapsed_s": round(time.time() - t0, 1),
            "campaign_identity_digest": self.campaign_identity_digest,
        }
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # -- server lifecycle -------------------------------------------------

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _wait_healthy(self, proc: subprocess.Popen, timeout_s: int = 180) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # Fail fast on an early crash (e.g. a missing ROCm DLL on PATH)
            # instead of spinning for the full timeout on a process that
            # already exited.
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(
                    f"{self._base_url()}/health", timeout=2
                ) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(3)
        return False

    def _post_json(self, path: str, payload: dict, timeout_s: int = 120) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url()}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s):
                pass
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise CampaignError(f"POST {path} failed: {exc}") from exc

    def _run_completions(self) -> None:
        for payload in COMPLETIONS:
            self._post_json("/completion", payload)

    def _shutdown(self, proc: subprocess.Popen, timeout_s: int = 90) -> None:
        try:
            self._post_json("/shutdown", {})
        except CampaignError:
            pass
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    def _launch(
        self, server: Path, env_overrides: dict[str, str], log_path: Path
    ) -> subprocess.Popen:
        env = dict(os.environ)
        env.update(env_overrides)
        # LLAMA_SERVER_ENABLE_SHUTDOWN gates the opt-in /shutdown route
        # (patches/0800_server_shutdown_endpoint.py) -- without it a plain
        # process kill on Windows skips backend teardown and silently
        # discards buffered HIP autotune measurements (found the hard way
        # this session: an earlier version of this campaign used a bare
        # kill and every tune run came back empty).
        env["LLAMA_SERVER_ENABLE_SHUTDOWN"] = "1"
        args = [
            str(server), "-m", str(self.model),
            "--port", str(self.port), "--host", self.host,
            "-ngl", str(self.n_gpu_layers), "-fa", "on",
            "--ctx-size", str(self.ctx_size), "--no-webui",
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                args, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
        return proc

    def run_server_stage(
        self, stage: str, server: Path, env_overrides: dict[str, str],
        journal_path: Path | None = None,
    ) -> None:
        log_path = self.logdir / f"{stage}-server.log"
        proc = self._launch(server, env_overrides, log_path)
        sampler_stop: threading.Event | None = None
        sampler_thread: threading.Thread | None = None
        try:
            if not self._wait_healthy(proc):
                tail = _tail(log_path)
                exited = proc.poll() is not None
                reason = f"process exited (code {proc.returncode})" if exited \
                    else "health check timed out"
                raise CampaignError(f"{stage}: server did not come up -- {reason}: {tail}")
            if journal_path is not None:
                # Sub-minute progress, on its own timer: hip-autotune-
                # journal.cpp fwrite+fflush+fsync's a record per candidate
                # attempt, so this file's line count is a live, durable
                # signal -- not a stale poll. Runs on a background thread so
                # it keeps sampling every PROGRESS_INTERVAL_S seconds while
                # the (blocking) completion requests below are in flight,
                # the same as the bash prototype's concurrent sampler loop.
                sampler_stop = threading.Event()
                t0 = time.time()

                def _sample_loop() -> None:
                    while not sampler_stop.is_set():
                        self.sample_progress(stage, journal_path, t0)
                        sampler_stop.wait(PROGRESS_INTERVAL_S)

                sampler_thread = threading.Thread(target=_sample_loop, daemon=True)
                sampler_thread.start()
            self._run_completions()
        finally:
            if sampler_stop is not None:
                sampler_stop.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=PROGRESS_INTERVAL_S + 5)
            self._shutdown(proc)

    # -- stages -------------------------------------------------------

    def s1_record(self) -> Path:
        db = self.workdir / "record.jsonl"
        stage = "S1_record"
        if db.exists() and db.stat().st_size > 0:
            self.write_status(stage, "done", f"{_count_lines(db)} observations (resumed)")
            return db
        self.write_status(stage, "running")
        db.unlink(missing_ok=True)
        self.run_server_stage(
            stage, self.tune_server,
            {"GGML_HIP_DISPATCH_MODE": "record", "GGML_HIP_DISPATCH_DB": str(db)},
        )
        if not db.exists() or db.stat().st_size == 0:
            raise CampaignError(f"{stage}: no record output")
        self.write_status(stage, "done", f"{_count_lines(db)} observations")
        return db

    def s1b_inventory(self, record_db: Path) -> tuple[Path, Path]:
        stage = "S1b_inventory"
        inventory_json = self.workdir / "inventory.json"
        inventory_sqlite = self.workdir / "inventory.sqlite"
        if inventory_json.exists() and inventory_sqlite.exists():
            self.write_status(stage, "done", "resumed")
            return inventory_json, inventory_sqlite
        self.write_status(stage, "running")
        _run_bigcherry(
            "inventory", "record", str(record_db),
            "--inventory", str(inventory_json), "--database", str(inventory_sqlite),
        )
        self.write_status(stage, "done")
        return inventory_json, inventory_sqlite

    def s2_tune(self) -> Path:
        tdb = self.workdir / "tune.jsonl"
        measurements = Path(f"{tdb}.measurements.jsonl")
        journal = Path(f"{tdb}.journal.jsonl")
        stage = "S2_tune"
        if measurements.exists() and measurements.stat().st_size > 0:
            self.write_status(stage, "done", f"{_count_lines(measurements)} lines (resumed)")
            return measurements
        self.write_status(stage, "running")
        for p in (tdb, measurements, journal):
            p.unlink(missing_ok=True)
        self.run_server_stage(
            stage, self.tune_server,
            {
                "GGML_HIP_DISPATCH_MODE": "tune",
                "GGML_HIP_DISPATCH_DB": str(tdb),
                "GGML_HIP_TUNE_SCREEN_SAMPLES": "3",
                "GGML_HIP_TUNE_FINAL_SAMPLES": "10",
                # HI64: on Windows/WDDM a single transient HIP timing flake
                # permanently poisons tuning for the rest of the process
                # (fail-closed by design; in-process recovery is out of
                # scope until that item lands). HIP_LAUNCH_BLOCKING=1 is
                # the item's own accepted local-gate workaround -- timings
                # become non-authoritative but the run completes, which is
                # what a process-validation smoke test needs.
                "HIP_LAUNCH_BLOCKING": "1",
            },
            journal_path=journal,
        )
        if not measurements.exists() or measurements.stat().st_size == 0:
            raise CampaignError(f"{stage}: no measurements output")
        self.write_status(stage, "done", f"{_count_lines(measurements)} lines")
        return measurements

    def s2c_dispatch_db(self, measurements: Path) -> Path:
        stage = "S2c_dispatch_db"
        dispatch_db = self.workdir / "dispatch.sqlite"
        self.write_status(stage, "running")
        _run_bigcherry(
            "inventory", "tuning", str(measurements),
            "--database", str(dispatch_db), "--manifest", str(self.manifest),
        )
        self.write_status(stage, "done")
        return dispatch_db

    def s3_promote(self, measurements: Path, dispatch_db: Path) -> Path:
        stage = "S3_promote"
        promoted = self.workdir / "promoted.jsonl"
        self.write_status(stage, "running")
        out = _run_bigcherry(
            "tune-promote", str(measurements),
            "--output", str(promoted), "--dispatch-db", str(dispatch_db),
        )
        self.write_status(stage, "done", out.strip().splitlines()[-1] if out.strip() else "")
        return promoted

    def s3b_correctness_evidence(
        self, measurements: Path, dispatch_db: Path, promoted: Path,
    ) -> Path:
        """HI80: optional (no-op unless --correctness-binary was given).

        s3_promote() already ran once above and rejected some rows with
        promotion_status == "rejected_no_correctness_evidence" -- a status
        promote() only ever assigns AFTER every statistical criterion (BH,
        bootstrap CI, effect threshold) already passed, since the
        correctness gate is evaluated last as a hard AND, never a rescue
        (see tune_promotion.promote()'s own comments). Generating evidence
        for exactly those rows and re-running promotion is therefore never
        spent on a candidate that would have failed statistically anyway.

        promote() is not safely rerunnable on its own OUTPUT -- a rejected
        row's promotion_status is not one of the pending_bh/
        confirmation_rejected values promote() requires to reconsider a row,
        so feeding promoted.jsonl back in would raise "unknown promotion
        status". Instead this selects the blocked rows' ORIGINAL (untouched,
        still pending_bh) entries out of `measurements`, generates evidence
        for just those against `dispatch_db`, then re-runs s3_promote against
        the original measurements file -- unchanged except that dispatch_db
        now carries evidence for the previously-blocked dispatches.
        """
        stage = "S3b_correctness_evidence"
        if self.correctness_binary is None:
            self.write_status(stage, "done", "skipped (no --correctness-binary given)")
            return promoted
        self.write_status(stage, "running")

        blocked_dispatches: set[str] = set()
        for line in promoted.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("promotion_status") == "rejected_no_correctness_evidence":
                blocked_dispatches.add(row.get("dispatch"))
        if not blocked_dispatches:
            self.write_status(stage, "done", "no rows blocked purely on missing correctness evidence")
            return promoted

        header_line: str | None = None
        selected_lines: list[str] = []
        for line in measurements.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("kind") == "header":
                header_line = stripped
            elif row.get("dispatch") in blocked_dispatches:
                selected_lines.append(stripped)
        if header_line is None or not selected_lines:
            raise CampaignError(
                f"{stage}: could not locate the blocked dispatch rows in the "
                "original measurements file"
            )

        evidence_input = self.workdir / "correctness_evidence_input.jsonl"
        evidence_input.write_text(
            header_line + "\n" + "\n".join(selected_lines) + "\n", encoding="utf-8"
        )

        _run_module(
            "bigcherry.hi80_generate_correctness_evidence", str(evidence_input),
            "--dispatch-db", str(dispatch_db), "--binary", str(self.correctness_binary),
        )

        repromoted = self.s3_promote(measurements, dispatch_db)
        self.write_status(
            stage, "done",
            f"{len(blocked_dispatches)} row(s) evidenced, re-ran S3_promote",
        )
        return repromoted

    def s4_export(self, promoted: Path, dispatch_db: Path) -> Path:
        stage = "S4_export"
        cache = self.workdir / "dispatch.cache"
        self.write_status(stage, "running")
        _run_module(
            "bigcherry.replay_cache", str(promoted),
            "--manifest", str(self.manifest), "--output", str(cache),
            "--dispatch-db", str(dispatch_db),
        )
        if not cache.exists() or cache.stat().st_size == 0:
            raise CampaignError(f"{stage}: no cache produced")
        self.write_status(stage, "done", f"{cache.stat().st_size} bytes")
        return cache

    def s5_replay(self, cache: Path) -> Path:
        stage = "S5_replay"
        coverage = self.workdir / "coverage.json"
        self.write_status(stage, "running")
        coverage.unlink(missing_ok=True)
        self.run_server_stage(
            stage, self.replay_server,
            {
                "GGML_HIP_DISPATCH_MODE": "replay",
                "GGML_HIP_DISPATCH_CACHE": str(cache),
                "GGML_HIP_DISPATCH_COVERAGE": str(coverage),
            },
        )
        if not coverage.exists():
            raise CampaignError(f"{stage}: no coverage output")
        self.write_status(stage, "done", coverage.read_text(encoding="utf-8"))
        return coverage

    # -- bench (stock vs native vs replay tokens/sec) -----------------------

    @staticmethod
    def _bench_clean_env() -> dict[str, str]:
        """A copy of the process environment with all BigCherry HIP dispatch
        configuration removed, so a parent shell or a previous bench arm
        cannot leak dispatch mode/cache configuration into the next arm."""
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("GGML_HIP_DISPATCH_") or key.startswith("BIGCHERRY_"):
                env.pop(key, None)
        return env

    @staticmethod
    def _bench_parse_json(stdout: str) -> list[dict[str, Any]]:
        text = stdout.strip()
        if not text:
            raise CampaignError("llama-bench produced no JSON output")
        # llama-bench.exe prints a non-JSON diagnostic line ("HIP Library
        # Path: ...") to stdout before the JSON payload on Windows/ROCm --
        # skip any preamble up to the first '[' or '{'.
        start = min(
            (i for i in (text.find("["), text.find("{")) if i != -1),
            default=-1,
        )
        if start > 0:
            text = text[start:]
        try:
            value = json.loads(text)
        except json.JSONDecodeError as array_error:
            # Tolerate a JSONL variant (one object per line) as a fallback.
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise CampaignError(
                        "failed to parse llama-bench JSON output"
                    ) from array_error
                if not isinstance(row, dict):
                    raise CampaignError(
                        f"llama-bench JSONL row {line_number} is not an object"
                    )
                rows.append(row)
            if not rows:
                raise CampaignError(
                    "failed to parse llama-bench JSON output"
                ) from array_error
            return rows

        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and isinstance(value.get("results"), list):
            rows = value["results"]
        elif isinstance(value, dict):
            rows = [value]
        else:
            raise CampaignError(
                f"unexpected llama-bench JSON root: {type(value).__name__}"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CampaignError(f"llama-bench JSON row {index} is not an object")
        return rows

    @staticmethod
    def _bench_int(row: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _bench_float(row: dict[str, Any], key: str, *, description: str) -> float:
        value = row.get(key)
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise CampaignError(f"{description} has invalid {key!r}: {value!r}") from exc
        if not math.isfinite(result):
            raise CampaignError(f"{description} has non-finite {key!r}: {value!r}")
        return result

    def _bench_select_metric_row(
        self, rows: list[dict[str, Any]], *, workload: str,
    ) -> dict[str, Any]:
        if workload not in {"pp", "tg"}:
            raise ValueError(f"unknown benchmark workload: {workload!r}")
        exact: list[dict[str, Any]] = []
        compatible: list[dict[str, Any]] = []
        labelled: list[dict[str, Any]] = []
        for row in rows:
            n_prompt = self._bench_int(row, "n_prompt", "n_prompts", "pp")
            n_gen = self._bench_int(row, "n_gen", "n_generation", "tg")
            if workload == "pp":
                if n_prompt == self.bench_prompt and (n_gen is None or n_gen == 0):
                    exact.append(row)
                elif n_prompt is not None and n_prompt > 0 and (n_gen is None or n_gen == 0):
                    compatible.append(row)
            else:
                if n_gen == self.bench_gen and (n_prompt is None or n_prompt == 0):
                    exact.append(row)
                elif n_gen is not None and n_gen > 0 and (n_prompt is None or n_prompt == 0):
                    compatible.append(row)
            test_label = str(
                row.get("test") or row.get("test_name") or row.get("name") or ""
            ).strip().lower()
            if workload == "pp" and test_label.startswith("pp"):
                labelled.append(row)
            elif workload == "tg" and test_label.startswith("tg"):
                labelled.append(row)
        candidates = exact or compatible or labelled
        deduplicated: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for row in candidates:
            if id(row) in seen_ids:
                continue
            seen_ids.add(id(row))
            deduplicated.append(row)
        if len(deduplicated) != 1:
            summary = [
                {"n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                 "test": r.get("test"), "avg_ts": r.get("avg_ts")}
                for r in rows
            ]
            raise CampaignError(
                f"expected exactly one llama-bench {workload} row, "
                f"found {len(deduplicated)}; rows={summary!r}"
            )
        return deduplicated[0]

    def _bench_metrics(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for workload in ("pp", "tg"):
            row = self._bench_select_metric_row(rows, workload=workload)
            description = f"llama-bench {workload} row"
            result[workload] = {
                "avg_ts": self._bench_float(row, "avg_ts", description=description),
                "stddev_ts": self._bench_float(row, "stddev_ts", description=description),
            }
        return result

    def _bench_build_identity(self, build_role: str) -> dict[str, object] | None:
        """The already-verified build identity for one bench arm, sourced
        from CompletedBuildEvidence.campaign_identity() via
        CampaignIdentityContext (patch_validation_campaign.py) -- not a
        second, invented provenance authority. Direct standalone
        e2e_smoke_campaign.py use has no CampaignIdentityContext, so this
        returns None there; that campaign's resume identity is still bound
        to executable file content by _make_campaign_identity()."""
        if self.identity_context is None:
            return None
        identity = self.identity_context.build_identities.get(build_role)
        if identity is None:
            raise CampaignError(
                f"S6_bench: CampaignIdentityContext has no build identity for "
                f"required build role {build_role!r}"
            )
        # A new outer mapping, so attaching bench metadata elsewhere can
        # never mutate CampaignIdentityContext's authoritative identity.
        return dict(identity)

    def _run_bench_config(
        self, *, name: str, binary: Path, dispatch_mode: str | None,
        build_role: str, build_identity: dict[str, object] | None,
        dispatch_cache: Path | None = None,
    ) -> dict[str, Any]:
        binary = Path(binary)
        if not binary.is_file():
            raise CampaignError(f"{name} llama-bench binary does not exist: {binary}")
        if dispatch_cache is not None:
            dispatch_cache = Path(dispatch_cache)
            if not dispatch_cache.is_file():
                raise CampaignError(f"{name} dispatch cache does not exist: {dispatch_cache}")
        env = self._bench_clean_env()
        env["LLAMA_SERVER_ENABLE_SHUTDOWN"] = "1"
        if dispatch_mode is not None:
            env["GGML_HIP_DISPATCH_MODE"] = dispatch_mode
        if dispatch_cache is not None:
            env["GGML_HIP_DISPATCH_CACHE"] = str(dispatch_cache.resolve())
        command = [
            str(binary.resolve()), "-m", str(Path(self.model).resolve()),
            "-p", str(self.bench_prompt), "-n", str(self.bench_gen),
            "-r", str(self.bench_repetitions), "-o", "json",
        ]
        log_path = self.logdir / f"bench-{name}.log"
        completed = subprocess.run(
            command, cwd=self.workdir, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        log_path.write_text(
            f"command: {command!r}\n\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise CampaignError(
                f"{name} llama-bench failed with exit code {completed.returncode} "
                f"(see {log_path})"
            )
        rows = self._bench_parse_json(completed.stdout)
        result: dict[str, Any] = {
            "binary": str(binary.resolve()),
            "dispatch_mode": dispatch_mode,
            # HI82 item 8: bind this measurement arm to the exact completed
            # build whose executable/runtime closure produced these numbers.
            # build_role is deliberately separate from `name`: the "native"
            # arm runs the TUNE build in native dispatch mode, so its real
            # role is "tune", not a fourth build.
            "build_role": build_role,
            "build_identity": build_identity,
            "metrics": self._bench_metrics(rows),
            "rows": rows,
        }
        if dispatch_cache is not None:
            result["dispatch_cache"] = str(dispatch_cache.resolve())
        return result

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def s6_bench(self, cache: Path) -> Path:
        """Benchmark the model three ways: stock (separate unpatched binary,
        no BigCherry dispatch env at all), native (dispatch on, no tuning
        influence), and replay (the promoted/tuned cache applied). Each arm
        runs as its own subprocess from a freshly cleaned environment so no
        GGML_HIP_DISPATCH_* state leaks between arms."""
        stage = "S6_bench"
        output = self.workdir / "bench.json"
        if output.exists():
            self.write_status(stage, "done", "resumed, already present")
            return output
        self.write_status(stage, "running")
        stock = self._run_bench_config(
            name="stock", binary=self.stock_bench, dispatch_mode=None,
            build_role="stock", build_identity=self._bench_build_identity("stock"),
        )
        native = self._run_bench_config(
            name="native", binary=self.tune_bench, dispatch_mode="native",
            build_role="tune", build_identity=self._bench_build_identity("tune"),
        )
        replay = self._run_bench_config(
            name="replay", binary=self.replay_bench, dispatch_mode="replay",
            build_role="replay", build_identity=self._bench_build_identity("replay"),
            dispatch_cache=cache,
        )
        payload: dict[str, Any] = {
            # v2 adds explicit per-arm completed-build provenance
            # (build_role/build_identity) -- see HI82 item 8.
            "schema_version": 2,
            "model": str(Path(self.model).resolve()),
            "params": {
                "n_prompt": self.bench_prompt, "n_gen": self.bench_gen,
                "repetitions": self.bench_repetitions,
            },
            "configs": {"stock": stock, "native": native, "replay": replay},
        }
        self._atomic_write_json(output, payload)
        self.write_status(stage, "done")
        return output

    def s7_report(self, bench_path: Path, measurements: Path) -> Path:
        stage = "S7_report"
        report_path = self.workdir / "report.md"
        self.write_status(stage, "running")
        from .e2e_smoke_report import generate_report
        generate_report(
            self.workdir, bench_path=bench_path, measurements_path=measurements,
            output_path=report_path,
        )
        self.write_status(stage, "done")
        return report_path

    def run(self) -> dict:
        # HI82 item 9: nothing below this point may inspect an existing
        # stage output for resume until the entire workdir has been proven
        # to belong to this exact campaign identity.
        self.ensure_campaign_identity()

        record_db = self.s1_record()
        _inventory_json, _inventory_sqlite = self.s1b_inventory(record_db)
        measurements = self.s2_tune()
        dispatch_db = self.s2c_dispatch_db(measurements)
        promoted = self.s3_promote(measurements, dispatch_db)
        promoted = self.s3b_correctness_evidence(measurements, dispatch_db, promoted)
        cache = self.s4_export(promoted, dispatch_db)
        coverage = self.s5_replay(cache)
        if self.stock_bench and self.tune_bench and self.replay_bench:
            bench_path = self.s6_bench(cache)
            self.s7_report(bench_path, measurements)
        self.write_status("CAMPAIGN", "done", "all stages complete")
        return json.loads(coverage.read_text(encoding="utf-8"))


def _tail(path: Path, n: int = 15) -> str:
    if not path.exists():
        return "(no log)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def _run_bigcherry(*args: str) -> str:
    return _run_module("bigcherry", *args)


def _run_module(module: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CampaignError(
            f"{module} {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry e2e-smoke-campaign")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tune-server", required=True, type=Path)
    parser.add_argument("--replay-server", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=42301)
    parser.add_argument(
        "--stock-bench", type=Path, default=None,
        help="unpatched/stock llama-bench binary; enables the bench+report "
             "stages when given together with --tune-bench/--replay-bench",
    )
    parser.add_argument("--tune-bench", type=Path, default=None,
                         help="bigcherry llama-bench binary for the native-dispatch bench arm")
    parser.add_argument("--replay-bench", type=Path, default=None,
                         help="bigcherry llama-bench binary for the replay-dispatch bench arm")
    parser.add_argument(
        "--correctness-binary", type=Path, default=None,
        help="patched test-backend-ops (patches 1222+1223 applied); enables "
             "HI80's S3b stage, generating RV49 correctness evidence for "
             "statistically-ready-but-unevidenced rows and re-promoting",
    )
    parser.add_argument("--bench-prompt", type=int, default=512)
    parser.add_argument("--bench-gen", type=int, default=128)
    parser.add_argument("--bench-repetitions", type=int, default=3)
    args = parser.parse_args(argv)

    campaign = Campaign(
        model=args.model, tune_server=args.tune_server,
        replay_server=args.replay_server, manifest=args.manifest,
        workdir=args.workdir, port=args.port,
        stock_bench=args.stock_bench, tune_bench=args.tune_bench,
        replay_bench=args.replay_bench, correctness_binary=args.correctness_binary,
        bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen, bench_repetitions=args.bench_repetitions,
    )
    try:
        coverage = campaign.run()
    except CampaignError as exc:
        try:
            campaign.write_status("CAMPAIGN", "failed", str(exc))
        except CampaignError:
            # Most importantly: an identity mismatch must not mutate the
            # existing status document merely to record that this new,
            # mismatched invocation failed.
            pass
        print(f"CAMPAIGN FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(coverage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
