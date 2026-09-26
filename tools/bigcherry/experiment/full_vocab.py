"""Full-vocabulary logprob comparison between two llama-server arms.

A fixed temperature-0 streaming /completion request is sent to a control
server and then a subject server. Every generated step returns the logprob
of every vocabulary token; the control's rows are spooled to disk as
little-endian f64 so memory stays flat, and the subject is compared step by
step against a ``Criterion``:

* ``BIT_IDENTICAL``: every logprob of every compared step equal (patches that
  claim bit identity);
* a distribution criterion (``NEAR_LOSSLESS``) for patches that are not
  bit-exact (PVPS06, GPT review req_4d131e8b7c1c452d): identical generated
  tokens, per-step KL(control || subject) bounded, the control's top-p set
  keeps its mass under the subject, and the logprob difference bounded only
  for materially probable tokens -- an absolute bound over every vocab entry
  overweights meaningless extreme-tail rounding (1207 failed on a token of
  probability ~1e-13).

After the first generated-token mismatch only that step is still comparable
(its input context was identical); later steps are not the same experiment.

Callers supply the server sessions (``AttestedServerSession`` instances or
anything with a ``base_url``); this module never starts a process.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from array import array
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class FullVocabError(RuntimeError):
    """Malformed server output or a protocol failure; never a verdict."""


@dataclass(frozen=True)
class Criterion:
    name: str
    #: True: every logprob must match exactly; the thresholds below are unused.
    bit_identical: bool
    max_kl: float = 0.0
    #: Tokens with probability above this in either arm are "material".
    material_prob: float = 0.0
    #: Max |logprob difference| over material tokens.
    material_logprob_tol: float = 0.0
    topp: float = 0.95
    #: Minimum subject probability mass on the control's top-p token set.
    min_topp_mass: float = 0.0

    def document(self) -> dict[str, object]:
        return dataclasses.asdict(self)


BIT_IDENTICAL = Criterion(name="bit-identical", bit_identical=True)
# Near-lossless kernels (bf16 operands, reordered accumulation, fusion
# placement). KL 5e-3 nats is far below any decoding-visible shift; 0.02
# logprob is ~2% relative probability on a token above 1e-4; the control's
# 95% nucleus keeping >= 94% under the subject bounds nucleus drift.
# Generated tokens must still match exactly.
NEAR_LOSSLESS = Criterion(
    name="near-lossless-v1", bit_identical=False, max_kl=5e-3, material_prob=1e-4,
    material_logprob_tol=0.02, topp=0.95, min_topp_mass=0.94,
)


class _Session(Protocol):
    @property
    def base_url(self) -> str: ...


SessionFactory = Callable[[], AbstractContextManager[Any]]


@dataclass(frozen=True)
class FullVocabComparison:
    passed: bool
    detail: str
    vocab_size: int
    decode_steps: int
    decode_steps_compared: int
    criterion: Criterion
    generated_tokens_match: bool
    first_generated_token_mismatch: dict[str, int] | None
    max_abs_logprob_diff: float
    worst: dict[str, float | int] | None
    max_kl: float
    max_material_logprob_diff: float
    worst_material: dict[str, float | int] | None
    min_topp_mass: float
    control_logprobs_sha256: str
    subject_logprobs_sha256: str
    request: dict[str, object]
    attestations: dict[str, object] = field(default_factory=dict)

    def document(self) -> dict[str, object]:
        return {
            "method": "llama-server-streaming-full-vocab-logprob",
            "passed": self.passed,
            "detail": self.detail,
            "vocab_size": self.vocab_size,
            "decode_steps": self.decode_steps,
            "decode_steps_compared": self.decode_steps_compared,
            "logprobs_compared": self.decode_steps_compared * self.vocab_size,
            "criterion": self.criterion.document(),
            "generated_tokens_match": self.generated_tokens_match,
            "first_generated_token_mismatch": self.first_generated_token_mismatch,
            "max_abs_logprob_diff": self.max_abs_logprob_diff,
            "worst": self.worst,
            "max_kl": self.max_kl,
            "max_material_logprob_diff": self.max_material_logprob_diff,
            "worst_material": self.worst_material,
            "min_topp_mass": self.min_topp_mass,
            "control_logprobs_sha256": self.control_logprobs_sha256,
            "subject_logprobs_sha256": self.subject_logprobs_sha256,
            "request": self.request,
            "execution_attestation": self.attestations,
        }


def request_payload(prompt: str, *, n_predict: int, vocab_size: int) -> dict[str, object]:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "n_probs": vocab_size,
        "post_sampling_probs": False,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": True,
        "return_tokens": True,
        "stream": True,
    }


def served_vocab_size(base_url: str, *, timeout_s: int = 60) -> int:
    """llama-server's /v1/models data[0].meta.n_vocab."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout_s) as response:
            models = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise FullVocabError(f"cannot read /v1/models: {exc}") from exc
    data = models.get("data") if isinstance(models, Mapping) else None
    meta = data[0].get("meta") if isinstance(data, list) and data and isinstance(data[0], Mapping) else None
    value = meta.get("n_vocab") if isinstance(meta, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FullVocabError("llama-server /v1/models does not report meta.n_vocab")
    return value


def stream_completion_rows(
    base_url: str, payload: Mapping[str, object], *, timeout_s: int
) -> Iterable[Mapping[str, object]]:
    """Yield one probability row per generated token (``stream=true`` SSE)."""
    request = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    saw_stop = False
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith((":", "event:")):
                    continue
                if not line.startswith("data:"):
                    raise FullVocabError(f"malformed SSE line: {line[:120]!r}")
                encoded = line.removeprefix("data:").strip()
                if encoded == "[DONE]":
                    saw_stop = True
                    break
                try:
                    event = json.loads(encoded)
                except json.JSONDecodeError as exc:
                    raise FullVocabError("malformed JSON in completion stream") from exc
                if not isinstance(event, Mapping) or not isinstance(event.get("stop"), bool):
                    raise FullVocabError("completion stream event lacks a boolean stop")
                if event["stop"]:
                    saw_stop = True
                    continue
                rows = event.get("completion_probabilities")
                # A speculative (MTP/draft) server streams every token accepted
                # in one verify step as one event, and can emit events that carry
                # no token at all. Rows are counted per arm against n_predict by
                # compare_servers(), so an empty event cannot hide a lost step.
                if rows is None:
                    continue
                if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
                    raise FullVocabError("completion_probabilities must be a list of row objects")
                yield from rows
    except (urllib.error.URLError, OSError, TimeoutError, UnicodeError) as exc:
        raise FullVocabError(f"streaming /completion failed: {exc}") from exc
    if not saw_stop:
        raise FullVocabError("completion stream ended without a stop event")


def dense_logprobs(row: Mapping[str, object], *, vocab_size: int, arm: str, step: int) -> tuple[int, array]:
    """Validate one full-vocabulary row; return (generated id, id-indexed logprobs)."""
    generated_id = row.get("id")
    if not isinstance(generated_id, int) or isinstance(generated_id, bool) or not 0 <= generated_id < vocab_size:
        raise FullVocabError(f"{arm} step {step} has an invalid generated token id")
    top = row.get("top_logprobs")
    if not isinstance(top, list) or len(top) != vocab_size:
        actual = len(top) if isinstance(top, list) else None
        raise FullVocabError(f"{arm} step {step} is not full-vocabulary: expected {vocab_size}, got {actual!r}")
    values = array("d", [math.nan]) * vocab_size
    seen = bytearray(vocab_size)
    for entry in top:
        token_id = entry.get("id") if isinstance(entry, Mapping) else None
        logprob = entry.get("logprob") if isinstance(entry, Mapping) else None
        if not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id < vocab_size:
            raise FullVocabError(f"{arm} step {step} has an invalid vocabulary token id {token_id!r}")
        if seen[token_id]:
            raise FullVocabError(f"{arm} step {step} duplicates token id {token_id}")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool) or not math.isfinite(float(logprob)):
            raise FullVocabError(f"{arm} step {step} token {token_id} has an invalid logprob")
        seen[token_id] = 1
        values[token_id] = float(logprob)
    return generated_id, values


def canonical_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    copied = array("d", values)
    copied.byteswap()
    return copied.tobytes()


@dataclass
class StepMetrics:
    kl: float
    max_abs_diff: float
    worst_token: int
    material_diff: float
    worst_material_token: int
    topp_mass: float


def step_metrics(control: array, subject: array, criterion: Criterion) -> StepMetrics:
    """Distribution comparison of one step's full-vocabulary logprobs."""
    kl = 0.0
    max_abs = -1.0
    worst = -1
    material = 0.0
    worst_material = -1
    threshold = math.log(criterion.material_prob) if criterion.material_prob > 0 else -math.inf
    for token_id, (c, s) in enumerate(zip(control, subject, strict=True)):
        delta = abs(s - c)
        if delta > max_abs:
            max_abs, worst = delta, token_id
        pc = math.exp(c)
        if pc > 0.0:
            kl += pc * (c - s)
        if (c > threshold or s > threshold) and delta > material:
            material, worst_material = delta, token_id
    order = sorted(range(len(control)), key=control.__getitem__, reverse=True)
    mass_c = 0.0
    mass_s = 0.0
    for token_id in order:
        if mass_c >= criterion.topp:
            break
        mass_c += math.exp(control[token_id])
        mass_s += math.exp(subject[token_id])
    return StepMetrics(kl=max(kl, 0.0), max_abs_diff=max_abs, worst_token=worst,
                       material_diff=material, worst_material_token=worst_material, topp_mass=mass_s)


def compare_servers(
    *,
    control_session: SessionFactory,
    subject_session: SessionFactory,
    prompt: str,
    n_predict: int,
    criterion: Criterion,
    scratch_dir: Path,
    timeout_s: int = 1800,
) -> FullVocabComparison:
    """Run the request on control then subject and compare every step."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spool_path = scratch_dir / "control-full-vocab-logprobs.f64"
    control_tokens: list[int] = []
    control_digest = hashlib.sha256()
    subject_digest = hashlib.sha256()
    attestations: dict[str, object] = {}
    max_abs_diff = 0.0
    worst: dict[str, float | int] | None = None
    max_kl = 0.0
    max_material = 0.0
    worst_material: dict[str, float | int] | None = None
    min_topp_mass = math.inf
    first_mismatch: dict[str, int] | None = None
    comparable = 0
    subject_steps = 0

    def _attest(role: str, session: Any) -> None:
        attestation = getattr(session, "attestation", None)
        if attestation is not None:
            attestations[role] = attestation.document()

    try:
        with control_session() as control, spool_path.open("wb") as spool:
            _attest("control", control)
            vocab_size = served_vocab_size(control.base_url)
            payload = request_payload(prompt, n_predict=n_predict, vocab_size=vocab_size)
            for step, row in enumerate(stream_completion_rows(control.base_url, payload, timeout_s=timeout_s)):
                if step >= n_predict:
                    raise FullVocabError("control emitted more decode steps than requested")
                token, values = dense_logprobs(row, vocab_size=vocab_size, arm="control", step=step)
                encoded = canonical_bytes(values)
                spool.write(encoded)
                control_digest.update(encoded)
                control_tokens.append(token)
            if len(control_tokens) != n_predict:
                raise FullVocabError(f"control emitted {len(control_tokens)} steps; expected {n_predict}")

        with subject_session() as subject, spool_path.open("rb") as spool:
            _attest("subject", subject)
            if served_vocab_size(subject.base_url) != vocab_size:
                raise FullVocabError("control and subject report different vocabulary sizes")
            for step, row in enumerate(stream_completion_rows(subject.base_url, payload, timeout_s=timeout_s)):
                if step >= n_predict:
                    raise FullVocabError("subject emitted more decode steps than requested")
                token, subject_values = dense_logprobs(row, vocab_size=vocab_size, arm="subject", step=step)
                subject_digest.update(canonical_bytes(subject_values))
                subject_steps += 1
                control_values = array("d")
                try:
                    control_values.fromfile(spool, vocab_size)
                except EOFError as exc:
                    raise FullVocabError("control logprob spool ended early") from exc
                if sys.byteorder != "little":
                    control_values.byteswap()
                if first_mismatch is None and control_tokens[step] != token:
                    first_mismatch = {"step": step, "control_token_id": control_tokens[step], "subject_token_id": token}
                if first_mismatch is None or first_mismatch["step"] == step:
                    comparable += 1
                    m = step_metrics(control_values, subject_values, criterion)
                    if m.max_abs_diff > max_abs_diff:
                        max_abs_diff = m.max_abs_diff
                        worst = {"step": step, "token_id": m.worst_token, "control": control_values[m.worst_token],
                                 "subject": subject_values[m.worst_token], "abs_diff": m.max_abs_diff}
                    if m.material_diff > max_material:
                        max_material = m.material_diff
                        worst_material = {"step": step, "token_id": m.worst_material_token,
                                          "control": control_values[m.worst_material_token],
                                          "subject": subject_values[m.worst_material_token],
                                          "abs_diff": m.material_diff}
                    max_kl = max(max_kl, m.kl)
                    min_topp_mass = min(min_topp_mass, m.topp_mass)
            if subject_steps != n_predict:
                raise FullVocabError(f"subject emitted {subject_steps} steps; expected {n_predict}")
            if spool.read(1):
                raise FullVocabError("control logprob spool has trailing data")
    finally:
        spool_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()

    tokens_match = first_mismatch is None
    compared = comparable * vocab_size
    passed, verdict = judge(criterion, tokens_match=tokens_match, max_abs_diff=max_abs_diff, max_kl=max_kl,
                            max_material=max_material, min_topp_mass=min_topp_mass)
    if first_mismatch is not None:
        detail = f"generated tokens diverged at step {first_mismatch['step']} after comparing {compared} logprobs"
    else:
        detail = f"{n_predict} decode steps, {compared} full-vocabulary logprobs; {criterion.name}: {verdict}"
    return FullVocabComparison(
        passed=passed,
        detail=detail,
        vocab_size=vocab_size,
        decode_steps=n_predict,
        decode_steps_compared=comparable,
        criterion=criterion,
        generated_tokens_match=tokens_match,
        first_generated_token_mismatch=first_mismatch,
        max_abs_logprob_diff=max_abs_diff,
        worst=worst,
        max_kl=max_kl,
        max_material_logprob_diff=max_material,
        worst_material=worst_material,
        min_topp_mass=min_topp_mass if math.isfinite(min_topp_mass) else 0.0,
        control_logprobs_sha256=control_digest.hexdigest(),
        subject_logprobs_sha256=subject_digest.hexdigest(),
        request=payload,
        attestations=attestations,
    )


def judge(
    criterion: Criterion, *, tokens_match: bool, max_abs_diff: float, max_kl: float,
    max_material: float, min_topp_mass: float,
) -> tuple[bool, str]:
    """Apply a criterion to the aggregated comparison metrics."""
    if not tokens_match:
        return False, "generated tokens differ"
    if criterion.bit_identical:
        ok = max_abs_diff == 0.0
        return ok, f"max_abs_logprob_diff={max_abs_diff:.9g} ({'exact' if ok else 'not exact'})"
    failures = []
    if max_kl > criterion.max_kl:
        failures.append(f"max_kl={max_kl:.3g} > {criterion.max_kl:.3g}")
    if max_material > criterion.material_logprob_tol:
        failures.append(f"material logprob diff={max_material:.3g} > {criterion.material_logprob_tol:.3g} "
                        f"(tokens with p > {criterion.material_prob:g})")
    if min_topp_mass < criterion.min_topp_mass:
        failures.append(f"top-{criterion.topp:g} mass under subject={min_topp_mass:.4f} < {criterion.min_topp_mass:.4f}")
    summary = (f"max_kl={max_kl:.3g}, material_diff={max_material:.3g}, top-p mass>={min_topp_mass:.4f}, "
               f"all-vocab max diff={max_abs_diff:.3g} (informational)")
    return (not failures), (summary if not failures else "; ".join(failures))
