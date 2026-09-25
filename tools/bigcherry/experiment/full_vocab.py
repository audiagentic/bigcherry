"""Full-vocabulary logprob comparison between two llama-server arms.

A fixed temperature-0 streaming /completion request is sent to a control
server and then a subject server. Every generated step returns the logprob
of every vocabulary token; the control's rows are spooled to disk as
little-endian f64 so memory stays flat, and the subject is compared step by
step. The arms agree when they generate the same tokens and every compared
logprob is within ``tolerance`` (``0.0`` means bit-identical).

After the first generated-token mismatch only that step is still comparable
(its input context was identical); later steps are not the same experiment.

Callers supply the server sessions (``AttestedServerSession`` instances or
anything with a ``base_url``); this module never starts a process.
"""

from __future__ import annotations

import contextlib
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
    tolerance: float
    generated_tokens_match: bool
    first_generated_token_mismatch: dict[str, int] | None
    max_abs_logprob_diff: float
    worst: dict[str, float | int] | None
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
            "tolerance": self.tolerance,
            "generated_tokens_match": self.generated_tokens_match,
            "first_generated_token_mismatch": self.first_generated_token_mismatch,
            "max_abs_logprob_diff": self.max_abs_logprob_diff,
            "worst": self.worst,
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
                # in one verify step as one event, so an event carries >= 1 rows.
                if not isinstance(rows, list) or not rows or not all(isinstance(r, Mapping) for r in rows):
                    raise FullVocabError("expected one or more completion_probabilities rows per event")
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


def compare_servers(
    *,
    control_session: SessionFactory,
    subject_session: SessionFactory,
    prompt: str,
    n_predict: int,
    tolerance: float,
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
                    for token_id, (c, s) in enumerate(zip(control_values, subject_values, strict=True)):
                        delta = abs(s - c)
                        if delta > max_abs_diff:
                            max_abs_diff = delta
                            worst = {"step": step, "token_id": token_id, "control": c, "subject": s, "abs_diff": delta}
            if subject_steps != n_predict:
                raise FullVocabError(f"subject emitted {subject_steps} steps; expected {n_predict}")
            if spool.read(1):
                raise FullVocabError("control logprob spool has trailing data")
    finally:
        spool_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()

    tokens_match = first_mismatch is None
    passed = tokens_match and max_abs_diff <= tolerance
    compared = comparable * vocab_size
    if first_mismatch is not None:
        detail = f"generated tokens diverged at step {first_mismatch['step']} after comparing {compared} logprobs"
    else:
        relation = "<=" if passed else ">"
        detail = (
            f"{n_predict} decode steps, {compared} full-vocabulary logprobs; "
            f"max_abs_logprob_diff={max_abs_diff:.9g} {relation} tolerance={tolerance:.9g}"
        )
    return FullVocabComparison(
        passed=passed,
        detail=detail,
        vocab_size=vocab_size,
        decode_steps=n_predict,
        decode_steps_compared=comparable,
        tolerance=tolerance,
        generated_tokens_match=tokens_match,
        first_generated_token_mismatch=first_mismatch,
        max_abs_logprob_diff=max_abs_diff,
        worst=worst,
        control_logprobs_sha256=control_digest.hexdigest(),
        subject_logprobs_sha256=subject_digest.hexdigest(),
        request=payload,
        attestations=attestations,
    )
