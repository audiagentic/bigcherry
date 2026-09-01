"""Publish verified BigCherry campaign builds into a live Toolchest registry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..campaign.lane import CampaignLaneResult
from ..campaign.planner import CampaignLane


class ToolchestPublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolchestRegistration:
    build_id: str
    payload: dict[str, object]
    response: dict[str, object]


_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(value: str) -> str:
    value = _ID_SAFE.sub("-", value).strip("-._")
    return value or "build"


def _nested_str(document: object, namespace: str, field: str) -> str | None:
    if not isinstance(document, dict):
        return None
    ns = document.get(namespace)
    if not isinstance(ns, dict):
        return None
    value = ns.get(field)
    return value if isinstance(value, str) and value else None


def toolchest_profile(backend: str) -> str:
    return {"hip": "rocm"}.get(backend, backend)


def registration_payload(
    lane: CampaignLane,
    result: CampaignLaneResult,
    *,
    backend: str,
) -> dict[str, object]:
    """Map one verified campaign result to Toolchest's external-build schema.

    The stable Toolchest ID carries the human lane identity plus the full
    content-addressed BuildPlan ID. Exact producer/source/runtime identities
    remain in the normal Toolchest provenance fields so benchmark snapshots
    stay self-describing after later registry changes.
    """
    primary_name = result.binary_ref.path.name
    if primary_name not in {"llama-server", "llama-server.exe"}:
        raise ToolchestPublishError(
            f"Toolchest publication requires llama-server as the primary artifact, got {primary_name!r}"
        )

    provenance = result.binary_ref.provenance
    bigcherry_revision = _nested_str(provenance, "project", "bigcherry_revision") or ""
    upstream_revision = (
        _nested_str(provenance, "source", "upstream_revision")
        or result.resolved_revision
    )
    effective_build_id = _nested_str(provenance, "build", "effective_build_id") or "unknown"
    runtime_bundle_hash = _nested_str(provenance, "build", "runtime_bundle_hash") or "unknown"

    lane_name = _safe_id(
        f"{lane.source_name}-{lane.build_name}-{lane.platform_name}"
    )
    build_id = f"{lane_name}-{result.build_plan_id}"
    tag = f"eb-{effective_build_id[:12]}-rb-{runtime_bundle_hash[:12]}"

    return {
        "id": build_id,
        "profile": toolchest_profile(backend),
        "binary_path": str(result.binary_ref.path.resolve()),
        "git_ref": upstream_revision,
        "git_sha": bigcherry_revision,
        "tag": tag,
        "replace": True,
    }


def register_build(
    base_url: str,
    payload: dict[str, object],
    *,
    timeout: float = 10.0,
    api_key: str | None = None,
) -> dict[str, object]:
    url = base_url.rstrip("/") + "/api/builds?external=1"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 -- operator-configured Toolchest endpoint
            raw = response.read()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise ToolchestPublishError(
            f"Toolchest registration failed with HTTP {exc.code}{suffix}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ToolchestPublishError(f"Toolchest registration failed: {exc}") from exc

    if not raw:
        return {}
    try:
        document: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolchestPublishError(
            "Toolchest registration returned invalid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise ToolchestPublishError("Toolchest registration returned a non-object response")
    return document


def publish_campaign_result(
    base_url: str,
    lane: CampaignLane,
    result: CampaignLaneResult,
    *,
    backend: str,
    timeout: float = 10.0,
    api_key: str | None = None,
) -> ToolchestRegistration:
    payload = registration_payload(lane, result, backend=backend)
    response = register_build(base_url, payload, timeout=timeout, api_key=api_key)
    return ToolchestRegistration(
        build_id=str(payload["id"]), payload=payload, response=response
    )
