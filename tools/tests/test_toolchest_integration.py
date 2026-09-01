from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from bigcherry.integrations.toolchest import (
    ToolchestPublishError,
    publish_campaign_result,
    register_build,
    registration_payload,
    toolchest_profile,
)


def _lane():
    return SimpleNamespace(
        source_name="bigcherry-native",
        build_name="control",
        platform_name="linux-multi",
    )


def _result(tmp_path: Path):
    server = tmp_path / "llama-server"
    server.write_text("server", encoding="utf-8")
    return SimpleNamespace(
        binary_ref=SimpleNamespace(
            path=server,
            provenance={
                "project": {"bigcherry_revision": "bc1234567890abcdef"},
                "source": {"upstream_revision": "upstream987654321"},
                "build": {
                    "effective_build_id": "effective1234567890",
                    "runtime_bundle_hash": "runtime1234567890abcdef",
                },
            },
        ),
        resolved_revision="fallback-upstream",
        build_plan_id="0123456789abcdef0123456789abcdef",
    )


def test_registration_payload_preserves_verified_identities(tmp_path: Path) -> None:
    payload = registration_payload(_lane(), _result(tmp_path), backend="hip")
    assert payload == {
        "id": "bigcherry-native-control-linux-multi-0123456789abcdef0123456789abcdef",
        "profile": "rocm",
        "binary_path": str((tmp_path / "llama-server").resolve()),
        "git_ref": "upstream987654321",
        "git_sha": "bc1234567890abcdef",
        "tag": "eb-effective123-rb-runtime12345",
        "replace": True,
    }


def test_profile_mapping() -> None:
    assert toolchest_profile("hip") == "rocm"
    assert toolchest_profile("vulkan") == "vulkan"


def test_registration_rejects_non_server_primary(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result.binary_ref.path = tmp_path / "llama-bench"
    result.binary_ref.path.write_text("bench", encoding="utf-8")
    with pytest.raises(ToolchestPublishError, match="requires llama-server"):
        registration_payload(_lane(), result, backend="hip")


class _Handler(BaseHTTPRequestHandler):
    status = 201
    response = {"id": "registered"}
    received_path = ""
    received_body: dict[str, object] | None = None

    def do_POST(self):  # noqa: N802 - stdlib handler API
        type(self).received_path = self.path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        type(self).received_body = json.loads(raw)
        body = json.dumps(type(self).response).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


def _server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_publish_campaign_result_posts_live_external_endpoint(tmp_path: Path) -> None:
    _Handler.status = 201
    _Handler.response = {"id": "registered"}
    server, thread = _server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        registration = publish_campaign_result(
            base, _lane(), _result(tmp_path), backend="hip"
        )
        assert _Handler.received_path == "/api/builds?external=1"
        assert _Handler.received_body == registration.payload
        assert registration.response == {"id": "registered"}
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_http_error_surfaces_toolchest_response() -> None:
    _Handler.status = 400
    _Handler.response = {"error": "bad build"}
    server, thread = _server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(ToolchestPublishError, match="HTTP 400"):
            register_build(base, {"id": "bad"})
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
