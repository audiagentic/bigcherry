"""Opt-in HTTP shutdown endpoint for graceful automation cleanup.

The endpoint is intentionally disabled by default. When
``LLAMA_SERVER_ENABLE_SHUTDOWN`` is set, a POST to ``/shutdown`` returns a
202 response and then enters the server's normal termination path. This is
needed by Windows benchmark harnesses, where terminating the process skips
backend destruction and loses buffered HIP autotune measurements.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch  # type: ignore[import-not-found]


SERVER_PATCH = FilePatch(
    path="tools/server/server.cpp",
    description="add an opt-in graceful HTTP shutdown endpoint",
    edits=(
        Edit(
            id="server-shutdown-cstdlib",
            anchor=r"^#include <clocale>$",
            rationale="standard-library include block",
            text="\n#include <cstdlib>",
            guard=r"^#include <cstdlib>$",
        ),
        Edit(
            id="server-shutdown-response",
            anchor=(
                r"^void llama_server_terminate\(\);\n"
                r"void llama_server_terminate\(\) \{\n"
                r"    if \(shutdown_handler\) \{\n"
                r"        shutdown_handler\(0\);\n"
                r"    \}\n"
                r"\}"
            ),
            rationale="the public server termination helper",
            text=(
                "\n\nstruct shutdown_response final : server_http_res {\n"
                "    void on_complete() override {\n"
                "        llama_server_terminate();\n"
                "    }\n"
                "};"
            ),
            guard=r"^struct shutdown_response final",
        ),
        Edit(
            id="server-shutdown-route",
            anchor=r"^    ctx_http\.get \([^;]*routes\.get_health\)\);",
            expect_matches=2,
            occurrence=1,
            rationale="the second health route registration before authenticated API routes",
            text=(
                "\n\n    // Optional local automation endpoint. It is opt-in because shutdown is a\n"
                "    // destructive operation; API-key middleware still protects it when keys\n"
                "    // are configured. The completion callback runs after the response is sent,\n"
                "    // allowing the normal server cleanup path to flush HIP autotune state.\n"
                '    if (std::getenv("LLAMA_SERVER_ENABLE_SHUTDOWN") != nullptr) {\n'
                '        ctx_http.post("/shutdown", ex_wrapper([](const server_http_req &) {\n'
                "            auto res = std::make_unique<shutdown_response>();\n"
                "            res->status = 202;\n"
                "            res->data = safe_json_to_str(json {\n"
                '                {"status", "shutting_down"}\n'
                "            });\n"
                "            return res;\n"
                "        }));\n"
                "    }"
            ),
            guard=r"LLAMA_SERVER_ENABLE_SHUTDOWN",
        ),
    ),
)


PATCHES = [SERVER_PATCH]
