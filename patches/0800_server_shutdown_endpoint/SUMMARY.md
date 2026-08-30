# 0800_server_shutdown_endpoint: Opt-in HTTP shutdown endpoint for graceful automation cleanup

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds a POST /shutdown endpoint, disabled by default and enabled via LLAMA_SERVER_ENABLE_SHUTDOWN, that returns 202 and then enters the server's normal termination path.

## Why

Windows benchmark harnesses that terminate the server process directly skip backend destruction and lose buffered HIP autotune measurements; a graceful shutdown path avoids that loss.

## Upstream / provenance

Local design, part of this project's own tooling/automation support.
