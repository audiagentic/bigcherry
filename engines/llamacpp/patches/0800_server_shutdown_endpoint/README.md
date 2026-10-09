# 0800_server_shutdown_endpoint framework validation

Adds an opt-in HTTP shutdown endpoint (`POST /shutdown`, disabled by default,
enabled via `LLAMA_SERVER_ENABLE_SHUTDOWN`) that returns 202 and then enters
the server's normal termination path. Needed by Windows benchmark harnesses,
where terminating the process directly skips backend destruction and loses
buffered HIP autotune measurements.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove the endpoint actually accepts a shutdown request or performs
a clean teardown at runtime.

## Upstream / provenance

Local design, part of this project's own tooling/automation support.
