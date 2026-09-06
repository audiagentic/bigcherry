# Environment helpers

Environment scripts provide shell setup only. They do not own product state, evidence, or validation.

## `bigcherry-env.sh` — host facts by role

Reads [`config/environment.toml`](../../config/environment.toml) and exports
the build host's facts as `BC_*`, so docs and scripts can say
`$BC_MODEL_ROOT` instead of hardcoding one machine's paths.

```bash
source tools/env/bigcherry-env.sh              # default host
source tools/env/bigcherry-env.sh build-server # a named host
```

The Python equivalent is `bigcherry.core.environment`, reading the SAME
config — the two must not drift, and `tools/tests/core/test_environment.py`
asserts they agree:

```python
from bigcherry.core import environment as env
host = env.load_default().host()
host.model_root, host.bench_port, host.visible_devices(0, 1)
```

See [ENVIRONMENT.md](../../docs/reference/ENVIRONMENT.md) for the convention:
documentation names a ROLE ("the build server"), the config resolves it.

## Other scripts

`rocm-env.sh` / `rocm-env.ps1` are ROCm toolchain setup and predate the
role-based config. Existing environment scripts remain at their current paths
until the dedicated cleanup phase; this directory is the canonical destination.
