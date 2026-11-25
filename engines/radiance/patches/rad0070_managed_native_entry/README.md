# rad0070_managed_native_entry

The attachment points of the managed native entry. See SUMMARY.md and `components/serving/bc_serving_managed.h`.

| Anchor | Why there |
|---|---|
| `core/main.cpp` before `if (check)` | after the YAML parse and the unsupported-key refusal, before Fleet or the offline Engine read the Config |
| `core/fleet/fleet.cpp` before `cfg.devices = chosen;` | the one place Fleet fixes a load's devices, before the Engine is constructed |
| `Fleet::reload()` | both SIGHUP and `/admin/reload` reach it |
| `POST /admin/models/:name/:action`, `POST /admin/reload` | the model-mutating routes answer 403 |
| `Fleet::reap()` and the return of `Fleet::run()` | an engine that stops ends the process with a failing exit status; nothing demands it back |

The state is process-global and set once (`bc_serving_managed()`), by design.
