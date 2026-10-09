# 0910_feature_sets

**Status:** validated
**Plan item:** QFP18/QFP23

Kind: framework (no behaviour change unless `BIGCHERRY_FEATURES` is set).

## What it does

- **Runtime profiles from config files.** Model- and scope-specific runtime settings live in `profile/*.ini`, not in
  patch code. The canonical copy is the source overlay's `src/profile/`; the build copies the folder next to the
  binaries (`bin/profile/`, part of the runtime bundle hash) and installs it with them. The loader finds it relative to
  `libggml-base`; `BIGCHERRY_PROFILES=<folder|file>` overrides.
- `BIGCHERRY_FEATURES=<profile>[,<profile>...]` applies profiles. A profile lists `NAME = VALUE` flags and may include
  other profiles (`@name`). Explicit environment variables win. Application is all-or-nothing: unknown/duplicate
  profiles, include cycles, conflicting assignments and malformed lines apply nothing; a failed set rolls back.
- `BIGCHERRY_FEATURES=auto` picks the profile by model architecture: when the first model loads
  (`llama_model_create`, before its hyperparameters and tensors), the profile whose `arch =` list contains the GGUF
  architecture is applied. Flags read before model load keep their values.
- `BIGCHERRY_FEATURES=help` prints the loaded profiles and every runtime flag documented by the patches in this build
  (name, values, default, owning patch, description). The library never exits; llama-server maps help to exit 0 and
  errors to exit 2.
- `ggml_bigcherry_features_init()` is idempotent and runs before any flag is read: `ggml_init`, `get_reg()` before
  the backend registry is constructed, `ggml_backend_load_all_from_path` before backends are dlopened, llama-server's
  `main`, and a GCC/Clang load-time constructor. Works on MSVC through the explicit hooks.

## Profiles shipped (src/profile/)

| File | Profiles |
|---|---|
| `base.ini` | `hip-q81` (HIP MMVQ Q8_1 activation path, any quantized model), `sched-async` (scheduler, multi-backend runs) |
| `flashnext.ini` | `flashnext` (Qwen3.8 Flash-Next on 2x XTX + R9700 + 6900 drafter: `@hip-q81 @sched-async` + placement) |

Tuned per-model values go into the model's profile file with the evidence run in a comment.

## Documenting a flag (patch authors)

A patch that reads a runtime flag documents it in its own patch.py and declares `requires = ["0910_feature_sets"]`:

```python
from bigcherry.patcher import EnvDoc
ENV_DOCS = (
    EnvDoc("BIGCHERRY_ACT_Q81", "0|1", "0", "activation ops write the Q8_1 activation directly"),
)
```

The patch loader turns `ENV_DOCS` into rows of this patch's help table, so the help reflects the actual build.
`patch-lint` validates the profile files (grammar, duplicates, unknown includes, cycles, conflicts) and rejects
flags that no patch documents.
