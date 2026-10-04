# 0910_feature_sets

**Status:** validated
**Plan item:** QFP18
Kind: framework (no behaviour change unless `BIGCHERRY_FEATURES` is set)

## What it does

- `BIGCHERRY_FEATURES=<set>[,<set>...]` enables a named profile of runtime switches with one flag. A load-time
  constructor in ggml.c expands each set before any backend or model code reads its flags. Explicitly set member
  variables win, so dev A/B runs can still turn one member off (e.g. `BIGCHERRY_ACT_Q81=0`). The expansion is logged
  once to stderr (`BIGCHERRY_FEATURES <set>: NAME=VALUE ...`) as activation evidence.
- `BIGCHERRY_FEATURES=help` (or `list`) prints every feature set and every runtime flag documented by the patches in
  this build (name, values, default, owning patch, description), then exits:

      BIGCHERRY_FEATURES=help ./llama-server

## Sets

Sets are named by what they act on, not by the model they were tuned on. A member is `NAME=VALUE` or `@other-set`.

| Set | Members | Scope |
|---|---|---|
| `hip-q81` | `GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1 BIGCHERRY_SCALE_ACT_FUSE=1` | HIP MMVQ decode, any quantized model (HC pre-mix only acts on hyper-connection models) |
| `sched-async` | `BIGCHERRY_SCHED_ASYNC_INPUTS=1` | scheduler, multi-backend / tensor-split runs |
| `flashnext` | `@hip-q81 @sched-async` | Qwen3.8 Flash-Next production profile; its Qwen4Exp-only patches (1308, 1327) are on by default |

Placement settings with real values (`BIGCHERRY_ATTN_TS`, `BIGCHERRY_DRAFT_VOCAB_N`) stay separate flags.

## Documenting a flag (patch authors)

A patch that reads a runtime flag documents it in its own patch.py and declares `requires = ["0910_feature_sets"]`:

```python
from bigcherry.patcher import EnvDoc
ENV_DOCS = (
    EnvDoc("BIGCHERRY_ACT_Q81", "0|1", "0", "activation ops write the Q8_1 activation directly"),
)
```

The patch loader (`registry.load_implementation`) turns `ENV_DOCS` into rows of this patch's help table, so they are
compiled into the binary only when the patch is in the build and the help reflects the actual build. The loader fails
closed when `ENV_DOCS` is present without the `0910_feature_sets` requirement. Per-file mechanics tests and rebase
checks see only the patch's own edits.
To add a profile, add one row to `bc_feature_sets` in patch.py.
