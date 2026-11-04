# Build and use BigCherry

Radiance is BigCherry's primary target for mixed-generation AMD GPUs.
llama.cpp remains an active secondary engine. See the
[Radiance overview](docs/reference/engines/radiance/README.md) and
[llama.cpp overview](docs/reference/engines/llamacpp/README.md).

The examples use the `bigcherry` CLI from the repository root
(`PYTHONPATH=tools`). Build lanes and patch selections are declared in
`engines/<engine>/recipes.toml`.

## Radiance

```bash
PYTHONPATH=tools python -m bigcherry build \
  --lane radiance-lab:stock:linux-radiance
```

Radiance builds its engine and kernel libraries; BigCherry installs its
additional components into the runtime bundle. The
[Radiance runbook](docs/reference/engines/radiance/RUNBOOK.md) covers build
lanes and A/B measurement. The [component catalogue](docs/reference/engines/radiance/ENHANCEMENTS.md)
links to implementation and evidence for each extension.

The strict C++ serving loader and engine manifest entry point are present.
Helper manifest consumption and process supervision are still being
implemented. The [serving component](engines/radiance/components/serving/README.md)
describes the implemented interface.

## llama.cpp

```bash
PYTHONPATH=tools python -m bigcherry --engine llamacpp patches
PYTHONPATH=tools python -m bigcherry --engine llamacpp pull --ref <upstream-revision>
PYTHONPATH=tools python -m bigcherry --engine llamacpp audit
PYTHONPATH=tools python -m bigcherry apply --source bigcherry
PYTHONPATH=tools python -m bigcherry generate --arch all
```

`pull` selects an upstream revision; `audit` checks the unpatched tree
before the overlay and anchored patches are applied. `generate` produces
measured-dispatch candidate artefacts. See the
[pin and rebase reference](docs/reference/build/PIN_BUMP.md).

Builds select a source, build profile and platform:

```bash
PYTHONPATH=tools python -m bigcherry build \
  --lane <source>:<build>:<platform> \
  --binary-relative-path bin/llama-server
```

Record and tune builds use a candidate inventory. Replay builds also need
a winners cache. The selected recipe specifies those requirements.
See the [build reference](docs/reference/build/BUILD.md) and
[patch catalogue](docs/reference/engines/llamacpp/PATCHES.md).

### HIP dispatch

| Build mode | Input | Behaviour |
| --- | --- | --- |
| Native | — | Upstream kernel selection |
| Record | `GGML_HIP_DISPATCH_DB` | Records operation signatures |
| Tune | `GGML_HIP_DISPATCH_DB` | Measures eligible kernels |
| Replay | `GGML_HIP_DISPATCH_CACHE` | Uses promoted winners; falls back to native on a miss |

Record, tune and replay are separate compile-time modes, not runtime switches.
Tune mode needs additional GPU memory for measurements. Record and tune data
is flushed on clean process exit.

## Validation

```bash
PYTHONPATH=tools python -m bigcherry doctor
PYTHONPATH=tools python -m bigcherry check --default
python -m pytest tools/tests -q
```

These checks cover repository and tooling behaviour; kernel performance
and numerical quality require evidence from the model, workload and device
under test. See [patch validation](docs/reference/testing/PATCH_VALIDATION.md)
and the [experiment contract](docs/reference/experiments/EXPERIMENT_CONTRACT.md).
