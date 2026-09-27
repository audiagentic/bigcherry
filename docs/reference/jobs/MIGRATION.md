# Legacy lab-queue migration

Migration creates new canonical job requests. It never imports shell-run success into managed history and never infers scientific planned-N from results already observed.

## Supported converter

For one historical `run_campaign.sh` invocation:

```bash
python -m bigcherry jobs migrate-legacy \
  --command 'tools/lab/plan-qualification/run_campaign.sh PATCH PATCH/producer gfx1100 0 RUN ...' \
  --planned-sessions 4 \
  --bc-model /models/model.gguf \
  --bc-hip-path /opt/rocm \
  --executor-id brutus --host-id brutus --platform-family linux-rocm
```

Output schema: `bigcherry.jobs.legacy-migration.v1`.

The converter accepts only the exact wrapper shape plus arguments with direct JobSpec equivalents:

```text
--baseline-source
--common-patches
--validation-producer
--producer-input NAME=VALUE
--producer-corpus
--production-lane
```

Legacy `--device-map`, `--workdir`, `--worktree-root`, and `--build-root` are discarded because managed execution owns placement/workspaces. The positional legacy GPU device and run-name are emitted only as migration metadata; neither becomes scientific device identity.

Unknown legacy flags fail closed and require explicit human/agent conversion. `BC_MODEL` and `BC_HIP_PATH` must be supplied from the original environment. `--planned-sessions` is mandatory.

## Migration workflow

1. Preserve old queue/log directories read-only.
2. Convert each pending logical campaign into a BatchSpec.
3. Review the output, especially producer inputs, common patches, production lane, target/toolchain and planned N.
4. Save the `batch` object as JSON/TOML.
5. Run `jobs validate` and `jobs acceptance` against current accepted inventory.
6. Submit with a new idempotency key.
7. Do not manufacture managed attempts/evidence from old `CAMPAIGN_EXIT=` lines.

If a legacy invocation used an unsupported one-off switch/flag, encode the real scientific requirement in JobSpec/domain code first; do not add arbitrary passthrough shell arguments to JobSpec.
