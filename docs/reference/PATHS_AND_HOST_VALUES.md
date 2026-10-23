# Paths and host-specific values

Status: standard. Enforced in part by `TR14.HOST_SPECIFIC_VALUE` and `TR14.FIXED_PARENT_DEPTH`.

## Why this exists

Logs, result records and evidence are read by people outside the machine that produced them.
Host names, addresses and absolute paths there reveal the lab layout and add nothing to the
result. Committed code also breaks on any other host if it hardcodes those values.

## The four places a path can appear

| Where | Rule | Example |
|---|---|---|
| Committed source and config | Repo-relative, or a `${VAR}` reference resolved from the environment | `config/models.toml`, `${ROCM_PATH}/bin/hipcc` |
| Host facts | Untracked `config/environment.local.toml`; override any key with `BIGCHERRY_HOST_<KEY>` | `BIGCHERRY_HOST_CORPUS=/data/corpus/kld.txt` |
| Console and log output | Logical name, not the resolved path | `${MODELS_ROOT}/qwen/…`, `<repo>/artifacts/…` |
| Evidence and result records | Run identifiers and results only. No host names, addresses or absolute paths | `run_id`, `schedule`, measured values |

## Rules

1. **No hardcoded defaults for host values.** A default that names a machine path is an error.
   Use `None` and read the value from the environment or the local config, failing with a message
   that names the variable to set.
2. **Find the repository root by marker, not by depth.** Search upward for `AGENTS.md` and
   `tools/bigcherry`. Do not use `Path(__file__).parents[N]`.
3. **Environment overrides use an explicit prefix.** `BC_PRIMARY_ROOT` is the existing example.
   Host keys use `BIGCHERRY_HOST_<KEY>`.
4. **Output shows the logical location.** Print `${VAR}/…` or a repo-relative path. Use the resolved
   absolute path only where the tool needs it to run, and keep it out of anything that is saved
   or shared.
5. **Evidence is about results.** Do not record where a run executed. If a reader needs to know the
   machine class, record a label such as `gpu-host-a`, not a name or address.
6. **Redact existing evidence rather than deleting it.** Replace a value with a placeholder that
   keeps its meaning (`${BIGCHERRY_ROOT}`, `<gpu-host>`, `<user>`, `<email>`). Keep the result data
   unchanged. Regenerate `SHA256SUMS` for the redacted files, and note in the commit that the
   checksums now refer to redacted content.

## Placeholder vocabulary

| Placeholder | Stands for |
|---|---|
| `${BIGCHERRY_ROOT}` | the repository checkout |
| `${DEV_ROOT}` | the development tree that holds the checkouts |
| `${MODELS_ROOT}` | the model store |
| `${DATA_ROOT}`, `${VAULT_ROOT}` | the mounted data and vault volumes |
| `${LLAMACPP_ROOT}` | the llama.cpp tree |
| `${HOME}` | the user's home directory |
| `${GPU_HOST_IP}` | the GPU host's address |
| `<gpu-host>` | the GPU host's name, in prose |
| `<user>`, `<email>`, `<win-root>` | a user name, an email address, a Windows drive root |

## Checking

`bigcherry check --quick` runs the TR14 checks. Run it before committing. A new host value
in tracked files fails the check and names the file and line.

## Enforcement

- **Machine names** are listed under `[forbidden] terms` in `config/environment.local.toml`
  (untracked), or in `BIGCHERRY_FORBIDDEN_TERMS`. They never go in a tracked file, so the list
  cannot leak the names it guards. `config/environment.example.toml` shows the shape with a
  placeholder.
- **Generic patterns** (absolute mount and home paths, drive-letter paths, private addresses,
  email addresses) are fixed in `tools/bigcherry/forbidden_terms.py`. Reserved example domains
  and version numbers are not reported.
- **Allowlist:** `config/forbidden-terms-allow.txt` lists technical phrases that contain a
  forbidden term by accident, such as `integrated_gpu_host_buffer`. Add a phrase there only when it
  is a real technical name; never to hide a host reference.
- **Pre-commit hook:** `.githooks/pre-commit` scans the staged index and blocks a commit that adds
  a finding. Install it once per clone with `python3 -m bigcherry.forbidden_terms --install-hook`.
  Each staged file is scanned in full, so a commit that touches a file with backlog findings is
  blocked until those findings are fixed. Clear a file's backlog when you change it.
- **Checker:** `TR14.FORBIDDEN_TERM` reports the backlog as warnings, one line per file.
  Run `python3 -m bigcherry.forbidden_terms --all` for the full list of matches.
