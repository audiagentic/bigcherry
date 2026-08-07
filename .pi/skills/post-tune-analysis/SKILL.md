---
name: post-tune-analysis
description: Load tuning measurements into SQLite on brutus, run reports (summary, signatures, families, hot), and query the results. Use when the user has a .measurements.jsonl file from a HIP autotune run and wants to understand which kernels won or analyze performance patterns.
---

# Post-tune measurement analysis

This skill walks through loading HIP autotune measurement results into SQLite for structured analysis and reporting on brutus.

## Prerequisites

- A `.measurements.jsonl` file from a tuning run (produced when `GGML_HIP_DISPATCH_MODE=tune` and `GGML_HIP_DISPATCH_DB=<path>`)
- Optional: manifest JSON at `artifacts/<revision>/hip-autotune-manifest.json` for full candidate data
- Optional: observation JSONL from record mode (`inventory record`) for the `hot` report

## Step 1 — Load measurements into SQLite

```bash
cd /mnt/vault/development/llmhosts/bigcherry
python3 -m bigcherry inventory tuning <measurements.jsonl> \
    --database <output.sqlite> \
    [--manifest <manifest.json>]
```

This reads the JSONL and writes three tables: `winner`, `measurement`, and `candidate`. The `--manifest` flag populates full candidate data; without it, candidates come from measurement rows only.

The output goes into `<measurements>.sqlite` by default (same directory, `.sqlite` extension).

## Step 2 — Quick summary

```bash
# Aggregate statistics
python3 -m bigcherry report summary --database <output.sqlite>

# JSON output for scripting
python3 -m bigcherry report summary --database <output.sqlite> --json
```

This shows: total signatures, how many had tuned winners vs native, average improvement, family breakdown of winners. Use this first to get oriented.

## Step 3 — Signature-level detail

```bash
# All signatures with winner info
python3 -m bigcherry report signatures --database <output.sqlite>

# Limit to first N results
python3 -m bigcherry report signatures --database <output.sqlite> --limit 50

# Specific dispatch (use hex digest from summary or another report)
python3 -m bigcherry report families --database <output.sqlite> \
    --dispatch abc123...
```

The `families` subcommand breaks down candidates by family for a given dispatch, showing which family won and by how much. This is useful for understanding cross-family wins.

## Step 4 — Hot signatures (performance impact)

Merge observation data into the same database first:

```bash
python3 -m bigcherry inventory record <observations.jsonl> \
    --database <output.sqlite>

# Then query hot paths
python3 -m bigcherry report hot --database <output.sqlite>
python3 -m bigcherry report hot --database <output.sqlite> --limit 10
```

This ranks signatures by call count, showing which wins matter most for real throughput. The `hot` command joins observation (call frequency) with winner data.

## Step 5 — Custom SQL queries

When you need analysis beyond the built-in reports:

```python
import sqlite3
conn = sqlite3.connect("tune.sqlite")
conn.row_factory = sqlite3.Row

# Top 10 improvements by percentage
r = conn.execute("""SELECT w.stable_name, w.improvement_pct, w.reason,
                    m.dispatch_digest
                    FROM winner w JOIN measurement m ON w.signature_id = m.signature_id
                    WHERE w.improvement_pct > 0
                    ORDER BY w.improvement_pct DESC LIMIT 10""").fetchall()

# Family breakdown of winners
r = conn.execute("""SELECT c.family, COUNT(*) as wins
                    FROM winner w JOIN candidate c ON w.candidate_id = c.candidate_id
                    GROUP BY c.family ORDER BY wins DESC""").fetchall()

# Candidates that failed correctness
r = conn.execute("""SELECT c.stable_name, m.reject_reason, m.dispatch_digest
                    FROM measurement m JOIN candidate c ON m.candidate_id = c.candidate_id
                    WHERE m.accepted = 0""").fetchall()

conn.close()
```

Key schema facts:

- `winner` table: one row per dispatch digest (best candidate)
- `measurement` table: one row per (dispatch, candidate) pair tested
- `candidate` table: catalog of all candidates with family, workspace, etc.
- `observation` table: call counts from record mode (only if merged in)
- **Dispatch digest is stored as BLOB** — compare with `X'hexstring'` in SQL or `.hex()` in Python

## Step 6 — Generate variant sets (optional)

If you want to use the tuning results to generate a focused catalog:

```bash
python3 -m bigcherry generate --variant-set replay-slim \
    --winners <measurements.jsonl> \
    --arch gfx1100
```

This generates a candidate catalog containing only winners (plus native fallbacks), producing a much smaller binary. Order matters: generate slim first, then export the cache with `replay_cache`.

## Gotchas

1. **Manifest vs no-manifest** — Without `--manifest`, the candidate table has minimal data (just what's in measurements). With it, you get full candidate metadata from the catalog. For thorough analysis, always include the manifest.

2. **Record mode DB is separate** — The observation database from record mode (`inventory record`) and the tuning database are different schemas merged into one. Load observations first with `record`, then overlay tuning with `tuning`.

3. **Dispatch digest is stored as BLOB** — In SQLite, `dispatch_digest` is a BLOB column. Compare with `X'hexstring'` in SQL or `.hex()` in Python.

4. **JSONL only at runtime** — Production builds write JSONL, not SQLite (Standards 9.1). The SQLite conversion is always offline/development-time. Don't assume the runtime writes SQLite directly.
