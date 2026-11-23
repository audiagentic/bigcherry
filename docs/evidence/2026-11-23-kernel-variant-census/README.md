# Kernel variant census — binary evidence

Release: bc-radiance-2020.8.0. Collected: 2026-11-23. Architectures: gfx1100 (libr3), gfx1201 (libr4d).

**Status:** Complete compiled-symbol inventory, **not** a completed launched-symbol census. Profiling an already-running process failed with ptrace EPERM. Launch-time rocprofv3 instrumentation caused GPU device errors during kernel initialization before inference, so neither startup trace is counted as a successful inference dispatch. Two configuration workloads completed normally without GPU kernel-symbol traces. All other shipped configs were not run. No dead-code conclusion is justified.

- inventory-summary.json: exact ELF symbol totals (unique symbols vs code-object/symbol instances), sizes and identical ISA body instances.
- libr3-kernels.csv.gz and libr4d-kernels.csv.gz: complete GPU kernel inventory. Each row: GPU code-object index, mangled symbol, demangled symbol including template arguments, heuristic function-name family, function st_size and SHA-256 hash of executable ISA bytes.
- libr3-families.json and libr4d-families.json: instance and size totals by leading kernel function name (heuristic buckets, not the project's semantic family taxonomy).
- chain-comparison.json: equal template argument sets and non-identical ISA for chain/base paired kernels.
- release-provenance.json: release tag, tree keys and manifest SHA-256 for libraries.
- coverage.json: explicit per-YAML measurement status; null launched counts mean **unknown**, never zero.

ELF method: extract each library's embedded GPU objects in scratch, read the ELF64 symtab and dynsym, keep STT_FUNC entries with .kd descriptors, deduplicate within a code object by symbol, calculate st_size and function-content digest, demangle with c++filt. An identical symbol compiled in two separate GPU code objects is two instances in ISA size accounting but one distinct symbol name.

No host names, model paths, binary assets or machine-specific traces appear in this evidence directory. Kernel sources, compiled binaries and deployment code were not changed.
