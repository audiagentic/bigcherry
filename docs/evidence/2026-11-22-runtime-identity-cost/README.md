# Runtime identity cost on a reused native start

Question: how much of a start that reuses a prior measured run is spent hashing the installed runtime, and is it worth
changing. Host-only, CPU-only, no GPU.

## Measurement

Subject: an installed 2020.0.1 radiance release (the engine binary plus its runtime home of plugins, kernels, helpers,
quantizers, vectors and architectures), read from a local disk with the files already in the page cache. The timing
program links the serving component's own `Sha256` and reads each file in 64 KiB chunks exactly as
`engine_identity` does, so the figure is the component's cost, not a system tool's.

| Quantity | Value |
|---|---|
| Runtime home | 409,331,369 bytes in 65 files |
| Engine binary | 19,162,936 bytes |
| Hashed per call | 428,494,305 bytes |
| One `engine_identity` call, three runs | 1186, 1176, 1211 ms (about 1.18 s, about 360 MB/s) |
| Same bytes with the system sha256sum (hardware SHA) | about 225 ms |

The component's SHA-256 is a portable implementation; the cost is its throughput, not I/O (warm cache).

## Calls per start (counted from `session.cpp`)

Native admission (`plan_native_v3`): once before the YAML inspection, once after it (the "changed during YAML
inspection" refusal), once per measuring round for the plan key, and, when the start measures or re-seals a carried
measurement, once after the round (the "changed during measurement" key check) and once at the final frozen
build check. That is 3 calls when nothing is measured and 5 when a carried measurement is re-sealed or a start measures.
Stored-manifest replay: one for the replay build comparison, one for the plan key, one after native admission: 3 calls.
Each start also hashes the running `bc-serve` binary once (the producer digest), a single file.

Total per start before the change: 3 to 5 calls, about 3.5 to 5.9 s of CPU on one core, for a start whose measuring is
being skipped. That is material against a start that is meant to be fast.

## Decision

Change it: hash the runtime once per process while no file moves. Hashing once per start would have made the
"changed during inspection / measurement" refusals compare a value with itself and so never fire. Instead
`engine_identity` memoises its digest against a stamp of every file (size, inode, modification time, change time),
taken before the bytes are read. The change time is set by the kernel on every write and cannot be set back by a
caller, so an unchanged stamp means unchanged content, and a file that moves, is added, removed or restored is hashed
again. Every comparison and refusal in `session.cpp` stays exactly as it was; a start that does not see the runtime
change pays one hash (about 1.2 s) and the later checks pay a stat of 66 files (well under a millisecond).

Provenance and cross-build re-sealing are unchanged: the identity value is the same digest as before, so a stored plan
from another build still compares unequal and is re-sealed or refused as before.

Tests: `engine_identity_test.cpp` (target `bc_engine_identity_test`) proves the runtime is hashed once across repeated
calls, hashed again when a kernel, the binary or the file set changes, restored bytes give the original identity, and
a missing binary still refuses.

## Follow-ups not done

- The release receipt of file digests: reusing it would make even the first call free, but only if the receipt is
  itself authenticated and checked against the files by stat. It would protect against a different build installed
  over the same path; it does not protect against edited files unless every file is stamped at install time. Needs the
  install step to record stamps; left as a follow-up.
- Hardware SHA-256: considered, rejected. Measured on the same 428,494,305 bytes: portable 1151 and 1122 ms, SHA-NI
  182 and 178 ms, identical digests. CPU-specific hash code is not wanted, and the better answer is that a start should
  not hash the runtime at all: `engine_identity` now persists its digest under the work root (`--work-root`,
  `cache/engine-identity/`), keyed by the stamp of every runtime file and the binary, so a start of an unchanged
  runtime reads one small file and a stat of each file.
- The one `bc-serve` producer digest per start is a single file and was left alone.

## Host pool change bound

`kMaxHostPoolChanges` is a protocol-level count, not a resource budget, and was kept with its reason written at the
definition. The native path's seen-pools set stops a cycle but not a pool that keeps moving to new values, and each
change is a full measuring run; the legacy planning loop has no seen-pools set at all. So the cap is not redundant.
