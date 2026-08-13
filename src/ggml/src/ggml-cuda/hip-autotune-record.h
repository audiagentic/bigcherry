// bigcherry: signature collection, record mode (HI10).
//
// Record mode observes and changes nothing. It runs a workload with upstream's
// own selection and writes down every distinct operation it saw: the canonical
// signature, the native candidate, the call count, and where it came from. That
// file is the input to `generate --variant-set workload-max`, which is why the
// inventory build exists at all (standards 6.2).
//
// **Output is JSONL, not SQLite.** The plan specified a SQLite writer, and
// `sql/dispatch-db.sql` is still the schema of record -- but it is populated
// offline by a Python tool rather than inline by the backend. Three reasons:
//
//   * no build dependency. libsqlite3-dev is not installed on either machine
//     and the tuning host has no passwordless sudo, so an in-process SQLite
//     writer would mean vendoring an amalgamation to buy nothing.
//   * crash safety. A tuning run killed at hour three keeps every observation
//     already flushed; a half-committed transaction does not.
//   * standards 9.1 becomes trivially true -- production cannot link SQLite
//     because nothing does.
//
// The cost is that the runtime cannot query "have I already measured this?".
// It keeps that in memory for the run, which is where it wants it anyway.

#pragma once

#include "hip-autotune-types.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_AUTOTUNE_RECORD)

// Observe one operation. Cheap on the warm path: the first sighting of a
// signature allocates and records, later sightings increment a counter under a
// lock that is only ever contended by the handful of host threads driving
// devices.
//
// Standards 15.1: a repeated signature increments `calls` and merges its site
// rather than appending a second record. Diagnostics never reach the digest.
void ggml_hip_record_observation(
    ggml_backend_cuda_context & ctx,
    const ggml_hip_dispatch_signature_v1 & sig,
    const ggml_hip_hardware_key_v1 & hw,
    const ggml_hip_digest & signature_digest,
    const ggml_hip_digest & hardware_digest,
    const ggml_hip_native_selection & native,
    const char * effective_api,
    size_t workspace_bytes);

// Count a repeat encounter of an already-recorded signature.
//
// The resolver's process cache (standards 15.2) returns on the warm path
// before reaching `ggml_hip_record_observation`, so without this every
// signature would report `calls == 1` no matter how often it ran -- and the
// hot-signature ranking that drives tuning priority (standards 7.4) would be
// ordering by nothing at all. A 27B run that executed ~17,000 launches
// reported 17 calls before this existed.
//
// It also matters for multi-GPU: two identical GPUs share a dispatch key
// (standards 10.2), so the second device *always* takes the warm path and
// would otherwise never appear in the device list at all.
//
// Cheap by construction: the digest has already been computed for the cache
// lookup, so this is one hash-map probe and two increments.
void ggml_hip_record_touch(const ggml_hip_digest & signature_digest,
                           const ggml_hip_digest & hardware_digest,
                           int device);

// Record-only effective execution telemetry, called immediately before the
// selected upstream BLAS API. It never reaches dispatch identity.
void ggml_hip_record_effective_call_api(const char * api);

// BLAS-0 observation diagnostics. These fields describe the effective native
// call and never participate in signature, hardware, or dispatch identity.
// Strings are copied by the recorder; the caller may provide short-lived
// literals or local buffers.
struct ggml_hip_blas_observation_v1 {
    const char * operand_a_type;
    const char * operand_b_type;
    const char * output_type;
    const char * accumulation_type;
    const char * source_a_conversion;
    const char * source_b_conversion;
    const char * output_conversion;
    const char * requested_precision;
    const char * effective_call_api;
    const char * effective_provider;
    const char * effective_backend;
    uint64_t     source_a_temp_bytes;
    uint64_t     source_b_temp_bytes;
    uint64_t     output_temp_bytes;
};

void ggml_hip_record_blas_metadata(const ggml_hip_blas_observation_v1 & metadata);

// Write everything observed so far to GGML_HIP_DISPATCH_DB (JSONL). Called at
// backend teardown and safe to call repeatedly -- it rewrites the file rather
// than appending, so a checkpoint mid-run leaves a complete, valid document.
void ggml_hip_record_flush();

// Human-readable coverage report: signatures seen, native candidate per
// signature, frequency ranking. This is what tells you whether a record run
// actually exercised the workload you thought it did.
void ggml_hip_record_write_report(const char * path);

size_t ggml_hip_record_signature_count();

#endif // GGML_USE_HIP && GGML_HIP_AUTOTUNE_RECORD
