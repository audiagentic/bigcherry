// bigcherry: append-only crash-safe tuning journal (HI48).
//
// One self-contained JSON event per line. Every event is independently
// checksummed (BLAKE2b-128) and flushed+fsync'd before the writer returns,
// so a killed process leaves a prefix of fully-durable, individually
// verifiable events -- never a torn write presented as complete.

#pragma once

#include "hip-autotune-types.h"

#include <string>

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

// Opens the journal file at `path` (exclusive-create -- refuses to append to
// or truncate an existing journal from a prior run) and writes the "start"
// event. Returns false (and leaves the journal closed) if the path already
// exists or cannot be created; the caller logs a warning and proceeds
// without journaling rather than failing the tune.
bool ggml_hip_journal_open(const char * path, const std::string & experiment_id,
                           const std::string & source_revision,
                           const std::string & manifest_hash,
                           const ggml_hip_digest & hardware_digest);

// Appends one "result" event and fsyncs before returning. `result_json` must
// be one complete, valid JSON value (typically an object) -- it is embedded
// as an opaque string, not reparsed or validated here.
void ggml_hip_journal_append_result(const std::string & result_json);

// Appends one "attempt" event and fsyncs before returning, same contract as
// append_result. Diagnostic only, written immediately before a candidate's
// GPU work begins rather than after it succeeds -- see the call site in
// hip-autotune-tuner.cu for why "after" is not good enough: a candidate
// whose kernel corrupts device memory never reaches append_result, so
// without this the journal's last entry is one candidate short of the
// truth. tune_journal.py treats a trailing "attempt" as a valid, compactable
// terminal state and surfaces it as the likely cause of an incomplete run.
// Added 2026-08-11 alongside the MMQ crash investigation (see EX01) -- this
// is exactly the tooling that pinned the crashing candidate down.
void ggml_hip_journal_append_attempt(const std::string & attempt_json);

// Writes the "complete" event and closes the file. Call once, at the same
// point ggml_hip_tuner_flush() already writes the final measurements.jsonl.
// A process that never reaches this point leaves the journal without a
// terminal event -- exactly the "interrupted, recoverable" signal
// tune_journal.py's read_current() already looks for; no separate crash
// handler is needed.
void ggml_hip_journal_close();

bool ggml_hip_journal_is_open();

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
