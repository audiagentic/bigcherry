#include "hip-autotune-journal.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "hip-autotune-blake2b.h"
#include "hip-autotune-signature.h"

#include <mutex>
#include <stdio.h>

#if defined(_WIN32)
#  include <io.h>
#else
#  include <unistd.h>
#endif

namespace {

FILE *      g_journal_file = nullptr;
std::string g_experiment_id;
uint64_t    g_sequence = 0;
std::mutex  g_journal_mutex;

// json.dumps(ensure_ascii=True)'s escaping for the ASCII range this file's
// inputs are drawn from (hex digests, revisions, identifiers, and the
// already-ASCII result JSON built elsewhere in this codebase). Values
// outside 0x00-0x7F are not expected here and are not specially escaped --
// every string this module ever embeds is hex/identifier text or a JSON
// blob this codebase already only ever emits as ASCII.
std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

std::string checksum_hex(const std::string & canonical_json) {
    uint8_t digest[16];
    ggml_hip_blake2b(digest, sizeof(digest), canonical_json.data(),
                     canonical_json.size(), "bc-journal-v1");
    static const char hex[] = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[(size_t) i * 2]     = hex[digest[i] >> 4];
        out[(size_t) i * 2 + 1] = hex[digest[i] & 0xF];
    }
    return out;
}

// Canonical (sorted-key, no-whitespace) form of the envelope, *without* the
// checksum field -- must match tools/bigcherry/tune_journal.py's
// checksum(event_without_checksum) exactly. The envelope's own field set is
// fixed and small (five keys), so it is hand-alphabetised here rather than
// run through a general canonicalizer: experiment_id, kind, payload,
// schema_version, sequence.
//
// `payload_value` must already be a complete, valid JSON value (a quoted
// string for "result"/"attempt" events, an already-sorted object literal for
// "start"/"complete") -- this function does not itself decide which.
std::string canonical_envelope(const std::string & experiment_id, const char * kind,
                               const std::string & payload_value, uint64_t sequence) {
    return "{\"experiment_id\":\"" + json_escape(experiment_id) + "\","
           "\"kind\":\"" + kind + "\","
           "\"payload\":" + payload_value + ","
           "\"schema_version\":1,"
           "\"sequence\":" + std::to_string(sequence) + "}";
}

void write_event(const char * kind, const std::string & payload_value) {
    if (g_journal_file == nullptr) {
        return;
    }
    const uint64_t sequence  = ++g_sequence;
    const std::string canonical = canonical_envelope(g_experiment_id, kind, payload_value, sequence);
    const std::string checksum  = checksum_hex(canonical);
    const std::string line = "{\"checksum\":\"" + checksum + "\"," + canonical.substr(1) + "\n";
    fwrite(line.data(), 1, line.size(), g_journal_file);
    fflush(g_journal_file);
#if defined(_WIN32)
    _commit(_fileno(g_journal_file));
#else
    fsync(fileno(g_journal_file));
#endif
}

} // namespace

bool ggml_hip_journal_open(const char * path, const std::string & experiment_id,
                           const std::string & source_revision,
                           const std::string & manifest_hash,
                           const ggml_hip_digest & hardware_digest) {
    std::lock_guard<std::mutex> lock(g_journal_mutex);
    if (g_journal_file != nullptr || path == nullptr || path[0] == '\0') {
        return false;
    }
    // "wbx": POSIX/C11 exclusive-create. Mirrors tune_journal.py's
    // JournalWriter, which refuses to open an already-existing journal path
    // rather than silently appending to or truncating one from a prior run.
    FILE * file = fopen(path, "wbx");
    if (file == nullptr) {
        return false;
    }
    g_journal_file  = file;
    g_experiment_id = experiment_id;
    g_sequence      = 0;

    const std::string payload =
        "{\"batch_size\":1,"
        "\"durability_mode\":\"durable_each\","
        "\"durability_scope\":\"storage_claimed\","
        "\"hardware_digest\":\"" + ggml_hip_digest_hex(hardware_digest) + "\","
        "\"manifest_hash\":\"" + json_escape(manifest_hash) + "\","
        "\"source_revision\":\"" + json_escape(source_revision) + "\","
        "\"storage_kind\":\"local\"}";
    write_event("start", payload);
    return true;
}

void ggml_hip_journal_append_result(const std::string & result_json) {
    std::lock_guard<std::mutex> lock(g_journal_mutex);
    write_event("result", "\"" + json_escape(result_json) + "\"");
}

void ggml_hip_journal_append_attempt(const std::string & attempt_json) {
    std::lock_guard<std::mutex> lock(g_journal_mutex);
    write_event("attempt", "\"" + json_escape(attempt_json) + "\"");
}

void ggml_hip_journal_close() {
    std::lock_guard<std::mutex> lock(g_journal_mutex);
    if (g_journal_file == nullptr) {
        return;
    }
    write_event("complete", "{}");
    fclose(g_journal_file);
    g_journal_file = nullptr;
}

bool ggml_hip_journal_is_open() {
    std::lock_guard<std::mutex> lock(g_journal_mutex);
    return g_journal_file != nullptr;
}

#endif
