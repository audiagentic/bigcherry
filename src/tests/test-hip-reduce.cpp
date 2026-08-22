// bigcherry: HI18 standalone SPLIT_REDUCE correctness probe (patches/1224).
//
// Drives the REAL production META backend (ggml_backend_meta_device -- the
// same construction llama.cpp's own LLAMA_SPLIT_MODE_TENSOR path uses, see
// src/llama.cpp's llama_prepare_model_devices) with a minimal 2-node graph:
//
//     partial = MUL_MAT(A[D,N], B[D,1])
//     out     = DUP(partial)
//
// A's and B's axis-0 split (one element per rank) makes handle_mul_mat()'s
// real axis0+axis0 rule assign `partial` GGML_BACKEND_SPLIT_AXIS_PARTIAL
// pre-synchronization and `out` MIRRORED post-synchronization (verified
// against ggml-backend-meta.cpp) -- so the subgraph partitioner creates a
// genuine allreduce boundary between them, driven through whichever
// provider GGML_HIP_REDUCE_PLAN selects (patches/0830), exactly as
// production does. This probe never hand-encodes PARTIAL/MIRRORED itself;
// only the two static leaf tensors get an explicit split state.
//
// The K=1 local-matmul trick means each rank's local contribution to
// `partial` IS its frozen rank-N.f32 input value exactly (rank_r[i] * 1.0),
// so frozen bytes are injected/read back without any private META
// accessor -- per-rank output is captured via the existing HI58 telemetry
// tensor-array hook (hip-autotune-reduce-telemetry.h's test-capture seam),
// reused rather than exposing ggml_backend_meta_buffer_simple_tensor
// (deliberately private -- only ggml_backend_meta_simple_backend is
// exported, and that gives a backend handle, not per-tensor readback).
//
// This is a Python-facing FACT reporter, not a correctness verdict: see
// tools/bigcherry/reduce_correctness.py, which owns every pass/fail
// decision. This probe's only obligation is to run the real production
// path once per invocation and report exactly what happened.
//
// Usage:
//   test-hip-reduce --case <case-dir> --plan auto|rccl|meta \
//       --devices 0,1 --out <result.json>
//
// GGML_HIP_REDUCE_PLAN must already be set in the process environment to
// the same value as --plan before this probe is invoked (the caller sets
// it, matching tools/bigcherry/correctness_evidence.py's env=... pattern
// for test-backend-ops) -- this probe reads it, it does not set it itself.
//
// case-dir must contain case.json (device_count, element_count,
// input_digests, ...) and rank-0.f32 .. rank-(D-1).f32 (frozen
// little-endian F32 bytes), written by reduce_correctness.py's write_case().
//
// Writes --out as machine-readable JSON and, alongside it, one
// <out-dir>/<out-stem>-rank-N.f32 per participating device: the raw
// post-reduction bytes for that device, read back only after the meta
// backend has been synchronized.
//
// HI18 D=2 slice only: built with generic-D mechanics (matches HI84's
// planned N>2 extension -- same split-state callback, same K=1 encoding)
// but this probe refuses to run with a device count other than 2. Relax
// that guard, not the mechanics above, when HI84's hardware work starts.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda/hip-autotune-reduce-telemetry.h"

#include "hash/hash.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::fprintf(stderr, "test-hip-reduce: %s\n", message.c_str());
    std::exit(1);
}

struct probe_options {
    std::string case_dir;
    std::string plan;
    std::vector<int> devices;
    std::string out_path;
};

std::vector<int> parse_device_list(const std::string & text) {
    std::vector<int> devices;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            devices.push_back(std::stoi(item));
        }
    }
    return devices;
}

probe_options parse_args(int argc, char ** argv) {
    probe_options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fail("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--case") {
            opts.case_dir = next();
        } else if (arg == "--plan") {
            opts.plan = next();
        } else if (arg == "--devices") {
            opts.devices = parse_device_list(next());
        } else if (arg == "--out") {
            opts.out_path = next();
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if (opts.case_dir.empty() || opts.plan.empty() || opts.devices.empty() || opts.out_path.empty()) {
        fail("usage: test-hip-reduce --case <dir> --plan auto|rccl|meta --devices 0,1 --out <result.json>");
    }
    if (opts.plan != "auto" && opts.plan != "rccl" && opts.plan != "meta") {
        fail("--plan must be one of auto|rccl|meta, got: " + opts.plan);
    }
    return opts;
}

std::string sha256_hex_of(const std::vector<uint8_t> & data) {
    return hash_sha256_hex(data.data(), data.size());
}

std::vector<uint8_t> read_file_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fail("cannot read file: " + path);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file_bytes(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fail("cannot write file: " + path);
    }
    f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

std::string dirname_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

std::string stem_of(const std::string & path) {
    std::string base = path;
    const size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    const size_t dot = base.find_last_of('.');
    return dot != std::string::npos ? base.substr(0, dot) : base;
}

struct loaded_case {
    json manifest;
    size_t device_count = 0;
    int64_t element_count = 0;
    int64_t slice_shape[4] = {1, 1, 1, 1};
    std::vector<std::vector<uint8_t>> rank_bytes;
    std::vector<std::string> input_digests;
};

loaded_case load_case(const std::string & case_dir) {
    loaded_case c;
    std::ifstream mf(case_dir + "/case.json");
    if (!mf) {
        fail("cannot read case.json in " + case_dir);
    }
    mf >> c.manifest;
    c.device_count = c.manifest.at("device_count").get<size_t>();
    c.element_count = c.manifest.at("element_count").get<int64_t>();

    const auto & shape = c.manifest.at("slice_shape");
    if (shape.size() != 4) {
        fail("case.json slice_shape must have exactly 4 entries, got " + std::to_string(shape.size()));
    }
    int64_t shape_product = 1;
    for (size_t i = 0; i < 4; ++i) {
        c.slice_shape[i] = shape.at(i).get<int64_t>();
        shape_product *= c.slice_shape[i];
    }
    if (shape_product != c.element_count) {
        fail("case.json slice_shape product " + std::to_string(shape_product) +
             " does not match element_count " + std::to_string(c.element_count));
    }

    const auto & recorded_digests = c.manifest.at("input_digests");

    for (size_t d = 0; d < c.device_count; ++d) {
        const std::string path = case_dir + "/rank-" + std::to_string(d) + ".f32";
        std::vector<uint8_t> bytes = read_file_bytes(path);
        if (static_cast<int64_t>(bytes.size()) != c.element_count * 4) {
            fail("rank-" + std::to_string(d) + ".f32 has " + std::to_string(bytes.size()) +
                 " bytes, expected " + std::to_string(c.element_count * 4));
        }
        const std::string digest = sha256_hex_of(bytes);
        const std::string recorded = recorded_digests.at(d).get<std::string>();
        if (digest != recorded) {
            fail("rank-" + std::to_string(d) + ".f32 digest mismatch: computed " + digest +
                 ", case.json says " + recorded + " -- file modified since generation");
        }
        c.rank_bytes.push_back(std::move(bytes));
        c.input_digests.push_back(digest);
    }
    return c;
}

// Mirrors llama.cpp's own LLAMA_SPLIT_MODE_TENSOR device-collection loop
// (src/llama.cpp's llama_prepare_model_devices): enumerate every registered
// device, keep the GPU ones in registration order, then index into that
// filtered list by the requested logical HIP ordinal.
std::vector<ggml_backend_dev_t> resolve_gpu_devices(const std::vector<int> & hip_ordinals) {
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> gpu_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_devs.push_back(dev);
        }
    }
    std::vector<ggml_backend_dev_t> selected;
    for (int ordinal : hip_ordinals) {
        if (ordinal < 0 || static_cast<size_t>(ordinal) >= gpu_devs.size()) {
            fail("requested device ordinal " + std::to_string(ordinal) + " but only " +
                 std::to_string(gpu_devs.size()) + " GPU devices are registered");
        }
        selected.push_back(gpu_devs[static_cast<size_t>(ordinal)]);
    }
    return selected;
}

struct split_state_config {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    size_t device_count = 0;
    // Elements of the split axis assigned to each rank (== the target
    // shape's s1; 1 in the original flattened-shape special case).
    int64_t rank_size = 1;
};

ggml_backend_meta_split_state probe_split_state(const ggml_tensor * tensor, void * userdata) {
    auto * cfg = static_cast<split_state_config *>(userdata);
    ggml_backend_meta_split_state s = {};
    s.nr[0] = 1;
    s.n_segments = 1;
    if (tensor == cfg->a || tensor == cfg->b) {
        // Static leaf tensors only: one axis-0 element per rank. META
        // derives the compute tensors' (partial, out) split state itself
        // from the real handle_mul_mat()/DUP rules -- this callback must
        // never encode PARTIAL/MIRRORED directly, or the probe would be
        // manufacturing the answer instead of exercising production
        // partitioning (verified against ggml-backend-meta.cpp).
        s.axis = GGML_BACKEND_SPLIT_AXIS_0;
        for (size_t rank = 0; rank < cfg->device_count; ++rank) {
            s.ne[rank] = cfg->rank_size;
        }
        return s;
    }
    s.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    const probe_options opts = parse_args(argc, argv);
    const size_t D = opts.devices.size();

    // HI18 D=2 slice: mechanics above are device-count-generic (see HI84
    // for the planned D=3/D=4 extension), but this build only claims D=2
    // hardware evidence. Relax this guard, not the mechanics, when HI84
    // starts.
    if (D != 2) {
        fail("HI18 currently qualifies D=2 only (got " + std::to_string(D) +
             " devices) -- see HI84 for the planned N>2 extension");
    }

    const char * env_plan = std::getenv("GGML_HIP_REDUCE_PLAN");
    if (env_plan == nullptr || opts.plan != env_plan) {
        fail("GGML_HIP_REDUCE_PLAN environment variable (" +
             std::string(env_plan != nullptr ? env_plan : "<unset>") +
             ") does not match --plan " + opts.plan +
             " -- the caller must set this env var before invoking the probe");
    }

    const loaded_case c = load_case(opts.case_dir);
    if (c.device_count != D) {
        fail("case device_count=" + std::to_string(c.device_count) +
             " does not match --devices count=" + std::to_string(D));
    }
    const int64_t N = c.element_count;

    const std::vector<ggml_backend_dev_t> simple_devs = resolve_gpu_devices(opts.devices);

    split_state_config split_cfg;
    split_cfg.device_count = D;

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
        const_cast<ggml_backend_dev_t *>(simple_devs.data()), simple_devs.size(),
        probe_split_state, &split_cfg);
    if (meta_dev == nullptr) {
        fail("ggml_backend_meta_device() returned null");
    }
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    if (meta_backend == nullptr) {
        fail("ggml_backend_dev_init(meta_dev) returned null");
    }

    // -- shape-preserving K=1-generalized construction --
    // Target output shape is the case's real slice_shape [s0,s1,s2,s3], not
    // merely its total element_count -- HI18's reduction_signature_key is
    // keyed on the full 4D shape (verified against tools/bigcherry/
    // telemetry.py), so a flattened [N,1,1,1] collective would produce
    // clean-looking evidence for the WRONG signature. Each rank's local
    // matmul reduction dimension is s1 (not 1): A[D*s1,s0,s2,s3] holds the
    // frozen per-rank data, B[D*s1,s1,s2,s3] is an s1xs1 identity block
    // repeated per rank/batch element, so partial_rank = A_rank x I =
    // A_rank exactly, landing the frozen bytes in the real output shape.
    // The original K=1 case is this construction's s1==1 special case.
    const int64_t s0 = c.slice_shape[0];
    const int64_t s1 = c.slice_shape[1];
    const int64_t s2 = c.slice_shape[2];
    const int64_t s3 = c.slice_shape[3];
    const int64_t K = static_cast<int64_t>(D) * s1;
    (void) N;

    const ggml_init_params static_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(static_params);

    ggml_tensor * a = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, K, s0, s2, s3);
    ggml_tensor * b = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, K, s1, s2, s3);
    ggml_set_name(a, "hi18.rank_values");
    ggml_set_name(b, "hi18.identity");
    split_cfg.a = a;
    split_cfg.b = b;
    split_cfg.rank_size = s1;

    ggml_backend_buffer_t static_buf = ggml_backend_alloc_ctx_tensors(ctx_static, meta_backend);
    if (static_buf == nullptr) {
        fail("failed to allocate static tensors on the meta backend");
    }
    ggml_backend_buffer_set_usage(static_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // -- compute graph: partial = MUL_MAT(A, B); out = DUP(partial) --
    const ggml_init_params compute_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(compute_params);

    ggml_tensor * partial = ggml_mul_mat(ctx_compute, a, b);
    ggml_tensor * out = ggml_dup(ctx_compute, partial);
    ggml_set_name(partial, "hi18.partial");
    ggml_set_name(out, "hi18.out");

    ggml_cgraph * graph = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(graph, out);

    // Load-bearing (GPT review, 2026-08-22 -- verified against ggml-backend-
    // meta.cpp before fixing): ggml_backend_alloc_ctx_tensors() on a META
    // buffer type immediately resolves and CACHES each tensor's split state
    // at allocation time (ggml_backend_meta_alloc_ctx_tensors_from_buft ->
    // ggml_backend_meta_buffer_init_tensor_impl, before any later
    // set_usage() call could run) -- so allocating `partial`/`out` that way
    // and setting COMPUTE usage afterward is too late: META would already
    // have asked probe_split_state() for them and cached MIRRORED. The
    // scheduler's graph allocator creates its buffers WITH
    // GGML_BACKEND_BUFFER_USAGE_COMPUTE from the start (ggml-alloc.c's
    // ggml_vbuffer_alloc), so tensor init sees the correct usage the first
    // time, and META derives partial/out's state from the real
    // handle_mul_mat()/DUP rules instead.
    ggml_backend_buffer_type_t meta_buft = ggml_backend_get_default_buffer_type(meta_backend);
    ggml_backend_t sched_backends[] = { meta_backend };
    ggml_backend_buffer_type_t sched_bufts[] = { meta_buft };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        sched_backends, sched_bufts, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (sched == nullptr) {
        fail("failed to create the meta backend scheduler");
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("failed to allocate the compute graph via the scheduler");
    }

    // -- populate A with frozen rank bytes at (i0,i1,i2,i3), B as a per-rank/per-batch s1xs1 identity --
    std::vector<float> a_host(static_cast<size_t>(K * s0 * s2 * s3), 0.0f);
    std::vector<float> b_host(static_cast<size_t>(K * s1 * s2 * s3), 0.0f);
    for (size_t rank = 0; rank < D; ++rank) {
        for (int64_t i3 = 0; i3 < s3; ++i3) {
            for (int64_t i2 = 0; i2 < s2; ++i2) {
                for (int64_t i1 = 0; i1 < s1; ++i1) {
                    const int64_t global_k = static_cast<int64_t>(rank) * s1 + i1;
                    for (int64_t i0 = 0; i0 < s0; ++i0) {
                        const int64_t flat = i0 + s0 * (i1 + s1 * (i2 + s2 * i3));
                        float value = 0.0f;
                        std::memcpy(&value, c.rank_bytes[rank].data() + flat * 4, 4);
                        const int64_t a_idx = global_k + K * (i0 + s0 * (i2 + s2 * i3));
                        a_host[static_cast<size_t>(a_idx)] = value;
                    }
                    for (int64_t m = 0; m < s1; ++m) {
                        const int64_t b_idx = global_k + K * (m + s1 * (i2 + s2 * i3));
                        b_host[static_cast<size_t>(b_idx)] = (m == i1) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }

    ggml_backend_tensor_set(a, a_host.data(), 0, a_host.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_host.data(), 0, b_host.size() * sizeof(float));

    ggml_hip_reduce_test_capture_reset();
    const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    ggml_backend_sched_synchronize(sched);
    const bool completion_synchronized = status == GGML_STATUS_SUCCESS;

    ggml_hip_reduce_test_snapshot_v1 snap;
    const bool captured = ggml_hip_reduce_test_capture_snapshot(&snap);

    const int64_t expected_nbytes = s0 * s1 * s2 * s3 * static_cast<int64_t>(sizeof(float));

    json result;
    result["schema_version"] = 1;
    result["case_id"] = c.manifest.value("case_id", "");
    result["plan"] = opts.plan;
    result["devices"] = opts.devices;
    result["device_count"] = D;
    result["input_digests"] = c.input_digests;
    result["graph_compute_status"] = ggml_status_to_string(status);
    result["completion_synchronized"] = completion_synchronized;
    result["captured"] = captured;

    // A successfully-computed graph with nothing captured (or captured for
    // fewer devices than requested) means the intended PARTIAL/allreduce
    // boundary was silently missed -- e.g. the exact COMPUTE-usage-timing
    // bug this construction was rewritten to avoid. That must never look
    // like a clean result: valid only if the graph computed successfully
    // AND a full-device-count reduction was actually observed AND its
    // runtime signature matches what this case claims.
    bool signature_matches = false;

    if (!captured) {
        result["requested_provider"] = opts.plan;
        result["effective_provider"] = "unknown";
        result["provider_succeeded"] = false;
        result["handoff"] = "unknown";
        result["fallback_depth"] = 0;
        result["outputs"] = json::array();
    } else {
        result["requested_provider"] = std::string(snap.requested_provider);
        result["effective_provider"] = std::string(snap.effective_provider);
        result["provider_succeeded"] = snap.provider_succeeded;
        result["handoff"] = std::string(snap.handoff);
        result["fallback_depth"] = snap.fallback_depth;

        json signature;
        signature["version"] = 1;
        signature["element_count"] = snap.element_count;
        signature["element_type"] = std::string(snap.element_type);
        signature["slice_shape"] = {snap.slice_shape[0], snap.slice_shape[1],
                                     snap.slice_shape[2], snap.slice_shape[3]};
        signature["topology_key"] = std::string(snap.topology_key);
        signature["peer_access"] = std::string(snap.peer_access);
        result["reduction_signature"] = signature;

        signature_matches = snap.element_count == c.element_count &&
            snap.slice_shape[0] == s0 && snap.slice_shape[1] == s1 &&
            snap.slice_shape[2] == s2 && snap.slice_shape[3] == s3;
        result["reduction_signature_matches_case"] = signature_matches;

        json outputs = json::array();
        const std::string out_dir = dirname_of(opts.out_path);
        const std::string out_stem = stem_of(opts.out_path);

        if (snap.device_count != D) {
            fail("captured snapshot has device_count=" + std::to_string(snap.device_count) +
                 ", expected " + std::to_string(D) +
                 " -- missing participant output, this arm is not valid evidence");
        }

        for (size_t rank = 0; rank < snap.device_count; ++rank) {
            ggml_tensor * t = snap.tensors[rank];
            if (t == nullptr) {
                fail("captured snapshot has a null tensor for rank " + std::to_string(rank));
            }
            if (snap.devices[rank] != opts.devices[rank]) {
                fail("captured snapshot rank " + std::to_string(rank) + " reports device " +
                     std::to_string(snap.devices[rank]) + ", expected " +
                     std::to_string(opts.devices[rank]));
            }
            const size_t nbytes = ggml_nbytes(t);
            if (static_cast<int64_t>(nbytes) != expected_nbytes) {
                fail("rank " + std::to_string(rank) + " output is " + std::to_string(nbytes) +
                     " bytes, expected " + std::to_string(expected_nbytes) +
                     " for shape [" + std::to_string(s0) + "," + std::to_string(s1) + "," +
                     std::to_string(s2) + "," + std::to_string(s3) + "]");
            }
            std::vector<uint8_t> out_bytes(nbytes);
            ggml_backend_tensor_get(t, out_bytes.data(), 0, nbytes);

            const std::string rank_path =
                out_dir + "/" + out_stem + "-rank-" + std::to_string(rank) + ".f32";
            write_file_bytes(rank_path, out_bytes);

            json entry;
            entry["device"] = snap.devices[rank];
            entry["byte_count"] = nbytes;
            entry["sha256"] = sha256_hex_of(out_bytes);
            entry["path"] = rank_path;
            outputs.push_back(entry);
        }
        result["outputs"] = outputs;
    }

    const bool probe_valid = completion_synchronized && captured &&
        snap.device_count == D && signature_matches;
    result["probe_valid"] = probe_valid;

    std::ofstream out_file(opts.out_path);
    if (!out_file) {
        fail("cannot write result file: " + opts.out_path);
    }
    out_file << result.dump(2);

    // Distinct from "the graph computed" (status): a successful compute
    // with no valid captured reduction (finding #1's exact failure mode)
    // must exit nonzero, not look like a clean run to the caller.
    return probe_valid ? 0 : 2;
}
