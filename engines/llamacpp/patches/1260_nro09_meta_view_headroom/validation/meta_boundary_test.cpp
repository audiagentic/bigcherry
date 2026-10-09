// PNRO09 / 1260 hardware-free Meta-context view-headroom boundary test.
//
// Probes the REAL Meta backend compute-container (stc_compute) capacity, which is
// sized `compute_headroom * ggml_get_mem_size(user_ctx)` in ggml-backend-meta.cpp
// (the constant under test: 16 -> 80), WITHOUT asserting the constant value.
//
// Mechanism (verified against ggml-backend-meta.cpp):
//   * The USER context's tensors (the static leaf `t`) are mapped into stc_static
//     (mem_size = S) by ggml_backend_alloc_ctx_tensors.
//   * The "views" that consume compute_headroom are VIEW tensors created BETWEEN
//     evals (NOT in the USER context). Each such view, when registered via
//     buf->iface.init_tensor (ggml_backend_meta_buffer_init_tensor), creates a
//     simple tensor in stc_compute's context via ggml_new_tensor(simple_ctx, ...).
//     Because simple_ctx has no_alloc=true, the simple tensor's object is the
//     fixed GGML_TENSOR_SIZE (metadata only, obj_alloc_size = 0) -- NOT the data.
//   * get_simple_tensor_container routes a tensor to stc_static only if it is
//     already in stc_static.simple_tensors; otherwise to stc_compute[index].
//     The views (in a separate context) are NOT in stc_static, so they land in
//     stc_compute[0] (stc_compute_index_next stays 0 without a graph_compute).
//   * stc_compute[0] mem_size = compute_headroom * S. Each view consumes one
//     GGML_TENSOR_SIZE object. So N views fit iff N * GGML_TENSOR_SIZE <=
//     compute_headroom * S, i.e. N_max = compute_headroom * S / GGML_TENSOR_SIZE.
//
// The test creates one static leaf `t` (size S) in the USER context, allocates it
// to a Meta backend (single CPU simple device), then creates N view tensors of `t`
// in a separate context and registers each into the meta buffer. It reports OK if
// all N views fit in the compute container, else FAIL (a capacity abort also yields
// a non-zero exit code).
//
// The producer drives this binary for N in {15,16,17,72,80,81} (and the observed
// boundary) against both the pinned (unpatched, headroom=16) and patched
// (headroom=80) builds and records the observed N_max. N_max scales with
// compute_headroom, so ratio(patched/unpatched) = 80/16 = 5.
//
// Build (lean, CPU + Meta backend, no GPU):
//   g++ -O2 -std=c++17 meta_boundary_test.cpp \
//     -I<vendor>/ggml/include -o meta_boundary_test \
//     <build>/ggml/src/libggml.a <build>/ggml/src/libggml-cpu.a \
//     <build>/ggml/src/libggml-base.a -lpthread -fopenmp
//
// Usage: meta_boundary_test <N> <S_bytes> <label>
//   Prints "OK <label>" on success or "FAIL <label>" / non-zero exit on failure.

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-backend-impl.h> // for struct ggml_backend_device + iface.init_backend
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Trivial single-segment, non-mirrored split state: the (single) simple device
// owns the full tensor along axis 0. ne[0] (segment 0, device 0) = full extent.
static struct ggml_backend_meta_split_state trivial_split_state(
        const struct ggml_tensor * tensor, void * /*userdata*/) {
    struct ggml_backend_meta_split_state ss;
    ss.axis          = GGML_BACKEND_SPLIT_AXIS_0;
    ss.n_segments    = 1;
    ss.nr[0]         = 1;
    ss.ne[0]         = tensor->ne[0]; // full extent on axis 0, single device
    return ss;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <N> <S_bytes> <label>\n", argv[0]);
        return 2;
    }
    const int    N    = atoi(argv[1]);
    const size_t S    = (size_t) atol(argv[2]);
    const char * label = argv[3];

    const int64_t n_elem = (int64_t) (S / 4); // F32 = 4 bytes
    if (n_elem <= 0) { fprintf(stderr, "FAIL %s (bad S=%zu)\n", label, S); return 1; }

    // Register the CPU backend (so ggml_backend_reg_by_name("CPU") resolves).
    ggml_backend_cpu_reg();

    // The "user" context: reserved mem_size == S (drives stc sizing).
    struct ggml_init_params p_user = {
        /*.mem_size   =*/ S,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(p_user);
    if (ctx == nullptr) { fprintf(stderr, "FAIL %s (ggml_init user)\n", label); return 1; }

    // Static leaf tensor `t` of size S (in the USER context -> stc_static).
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
    if (t == nullptr) { fprintf(stderr, "FAIL %s (leaf)\n", label); return 1; }

    // CPU simple device -> Meta device -> Meta backend.
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    if (cpu_reg == nullptr) { fprintf(stderr, "FAIL %s (cpu_reg)\n", label); return 1; }
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(cpu_reg, 0);
    if (cpu_dev == nullptr) { fprintf(stderr, "FAIL %s (cpu_dev)\n", label); return 1; }

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(&cpu_dev, 1, trivial_split_state, nullptr);
    if (meta_dev == nullptr) { fprintf(stderr, "FAIL %s (meta_dev)\n", label); return 1; }

    ggml_backend_t meta_backend = meta_dev->iface.init_backend(meta_dev, nullptr);
    if (meta_backend == nullptr) { fprintf(stderr, "FAIL %s (meta_backend)\n", label); return 1; }

    // Allocate the USER context to the meta backend (maps `t` -> stc_static;
    // sizes stc_compute = compute_headroom * S).
    struct ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors(ctx, meta_backend);
    if (buf == nullptr) { fprintf(stderr, "FAIL %s (alloc_ctx)\n", label); return 1; }

    // A separate context for the between-eval "view" tensors (NOT the USER
    // context, so they are NOT in stc_static.simple_tensors and land in
    // stc_compute[0] when registered). Its pool must be large enough to hold all
    // N view objects (each ~ S + metadata, allocated here) so that the binding
    // capacity constraint is stc_compute (mem_size = compute_headroom * S), not
    // this context's pool.
    struct ggml_init_params p_views = {
        /*.mem_size   =*/ (size_t) N * (S + 2048),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx2 = ggml_init(p_views);
    if (ctx2 == nullptr) { fprintf(stderr, "FAIL %s (ggml_init views)\n", label); return 1; }

    // Create N view tensors of `t` (the between-eval snapshots) and register each
    // into the meta buffer. Each registration creates a simple tensor in
    // stc_compute[0]'s context (fixed GGML_TENSOR_SIZE object). The capacity
    // probe: N views fit iff N * GGML_TENSOR_SIZE <= compute_headroom * S.
    for (int i = 0; i < N; i++) {
        struct ggml_tensor * view = ggml_view_1d(ctx2, t, 0, n_elem);
        if (view == nullptr) { fprintf(stderr, "FAIL %s (view %d)\n", label, i); return 1; }
        view->buffer = buf; // the meta buffer (required by init_tensor)
        enum ggml_status st = buf->iface.init_tensor(buf, view);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL %s (view %d init_tensor status=%d)\n", label, i, (int) st);
            return 1;
        }
    }

    printf("OK %s\n", label);
    return 0;
}
