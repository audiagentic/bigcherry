// QFP49 rungs A and B: the staggered half-batch pair on the real tensor-split (Meta) backend, without llama.
//
// A toy model of L layers on every visible GPU. Each half starts with a CPU prefix op whose output is copied by
// the scheduler into the final Meta split, matching the real prompt graph's host-prefix/input-copy path. A and B
// intentionally share the CPU prefix landing tensor: the pre-fix pair preparation overwrites A with B before A's
// scheduler copy, while sequential execution and the fixed A-prefix/A-copy/B-prefix/B-copy order are safe.
// Per layer, for a half of T tokens with hidden state h [n, T]:
//     u = MUL_MAT(W1[n, K], h)        W1 (Q8_0) split along axis 1: every card computes its slice of u
//     y = MUL_MAT(W2[K, n], u)        W2 split along axis 0: every card holds a partial sum -> cross-card sum
//     z = RMS_NORM(y)
//     z = z + C[:, other half]        A reads B from the previous call; B reads A from this call
//     C[:, half] = z                  a static, mirrored cache both halves share (the KV/recurrent-state role)
//     h = h + z
// Half A and half B are two graphs on two schedulers, as 1361 runs them in llama. B deliberately has one extra
// split/reduction output tail (the real prompt's last-token logits-head asymmetry); A does not. Both halves also
// have a CPU-produced host input copied into Meta. The growing-input stress rebuilds that input larger every pair
// through >4 MiB. The carried-state stress still submits fixed-shape pairs without a host synchronization.
//
// Arms, each from zeroed caches and the same inputs:
//   sequential   A computed, then B: the reference
//   pair         ggml_backend_sched_graph_compute_async_pair(A, B): must equal the reference bit for bit
//   wrong order  the pair entered as (B, A): B's layers run ahead of A's, so B must NOT equal the reference -
//                if it does, the harness is not testing the ordering rule
//   full         one graph of 2T tokens (the -ub 2T shape), timing only
//
// Usage: meta_pair_harness [T=256] [L=96] [n=2560] [k=1024] [reps=8] [cache=1]   (K = k x the number of cards;
//        cache=0 drops the shared cache: rung A, where the wrong-order arm cannot differ and is not judged)
// The carried-state stress uses max(32, 4*reps) consecutive pairs.
// Exit 0: pair identical and wrong order different. 1: pair differs. 2: wrong order not detected. 3: pair unsupported.
// Exit 4: carried-state pair differs from the sequential reference. Exit 5: growing-input pair differs.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::fprintf(stderr, "meta_pair_harness: %s\n", message.c_str());
    std::exit(10);
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct model {
    size_t  n_devs = 0;
    int64_t n = 0, K = 0, k = 0, T = 0;
    int     L = 0;
    bool    use_cache = true;  // false: rung A, no state shared between the halves
    std::vector<ggml_tensor *> w1, w2, cache;
    // A/B deliberately land their CPU prefix in the same static host tensor. Sequential execution is safe because
    // A is copied before B writes it; the pre-fix pair order (prefix A, prefix B, copy A, copy B) deterministically
    // feeds B's input to both halves.
    ggml_tensor * prefix_shared = nullptr;
    ggml_tensor * y_a = nullptr, * y_b = nullptr, * y_full = nullptr;  // results copied out of the compute arena
};

ggml_backend_meta_split_state split_state(const ggml_tensor * tensor, void * userdata) {
    const model * m = static_cast<const model *>(userdata);
    ggml_backend_meta_split_state s = {};
    s.nr[0] = 1;
    s.n_segments = 1;
    s.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    for (int l = 0; l < m->L && l < (int) m->w1.size(); ++l) {
        if (tensor == m->w1[l] || tensor == m->w2[l]) {
            s.axis = tensor == m->w1[l] ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
            for (size_t d = 0; d < m->n_devs; ++d) {
                s.ne[d] = m->k;
            }
            return s;
        }
    }
    return s;
}

struct half_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  graph = nullptr;
    ggml_tensor *  input = nullptr;
    ggml_tensor *  prefix = nullptr;
    ggml_tensor *  grow_input = nullptr;
    ggml_tensor *  grow_prefix = nullptr;
    ggml_tensor *  grow_consumer = nullptr;
    ggml_tensor *  out = nullptr;
    ggml_backend_sched_t sched = nullptr;
};

// first = cache column this graph writes; read_first = cache column it reads first (-1 for no read).
half_graph build(
        const model & m, ggml_tensor * result, int64_t first, int64_t tokens, int64_t read_first,
        int64_t grow_elems, bool output_tail) {
    half_graph g;
    const int graph_nodes = m.L * 12 + 24;
    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g.ctx = ggml_init(params);
    g.graph = ggml_new_graph_custom(g.ctx, graph_nodes, false);
    g.input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, m.n, tokens);
    ggml_set_input(g.input);
    ggml_format_name(g.input, "host_input_%lld", (long long) first);
    if (tokens == m.T) {
        GGML_ASSERT(m.prefix_shared != nullptr);
        // Distinct graph tensor objects alias the same CPU storage. This shares only the bytes, not op/src metadata.
        ggml_tensor * prefix_view = ggml_view_2d(
            g.ctx, m.prefix_shared, m.n, tokens, m.prefix_shared->nb[1], 0);
        g.prefix = ggml_cpy(g.ctx, g.input, prefix_view);
    } else {
        g.prefix = ggml_scale(g.ctx, g.input, 1.0f);
    }
    ggml_format_name(g.prefix, "cpu_prefix_%lld", (long long) first);

    // A second CPU prefix is intentionally variable-sized. grow_consumer is pinned to Meta by the harness, so the
    // complete grow_prefix tensor is a final-split host input and exercises 1361's pair-owned pinned H2D staging.
    g.grow_input = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, grow_elems);
    ggml_set_input(g.grow_input);
    ggml_format_name(g.grow_input, "grow_input_%lld", (long long) first);
    g.grow_prefix = ggml_scale(g.ctx, g.grow_input, 1.0f);
    ggml_format_name(g.grow_prefix, "grow_prefix_%lld", (long long) first);
    g.grow_consumer = ggml_sum(g.ctx, g.grow_prefix);
    ggml_format_name(g.grow_consumer, "grow_sum_%lld", (long long) first);

    ggml_tensor * h = g.prefix;
    for (int l = 0; l < m.L; ++l) {
        ggml_tensor * u = ggml_mul_mat(g.ctx, m.w1[l], h);
        ggml_tensor * y = ggml_mul_mat(g.ctx, m.w2[l], u);
        ggml_tensor * z = ggml_rms_norm(g.ctx, y, 1e-6f);
        if (m.use_cache) {
            const size_t col = m.cache[l]->nb[1];
            if (read_first >= 0) {
                ggml_tensor * theirs = ggml_view_2d(g.ctx, m.cache[l], m.n, tokens, col, (size_t) read_first * col);
                z = ggml_add(g.ctx, z, theirs);
            }
            ggml_tensor * mine = ggml_view_2d(g.ctx, m.cache[l], m.n, tokens, col, (size_t) first * col);
            ggml_build_forward_expand(g.graph, ggml_cpy(g.ctx, z, mine));
        }
        h = ggml_add(g.ctx, h, z);
    }
    if (output_tail) {
        // Real llama prompt halves are asymmetric: only B normally carries the requested last-token logits head.
        // One extra split matmul makes that difference visible at Meta subgraph granularity.
        ggml_tensor * u_tail = ggml_mul_mat(g.ctx, m.w1[0], h);
        ggml_tensor * y_tail = ggml_mul_mat(g.ctx, m.w2[0], u_tail);
        h = ggml_add(g.ctx, h, ggml_rms_norm(g.ctx, y_tail, 1e-6f));
    }
    // Consume the variable-size CPU prefix on Meta without changing the tensor shape of h.
    h = ggml_add1(g.ctx, h, g.grow_consumer);

    // a compute tensor's memory belongs to the per-device arena (1340) and is not readable after the graph; the
    // result is copied to a static tensor, as llama copies what it keeps
    g.out = result;
    ggml_build_forward_expand(g.graph, ggml_cpy(g.ctx, h, result));
    return g;
}

void zero_caches(const model & m) {
    std::vector<float> zeros((size_t) (m.n * 2 * m.T), 0.0f);
    for (ggml_tensor * c : m.cache) {
        ggml_backend_tensor_set(c, zeros.data(), 0, zeros.size() * sizeof(float));
    }
}

std::vector<float> read(const half_graph & g) {
    std::vector<float> v((size_t) ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, v.data(), 0, v.size() * sizeof(float));
    return v;
}

size_t differing(const std::vector<float> & a, const std::vector<float> & b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return n;
}

} // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // a hang must not swallow what was printed before it
    model m;
    m.T = argc > 1 ? std::atoll(argv[1]) : 256;
    m.L = argc > 2 ? std::atoi(argv[2]) : 96;
    m.n = argc > 3 ? std::atoll(argv[3]) : 2560;
    m.k = argc > 4 ? std::atoll(argv[4]) : 1024;
    const int reps = argc > 5 ? std::atoi(argv[5]) : 8;
    m.use_cache = argc > 6 ? std::atoi(argv[6]) != 0 : true;

    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(dev);
            std::printf("card %zu: %s\n", devs.size() - 1, ggml_backend_dev_description(dev));
        }
    }
    if (devs.size() < 2) {
        fail("needs at least two GPU devices");
    }
    m.n_devs = devs.size();
    m.K = m.k * (int64_t) m.n_devs;

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs.data(), devs.size(), split_state, &m);
    ggml_backend_t meta = meta_dev != nullptr ? ggml_backend_dev_init(meta_dev, nullptr) : nullptr;
    if (meta == nullptr) {
        fail("could not create the Meta backend");
    }
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);

    // static tensors: the weights (split) and the caches (mirrored)
    const ggml_init_params static_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (m.L * 3 + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(static_params);
    ggml_context * ctx_state = ggml_init(static_params);
    ggml_context * ctx_host = ggml_init(static_params);
    m.prefix_shared = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, m.n, m.T);
    ggml_set_name(m.prefix_shared, "shared_cpu_prefix");
    if (ggml_backend_alloc_ctx_tensors(ctx_host, cpu) == nullptr) {
        fail("could not allocate shared host prefix");
    }
    for (int l = 0; l < m.L; ++l) {
        // Q8_0 weights: the quantized matmul kernels production runs (and no BLAS library in the loop)
        m.w1.push_back(ggml_new_tensor_2d(ctx_static, GGML_TYPE_Q8_0, m.n, m.K));
        m.w2.push_back(ggml_new_tensor_2d(ctx_static, GGML_TYPE_Q8_0, m.K, m.n));
        m.cache.push_back(ggml_new_tensor_2d(ctx_state, GGML_TYPE_F32, m.n, 2 * m.T));
    }
    m.y_a = ggml_new_tensor_2d(ctx_state, GGML_TYPE_F32, m.n, m.T);
    m.y_b = ggml_new_tensor_2d(ctx_state, GGML_TYPE_F32, m.n, m.T);
    m.y_full = ggml_new_tensor_2d(ctx_state, GGML_TYPE_F32, m.n, 2 * m.T);
    // the caches and inputs get their own buffer, not a weights buffer: llama's KV cache is allocated this way
    if (ggml_backend_alloc_ctx_tensors(ctx_state, meta) == nullptr) {
        fail("could not allocate the caches");
    }
    ggml_backend_buffer_t static_buf = ggml_backend_alloc_ctx_tensors(ctx_static, meta);
    if (static_buf == nullptr) {
        fail("could not allocate the static tensors");
    }
    ggml_backend_buffer_set_usage(static_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        std::vector<float> w((size_t) (m.n * m.K));
        std::vector<uint8_t> q(ggml_nbytes(m.w1[0]));
        for (int l = 0; l < m.L; ++l) {
            for (float & v : w) v = dist(rng);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), q.data(), 0, m.K, m.n, nullptr);
            ggml_backend_tensor_set(m.w1[l], q.data(), 0, ggml_nbytes(m.w1[l]));
            for (float & v : w) v = dist(rng);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), q.data(), 0, m.n, m.K, nullptr);
            ggml_backend_tensor_set(m.w2[l], q.data(), 0, ggml_nbytes(m.w2[l]));
        }
    }

    // Reserve against a >4 MiB growing input even though normal correctness/timing runs use one float. The explicit
    // growing-input stress below rebuilds from small to this maximum in eight consecutive pair calls.
    const int grow_steps = 8;
    const int64_t grow_max_elems = (int64_t) (6 * 1024 * 1024 / sizeof(float));
    const int64_t grow_base_elems = 1;

    half_graph a = build(m, m.y_a, 0, m.T, m.T, grow_base_elems, false);
    half_graph b = build(m, m.y_b, m.T, m.T, 0, grow_base_elems, true);
    half_graph full = build(m, m.y_full, 0, 2 * m.T, -1, grow_base_elems, true);

    ggml_backend_t backends[] = { meta, cpu };
    ggml_backend_buffer_type_t bufts[] = { ggml_backend_get_default_buffer_type(meta), ggml_backend_get_default_buffer_type(cpu) };
    auto place_graph = [&](half_graph & g) {
        ggml_backend_sched_set_tensor_backend(g.sched, g.prefix, cpu);
        ggml_backend_sched_set_tensor_backend(g.sched, g.grow_prefix, cpu);
        ggml_backend_sched_set_tensor_backend(g.sched, g.grow_consumer, meta);
    };
    for (half_graph * g : { &a, &b, &full }) {
        g->sched = ggml_backend_sched_new(backends, bufts, 2, (size_t) (m.L * 12 + 24), false, false);
        if (g == &b) {
            ggml_backend_sched_set_stagger_pair_arena(g->sched, true);  // B's own compute arena, as 1361 sets it in llama
        }
        // reserve first, as llama does: without it the per-device arena (1340) grows after the graph is allocated and
        // the graph's tensors are left pointing into the arena it replaced
        // (with a graph of its own: reserving moves the Meta buffers on to their other view container, so the
        // graph that is computed has to be allocated after it, as llama builds a fresh graph a batch)
        half_graph measure = g == &a ? build(m, m.y_a, 0, m.T, m.T, grow_max_elems, false)
                           : g == &b ? build(m, m.y_b, m.T, m.T, 0, grow_max_elems, true)
                                     : build(m, m.y_full, 0, 2 * m.T, -1, grow_base_elems, true);
        // measure belongs to g's scheduler; set its placement directly before reserve.
        ggml_backend_sched_set_tensor_backend(g->sched, measure.prefix, cpu);
        ggml_backend_sched_set_tensor_backend(g->sched, measure.grow_prefix, cpu);
        ggml_backend_sched_set_tensor_backend(g->sched, measure.grow_consumer, meta);
        if (!ggml_backend_sched_reserve(g->sched, measure.graph)) {
            fail("could not reserve a graph");
        }
        ggml_backend_sched_reset(g->sched);
        ggml_free(measure.ctx);
    }
    // every reserve is done before any graph is allocated: a reserve empties the view container the earlier graphs'
    // views of the shared static tensors live in
    for (half_graph * g : { &a, &b, &full }) {
        place_graph(*g);
        if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
            fail("could not allocate a graph");
        }
        if (ggml_backend_sched_get_n_splits(g->sched) != 2) {
            fail("host-prefix harness graph did not produce exactly CPU + Meta scheduler splits");
        }
    }

    std::vector<float> xa((size_t) (m.n * m.T)), xb((size_t) (m.n * m.T)), xf((size_t) (m.n * 2 * m.T));
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float & v : xa) v = dist(rng);
        for (float & v : xb) v = dist(rng);
        std::memcpy(xf.data(), xa.data(), xa.size() * sizeof(float));
        std::memcpy(xf.data() + xa.size(), xb.data(), xb.size() * sizeof(float));
    }
    const std::vector<float> xa0 = xa, xb0 = xb;
    auto set_grow_input = [](half_graph & g, float marker) {
        std::vector<float> x((size_t) ggml_nelements(g.grow_input), 0.0f);
        if (!x.empty()) x[0] = marker;
        ggml_backend_tensor_set(g.grow_input, x.data(), 0, x.size() * sizeof(float));
    };
    auto set_inputs = [&](int call = 0) {
        const float da = 0.0009765625f * (float) call;
        const float db = -0.001953125f * (float) call;
        for (size_t i = 0; i < xa.size(); ++i) xa[i] = xa0[i] + da;
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = xb0[i] + db;
        ggml_backend_tensor_set(a.input, xa.data(), 0, xa.size() * sizeof(float));
        ggml_backend_tensor_set(b.input, xb.data(), 0, xb.size() * sizeof(float));
        set_grow_input(a, da);
        set_grow_input(b, db);
    };
    auto sync = [&] {
        ggml_backend_sched_synchronize(a.sched);
        ggml_backend_sched_synchronize(b.sched);
        ggml_backend_synchronize(meta);
    };
    auto sequential = [&] {
        if (ggml_backend_sched_graph_compute_async(a.sched, a.graph) != GGML_STATUS_SUCCESS ||
            ggml_backend_sched_graph_compute_async(b.sched, b.graph) != GGML_STATUS_SUCCESS) {
            fail("sequential compute failed");
        }
    };
    auto pair = [&](half_graph & first, half_graph & second) -> bool {
        bool supported = false;
        if (ggml_backend_sched_graph_compute_async_pair(first.sched, first.graph, second.sched, second.graph, &supported) != GGML_STATUS_SUCCESS) {
            fail("pair compute failed");
        }
        return supported;
    };
    auto rebuild_half = [&](half_graph & g, ggml_tensor * result, int64_t first, int64_t read_first,
                            int64_t grow_elems, bool output_tail) {
        ggml_backend_sched_t keep = g.sched;
        ggml_backend_sched_reset(keep);
        ggml_free(g.ctx);
        g = build(m, result, first, m.T, read_first, grow_elems, output_tail);
        g.sched = keep;
        place_graph(g);
        if (!ggml_backend_sched_alloc_graph(g.sched, g.graph)) {
            fail("could not allocate rebuilt growing-input graph");
        }
        if (ggml_backend_sched_get_n_splits(g.sched) != 2) {
            fail("rebuilt growing-input graph did not produce exactly CPU + Meta scheduler splits");
        }
    };

    std::printf("\n== %d layers, %lld tokens a half, hidden %lld, %lld rows a card, %zu cards; one %0.2f MB sum a layer\n", m.L,
                (long long) m.T, (long long) m.n, (long long) m.k, m.n_devs, m.n * m.T * 4 / 1e6);

    // reference
    zero_caches(m); set_inputs(); sequential(); sync();
    const std::vector<float> ref_a = read(a), ref_b = read(b);
    // again: the reference must reproduce itself
    zero_caches(m); set_inputs(); sequential(); sync();
    std::printf("sequential again: %zu + %zu values differ\n", differing(ref_a, read(a)), differing(ref_b, read(b)));

    zero_caches(m); set_inputs();
    if (!pair(a, b)) {
        std::printf("PAIR UNSUPPORTED: the backend declined the pair before computing (sum route or graph shape)\n");
        return 3;
    }
    sync();
    const size_t pair_a = differing(ref_a, read(a)), pair_b = differing(ref_b, read(b));
    std::printf("pair (A, B): %zu + %zu of %zu values differ from sequential\n", pair_a, pair_b, ref_a.size());

    zero_caches(m); set_inputs();
    pair(b, a);
    sync();
    const size_t wrong_b = differing(ref_b, read(b));
    std::printf("wrong order (B, A): half B differs in %zu values (must be more than 0)\n", wrong_b);

    // Consecutive-call correctness/lifetime stress. Sequential is synchronized per call; pair intentionally is not,
    // so the next CPU prefix and H2D input copies arrive while the previous pair can still be executing.
    const int stress_calls = std::max(32, 4 * reps);
    zero_caches(m);
    for (int c = 0; c < stress_calls; ++c) {
        set_inputs(c + 1);
        sequential();
        sync();
    }
    const std::vector<float> carry_ref_a = read(a), carry_ref_b = read(b);

    zero_caches(m);
    bool carry_supported = true;
    for (int c = 0; c < stress_calls; ++c) {
        set_inputs(c + 1);
        carry_supported = pair(a, b) && carry_supported;
    }
    sync();
    const size_t carry_a = differing(carry_ref_a, read(a));
    const size_t carry_b = differing(carry_ref_b, read(b));
    std::printf("carried-state stress: %d pairs, %zu + %zu values differ from sequential\n",
                stress_calls, carry_a, carry_b);
    if (!carry_supported) {
        std::printf("PAIR UNSUPPORTED during carried-state stress\n");
        return 3;
    }

    // Variable-size staging proof. Every iteration rebuilds both halves with a larger CPU->Meta input; the final
    // pair crosses the former 4 MiB gate. The schedulers were reserved at grow_max_elems before any graph allocation.
    size_t grow_diff_a = 0, grow_diff_b = 0;
    bool grow_supported = true;
    for (int c = 0; c < grow_steps; ++c) {
        sync();
        const int64_t elems = grow_max_elems * (c + 1) / grow_steps;
        rebuild_half(a, m.y_a, 0, m.T, elems, false);
        rebuild_half(b, m.y_b, m.T, 0, elems, true);

        zero_caches(m); set_inputs(2000 + c);
        sequential(); sync();
        const std::vector<float> grow_ref_a = read(a), grow_ref_b = read(b);

        zero_caches(m); set_inputs(2000 + c);
        grow_supported = pair(a, b) && grow_supported;
        sync();
        grow_diff_a += differing(grow_ref_a, read(a));
        grow_diff_b += differing(grow_ref_b, read(b));
    }
    std::printf("growing-input stress: %d pairs, max %.2f MiB/half, %zu + %zu values differ from sequential\n",
                grow_steps, grow_max_elems * sizeof(float) / 1024.0 / 1024.0, grow_diff_a, grow_diff_b);
    if (!grow_supported) {
        std::printf("PAIR UNSUPPORTED during growing-input stress\n");
        return 3;
    }
    if (grow_diff_a != 0 || grow_diff_b != 0) {
        return 5;
    }

    // Restore the small-input shape so the historical timing arm still measures the stagger mechanism rather than
    // the staging stress.
    sync();
    rebuild_half(a, m.y_a, 0, m.T, grow_base_elems, false);
    rebuild_half(b, m.y_b, m.T, 0, grow_base_elems, true);

    // timing
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = i < xa0.size() ? xa0[i] : xb0[i - xa0.size()];
    ggml_backend_tensor_set(full.input, xf.data(), 0, xf.size() * sizeof(float));
    for (int r = 0; r < reps; ++r) {
        zero_caches(m); set_inputs(1000 + r); sync();
        double t0 = now_ms();
        sequential(); sync();
        const double t_seq = now_ms() - t0;
        zero_caches(m); set_inputs(); sync();
        t0 = now_ms();
        pair(a, b); sync();
        const double t_pair = now_ms() - t0;
        t0 = now_ms();
        if (ggml_backend_sched_graph_compute_async(full.sched, full.graph) != GGML_STATUS_SUCCESS) {
            fail("full compute failed");
        }
        ggml_backend_sched_synchronize(full.sched);
        ggml_backend_synchronize(meta);
        const double t_full = now_ms() - t0;
        std::printf("  ms: full %.1f  halves in sequence %.1f  pair %.1f  (pair against full %+.1f%%, against sequence %+.1f%%)\n", t_full, t_seq,
                    t_pair, 100.0 * (t_pair - t_full) / t_full, 100.0 * (t_pair - t_seq) / t_seq);
        std::fflush(stdout);
    }

    if (pair_a != 0 || pair_b != 0) {
        return 1;
    }
    if (carry_a != 0 || carry_b != 0) {
        return 4;
    }
    return m.use_cache && wrong_b == 0 ? 2 : 0;
}
