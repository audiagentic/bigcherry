// QFP49/QFP41: prefill of the real model through the llama library alone - no server or sampling.
//
// Loads the target model tensor-split across the given cards, feeds it N token ids in calls of -b tokens, waits for
// the target cards, and reports wall time plus the TARGET last-token logits. Optional --draft-model loads a second
// llama_model/llama_context on --draft-dev and, immediately after every target llama_decode returns, submits
// --draft-n token ids to that context on the main thread. This reproduces the server's post-target hook timing without
// adding a harness thread; it is default-off, so all existing invocations retain their old behavior.
//
// Usage: llama_prefill_harness -m <gguf> [-n tokens=4096] [-b 512] [-ub 256] [-c 65536] [-dev ROCm0,ROCm1,ROCm2]
//   [-ts 0.31,0.27,0.42] [-ot '<regex>=CPU' ...] [--calls K] [--tokens file] [-o logits.bin]
//   [--draft-model <gguf> --draft-dev ROCm3 --draft-n N [--draft-ctx C]]
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<std::string> split(const std::string & text, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(sep, start);
        out.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_path, devs = "ROCm0,ROCm1,ROCm2", ts = "0.31,0.27,0.42", out_path, tokens_path;
    std::string draft_model_path, draft_dev = "ROCm3";
    std::vector<std::string> overrides;
    int n_tokens = 4096, n_batch = 512, n_ubatch = 256, n_ctx = 65536, max_calls = 0;
    int draft_n = 1, draft_ctx_size = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(10);
            }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-n") n_tokens = std::atoi(next().c_str());
        else if (a == "-b") n_batch = std::atoi(next().c_str());
        else if (a == "-ub") n_ubatch = std::atoi(next().c_str());
        else if (a == "-c") n_ctx = std::atoi(next().c_str());
        else if (a == "-dev") devs = next();
        else if (a == "-ts") ts = next();
        else if (a == "-ot") overrides.push_back(next());
        else if (a == "--calls") max_calls = std::atoi(next().c_str());
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--draft-model") draft_model_path = next();
        else if (a == "--draft-dev") draft_dev = next();
        else if (a == "--draft-n") draft_n = std::atoi(next().c_str());
        else if (a == "--draft-ctx") draft_ctx_size = std::atoi(next().c_str());
        else if (a == "-o") out_path = next();
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 10;
        }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: llama_prefill_harness -m <gguf> [-n tokens] [-b 512] [-ub 256] [-c ctx] [-dev a,b,c] [-ts x,y,z] [-ot 're=CPU'] [--calls K] [-o logits.bin] [--draft-model gguf --draft-dev ROCm3 --draft-n N --draft-ctx C]\n");
        return 10;
    }

    llama_backend_init();

    std::vector<ggml_backend_dev_t> devices;
    for (const std::string & name : split(devs, ',')) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr) {
            std::fprintf(stderr, "no device named %s\n", name.c_str());
            return 10;
        }
        devices.push_back(dev);
    }
    devices.push_back(nullptr);
    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    {
        const std::vector<std::string> parts = split(ts, ',');
        for (size_t i = 0; i < parts.size() && i < tensor_split.size(); ++i) {
            tensor_split[i] = std::strtof(parts[i].c_str(), nullptr);
        }
    }
    // -ot '<regex>=CPU': the server's placement override, CPU only (what the lab profiles use)
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> buft_overrides;
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    for (const std::string & o : overrides) {
        const size_t eq = o.rfind('=');
        if (eq == std::string::npos || o.substr(eq + 1) != "CPU" || cpu_dev == nullptr) {
            std::fprintf(stderr, "override must be '<regex>=CPU': %s\n", o.c_str());
            return 10;
        }
        patterns.push_back(o.substr(0, eq));
    }
    for (const std::string & p : patterns) {
        buft_overrides.push_back({ p.c_str(), ggml_backend_dev_buffer_type(cpu_dev) });
    }
    buft_overrides.push_back({ nullptr, nullptr });

    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devices.data();
    mparams.n_gpu_layers = 99;
    mparams.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mparams.tensor_split = tensor_split.data();
    mparams.tensor_buft_overrides = patterns.empty() ? nullptr : buft_overrides.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "could not load the model\n");
        return 11;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) n_batch;
    cparams.n_ubatch = (uint32_t) n_ubatch;
    cparams.n_threads = 16;
    cparams.n_threads_batch = 16;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "could not create the context\n");
        return 11;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_model * draft_model = nullptr;
    llama_context * draft_ctx = nullptr;
    int draft_vocab = 0;
    if (!draft_model_path.empty()) {
        if (draft_n <= 0) {
            std::fprintf(stderr, "--draft-n must be > 0\n");
            return 10;
        }
        ggml_backend_dev_t draft_backend_dev = ggml_backend_dev_by_name(draft_dev.c_str());
        if (draft_backend_dev == nullptr) {
            std::fprintf(stderr, "no draft device named %s\n", draft_dev.c_str());
            return 10;
        }
        std::vector<ggml_backend_dev_t> draft_devices = {draft_backend_dev, nullptr};
        llama_model_params dmparams = llama_model_default_params();
        dmparams.devices = draft_devices.data();
        dmparams.n_gpu_layers = 99;
        dmparams.split_mode = LLAMA_SPLIT_MODE_NONE;
        draft_model = llama_model_load_from_file(draft_model_path.c_str(), dmparams);
        if (draft_model == nullptr) {
            std::fprintf(stderr, "could not load the draft model\n");
            return 11;
        }

        if (draft_ctx_size <= 0) {
            draft_ctx_size = n_ctx;
        }
        llama_context_params dcparams = llama_context_default_params();
        dcparams.n_ctx = (uint32_t) draft_ctx_size;
        dcparams.n_batch = (uint32_t) draft_n;
        dcparams.n_ubatch = (uint32_t) draft_n;
        dcparams.n_threads = 16;
        dcparams.n_threads_batch = 16;
        dcparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        dcparams.type_k = GGML_TYPE_F16;
        dcparams.type_v = GGML_TYPE_F16;
        draft_ctx = llama_init_from_model(draft_model, dcparams);
        if (draft_ctx == nullptr) {
            std::fprintf(stderr, "could not create the draft context\n");
            return 11;
        }
        draft_vocab = llama_vocab_n_tokens(llama_model_get_vocab(draft_model));
    }

    std::vector<llama_token> tokens;
    if (!tokens_path.empty()) {
        std::ifstream f(tokens_path);
        for (long long id; f >> id;) {
            tokens.push_back((llama_token) id);
        }
    } else {
        uint64_t state = 0x9e3779b97f4a7c15ull;  // fixed sequence, the same in every run
        for (int i = 0; i < n_tokens; ++i) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            tokens.push_back((llama_token) (1000 + (state >> 33) % (uint64_t) (n_vocab > 61000 ? 60000 : n_vocab - 1000)));
        }
    }
    if (max_calls > 0 && (size_t) max_calls * (size_t) n_batch < tokens.size()) {
        tokens.resize((size_t) max_calls * (size_t) n_batch);
    }
    const size_t target_calls = (tokens.size() + (size_t) n_batch - 1) / (size_t) n_batch;
    if (draft_ctx != nullptr && target_calls * (size_t) draft_n > (size_t) draft_ctx_size) {
        std::fprintf(stderr, "draft workload needs %zu positions but --draft-ctx is %d\n",
                     target_calls * (size_t) draft_n, draft_ctx_size);
        return 10;
    }

    std::printf("\n== prefill %zu tokens, -b %d -ub %d, ctx %d, cards %s split %s\n", tokens.size(), n_batch, n_ubatch, n_ctx,
                devs.c_str(), ts.c_str());
    if (draft_ctx != nullptr) {
        std::printf("== interleave draft %s on %s: %d tokens after each target decode, ctx %d\n",
                    draft_model_path.c_str(), draft_dev.c_str(), draft_n, draft_ctx_size);
    }
    std::vector<double> held;
    std::vector<double> draft_held;
    std::vector<llama_token> draft_tokens((size_t) std::max(draft_n, 1));
    uint64_t draft_state = 0xd1b54a32d192ed03ull;
    const double t0 = now_ms();
    for (size_t i = 0; i < tokens.size(); i += (size_t) n_batch) {
        const int n = (int) std::min((size_t) n_batch, tokens.size() - i);
        const double c0 = now_ms();
        const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n));
        held.push_back(now_ms() - c0);
        if (rc != 0) {
            std::fprintf(stderr, "llama_decode failed with %d at token %zu\n", rc, i);
            return 12;
        }
        if (draft_ctx != nullptr) {
            const int base = draft_vocab > 1024 ? 1000 : 0;
            const int span = std::max(1, draft_vocab - base);
            for (int j = 0; j < draft_n; ++j) {
                draft_state = draft_state * 6364136223846793005ull + 1442695040888963407ull;
                draft_tokens[(size_t) j] = (llama_token) (base + (int) ((draft_state >> 33) % (uint64_t) span));
            }
            const double d0 = now_ms();
            const int drc = llama_decode(draft_ctx, llama_batch_get_one(draft_tokens.data(), draft_n));
            draft_held.push_back(now_ms() - d0);
            if (drc != 0) {
                std::fprintf(stderr, "draft llama_decode failed with %d after target token %zu\n", drc, i);
                return 12;
            }
        }
    }
    const double t_submitted = now_ms() - t0;
    llama_synchronize(ctx);
    const float * logits = llama_get_logits_ith(ctx, -1);
    const double t_total = now_ms() - t0;
    if (draft_ctx != nullptr) {
        llama_synchronize(draft_ctx);
    }

    double sum_held = 0.0, max_held = 0.0;
    for (double h : held) {
        sum_held += h;
        max_held = std::max(max_held, h);
    }
    std::printf("wall %.1f ms = %.1f t/s; llama_decode held the host %.1f ms in %zu calls (mean %.2f, max %.2f), "
                "returned from the last call at %.1f ms\n", t_total, 1000.0 * tokens.size() / t_total, sum_held, held.size(),
                sum_held / held.size(), max_held, t_submitted);
    std::printf("host ms a call:");
    for (size_t i = 0; i < held.size() && i < 24; ++i) {
        std::printf(" %.1f", held[i]);
    }
    std::printf("%s\n", held.size() > 24 ? " ..." : "");
    if (!draft_held.empty()) {
        double draft_sum = 0.0, draft_max = 0.0;
        for (double h : draft_held) {
            draft_sum += h;
            draft_max = std::max(draft_max, h);
        }
        std::printf("draft llama_decode held the main thread %.1f ms in %zu calls (mean %.2f, max %.2f)\n",
                    draft_sum, draft_held.size(), draft_sum / draft_held.size(), draft_max);
    }

    if (logits == nullptr) {
        std::fprintf(stderr, "no logits\n");
        return 13;
    }
    uint64_t hash = 1469598103934665603ull;  // FNV-1a over the raw bytes
    const unsigned char * bytes = (const unsigned char *) logits;
    for (size_t i = 0; i < (size_t) n_vocab * sizeof(float); ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    std::printf("logits of the last token: fnv %016llx, arg-max %d (%.6f), %d values\n", (unsigned long long) hash, best,
                logits[best], n_vocab);
    if (!out_path.empty()) {
        std::ofstream f(out_path, std::ios::binary);
        f.write((const char *) logits, (std::streamsize) ((size_t) n_vocab * sizeof(float)));
    }
    if (draft_ctx != nullptr) {
        llama_free(draft_ctx);
        llama_model_free(draft_model);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
