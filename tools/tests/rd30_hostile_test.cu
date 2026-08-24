// Standalone real-hardware test of RD30's mmq_build_moe_block_map kernel
// against hostile MoE routing distributions (RD94/EC13 concern), at real
// production scale (n_experts=256). Verbatim copy of the kernel from
// patches/1237_rd30_moe_mmq_compact_grid.py -- if this file and the patch
// ever diverge, the patch is authoritative and this file is stale.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

static __global__ void mmq_build_moe_block_map(
        const int32_t * __restrict__ expert_bounds,
        const int n_experts,
        const int J,
        int32_t * __restrict__ block_start,
        int32_t * __restrict__ block_expert) {

    extern __shared__ int32_t rd30_s_start[];
    const int tid = threadIdx.x;

    for (int e = tid; e < n_experts; e += blockDim.x) {
        const int count = expert_bounds[e + 1] - expert_bounds[e];
        rd30_s_start[e] = (count + J - 1) / J;
    }
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int count = rd30_s_start[e];
            rd30_s_start[e] = total;
            total += count;
        }
        rd30_s_start[n_experts] = total;
    }
    __syncthreads();

    for (int e = tid; e <= n_experts; e += blockDim.x) {
        block_start[e] = rd30_s_start[e];
    }

    const int total = rd30_s_start[n_experts];

    for (int m = tid; m < total; m += blockDim.x) {
        int lo = 0;
        int hi = n_experts;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (rd30_s_start[mid] <= m) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        block_expert[m] = lo - 1;
    }
}

struct Case {
    const char * name;
    std::vector<int32_t> expert_bounds; // n_experts+1, exclusive prefix sum of token counts
};

static void reference(const std::vector<int32_t> & expert_bounds, int n_experts, int J,
                       std::vector<int32_t> & block_start, std::vector<int32_t> & block_expert) {
    block_start.assign(n_experts + 1, 0);
    int total = 0;
    for (int e = 0; e < n_experts; ++e) {
        int count = expert_bounds[e + 1] - expert_bounds[e];
        int tiles = (count + J - 1) / J;
        block_start[e] = total;
        total += tiles;
    }
    block_start[n_experts] = total;
    block_expert.assign(total, -1);
    for (int e = 0; e < n_experts; ++e) {
        for (int m = block_start[e]; m < block_start[e + 1]; ++m) {
            block_expert[m] = e;
        }
    }
}

int main() {
    const int n_experts = 256;
    const int J = 8; // representative real MMQ tile width

    std::vector<Case> cases;

    // single-hot: every token routed to expert 0, all others get zero.
    {
        std::vector<int32_t> eb(n_experts + 1, 0);
        for (int e = 1; e <= n_experts; ++e) eb[e] = 4096; // expert 0 absorbs everything
        cases.push_back({"single-hot", eb});
    }

    // concentrated: only 8 of 256 experts active, evenly loaded, rest zero.
    {
        std::vector<int32_t> eb(n_experts + 1, 0);
        int per = 512;
        int cum = 0;
        for (int e = 0; e < n_experts; ++e) {
            eb[e] = cum;
            if (e < 8) cum += per;
        }
        eb[n_experts] = cum;
        cases.push_back({"concentrated-8of256", eb});
    }

    // zipf-ish skew: expert e gets weight ~ 1/(e+1), heavily front-loaded.
    {
        std::vector<int32_t> eb(n_experts + 1, 0);
        int cum = 0;
        for (int e = 0; e < n_experts; ++e) {
            eb[e] = cum;
            int w = 2048 / (e + 1);
            cum += w;
        }
        eb[n_experts] = cum;
        cases.push_back({"zipf-skew", eb});
    }

    // uniform: every expert gets exactly the same tiny count.
    {
        std::vector<int32_t> eb(n_experts + 1, 0);
        for (int e = 0; e <= n_experts; ++e) eb[e] = e * 4;
        cases.push_back({"uniform", eb});
    }

    // adversarial-empty: every expert has zero tokens (degenerate, e.g. an
    // empty ubatch) -- total = 0, block_expert should be an empty array and
    // nothing should read out of bounds.
    {
        std::vector<int32_t> eb(n_experts + 1, 0);
        cases.push_back({"all-zero", eb});
    }

    int failures = 0;

    for (auto & c : cases) {
        std::vector<int32_t> ref_start, ref_expert;
        reference(c.expert_bounds, n_experts, J, ref_start, ref_expert);
        int total = ref_start[n_experts];

        int32_t * d_expert_bounds;
        int32_t * d_block_start;
        int32_t * d_block_expert;
        HIP_CHECK(hipMalloc(&d_expert_bounds, sizeof(int32_t) * (n_experts + 1)));
        HIP_CHECK(hipMalloc(&d_block_start, sizeof(int32_t) * (n_experts + 1)));
        HIP_CHECK(hipMalloc(&d_block_expert, sizeof(int32_t) * (total > 0 ? total : 1)));
        HIP_CHECK(hipMemcpy(d_expert_bounds, c.expert_bounds.data(),
                             sizeof(int32_t) * (n_experts + 1), hipMemcpyHostToDevice));

        size_t smem = sizeof(int32_t) * (n_experts + 1);
        mmq_build_moe_block_map<<<1, 256, smem>>>(d_expert_bounds, n_experts, J,
                                                    d_block_start, d_block_expert);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<int32_t> got_start(n_experts + 1);
        std::vector<int32_t> got_expert(total > 0 ? total : 1);
        HIP_CHECK(hipMemcpy(got_start.data(), d_block_start,
                             sizeof(int32_t) * (n_experts + 1), hipMemcpyDeviceToHost));
        if (total > 0) {
            HIP_CHECK(hipMemcpy(got_expert.data(), d_block_expert,
                                 sizeof(int32_t) * total, hipMemcpyDeviceToHost));
        }

        bool ok = (got_start == ref_start);
        if (ok) {
            for (int i = 0; i < total; ++i) {
                if (got_expert[i] != ref_expert[i]) { ok = false; break; }
            }
        }

        printf("%-24s n_experts=%d total_blocks=%-6d %s\n", c.name, n_experts, total,
               ok ? "OK" : "MISMATCH");
        if (!ok) {
            failures++;
            for (int i = 0; i < n_experts + 1 && i < 10; ++i) {
                printf("  block_start[%d] got=%d ref=%d\n", i, got_start[i], ref_start[i]);
            }
        }

        HIP_CHECK(hipFree(d_expert_bounds));
        HIP_CHECK(hipFree(d_block_start));
        HIP_CHECK(hipFree(d_block_expert));
    }

    printf("\n%d/%zu cases failed\n", failures, cases.size());
    return failures == 0 ? 0 : 1;
}
