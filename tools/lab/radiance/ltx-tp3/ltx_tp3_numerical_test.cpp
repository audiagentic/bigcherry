// Deterministic C++20 card-free LTX TP3 attention data-path oracle.
// This simulates fused QKV bank ownership, all-head Q/K RMS, attention and
// one output reduction with uneven ranks. NOT a HIP kernel or BF16 parity test.
#include "ltx_tp3_reference.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace bc_ltx_tp3_reference;
using Vec = std::vector<double>;
using Matrix = std::vector<Vec>;

constexpr int kHeads = 32;
constexpr int kHeadDim = 4;   // Deliberately tiny, card-free mathematical shape.
constexpr int kWidth = kHeads * kHeadDim;
constexpr int kInputWidth = 9;
constexpr int kOutWidth = 13;
constexpr double kEps = 1e-6;

double value(int i, int j, int phase) {
    return (double((i * 31 + j * 17 + phase * 11) % 71) - 31.0) / 37.0;
}
Matrix matrix(int rows, int cols, int phase) {
    Matrix v(static_cast<std::size_t>(rows), Vec(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value(i, j, phase);
    return v;
}
Vec row(int len, int phase) {
    Vec v(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) v[static_cast<std::size_t>(i)] = value(0, i, phase);
    return v;
}
Matrix project(const Matrix& x, const Matrix& w, const Vec& bias) {
    assert(!w.empty() && w.size() == bias.size());
    Matrix y(x.size(), Vec(bias.size()));
    for (std::size_t t = 0; t < x.size(); ++t) {
        for (std::size_t j = 0; j < w.size(); ++j) {
            double sum = bias[j];
            for (std::size_t i = 0; i < x[t].size(); ++i) sum += w[j][i] * x[t][i];
            y[t][j] = sum;
        }
    }
    return y;
}
Matrix bank(const Matrix& x, int offset, int width) {
    Matrix y;
    for (const Vec& r : x)
        y.emplace_back(r.begin() + offset, r.begin() + offset + width);
    return y;
}
Matrix normalize_dense(const Matrix& raw, const Vec& gain) {
    Matrix y = raw;
    for (Vec& r : y) {
        double sq = 0.0;
        for (double x : r) sq += x * x;
        const double scale = 1.0 / std::sqrt(sq / double(r.size()) + kEps);
        for (std::size_t i = 0; i < r.size(); ++i) r[i] *= scale * gain[i];
    }
    return y;
}
Matrix normalize_tp(const Matrix& raw, const Vec& gain,
                    const std::array<Slice, 3>& slices) {
    Matrix result(raw.size(), Vec(kWidth, 0.0));
    for (std::size_t r = 0; r < raw.size(); ++r) {
        std::array<double, 3> local{};
        for (int rank = 0; rank < 3; ++rank) {
            for (int i = slices[rank].offset; i < slices[rank].offset+slices[rank].width; ++i)
                local[rank] += raw[r][i] * raw[r][i];
        }
        // Global width, even when the ranks have different head counts.
        const double scale = 1.0 / std::sqrt((local[0]+local[1]+local[2]) /
                                              double(kWidth) + kEps);
        for (int rank = 0; rank < 3; ++rank)
            for (int i = slices[rank].offset; i < slices[rank].offset+slices[rank].width; ++i)
                result[r][i] = raw[r][i] * scale * gain[i];
    }
    return result;
}
double activation_gate(double x) { return 2.0 / (1.0 + std::exp(-x)); }

Matrix dense_attention(const Matrix& q, const Matrix& k, const Matrix& v,
                       const Matrix& gates, const Matrix& wout, const Vec& bout) {
    const int nq = static_cast<int>(q.size()), nk = static_cast<int>(k.size());
    Matrix attn(nq, Vec(kWidth, 0.0));
    for (int t = 0; t < nq; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            Vec logits(nk);
            for (int s = 0; s < nk; ++s) {
                for (int d = 0; d < kHeadDim; ++d)
                    logits[s] += q[t][h*kHeadDim+d] * k[s][h*kHeadDim+d];
                logits[s] *= 1.0/std::sqrt(double(kHeadDim));
            }
            const double maximum = *std::max_element(logits.begin(), logits.end());
            double denom = 0;
            for (double& x : logits) { x = std::exp(x-maximum); denom += x; }
            for (int d = 0; d < kHeadDim; ++d)
                for (int s = 0; s < nk; ++s)
                    attn[t][h*kHeadDim+d] += logits[s]/denom * v[s][h*kHeadDim+d];
            const double g = activation_gate(gates[t][h]);
            for (int d = 0; d < kHeadDim; ++d) attn[t][h*kHeadDim+d] *= g;
        }
    }
    return project(attn, wout, bout);
}

Matrix sharded_attention(const Matrix& q, const Matrix& k, const Matrix& v,
                         const Matrix& gates, const Matrix& wout, const Vec& bout,
                         const std::array<Slice, 3>& slices) {
    const int nq = static_cast<int>(q.size()), nk = static_cast<int>(k.size());
    Matrix output(nq, bout);     // Exactly one output bias, not once per rank.
    for (int rank = 0; rank < 3; ++rank) {
        const int hs = slices[rank].offset/kHeadDim, count = slices[rank].width/kHeadDim;
        Matrix local(nq, Vec(static_cast<std::size_t>(count*kHeadDim), 0.0));
        for (int t = 0; t < nq; ++t)
            for (int h = hs; h < hs+count; ++h) {
                Vec score(nk);
                for (int s = 0; s < nk; ++s)
                    for (int d = 0; d < kHeadDim; ++d)
                        score[s] += q[t][h*kHeadDim+d] * k[s][h*kHeadDim+d] /
                                    std::sqrt(double(kHeadDim));
                const double maximum = *std::max_element(score.begin(), score.end());
                double total = 0;
                for (double& e : score) { e = std::exp(e-maximum); total += e; }
                for (int d = 0; d < kHeadDim; ++d) {
                    double head = 0;
                    for (int s = 0; s < nk; ++s)
                        head += score[s]/total * v[s][h*kHeadDim+d];
                    local[t][(h-hs)*kHeadDim+d] =
                        head * activation_gate(gates[t][h]);
                }
            }
        // Row-parallel output projection: each rank contributes a partial
        // with K-local weights and no output bias; reduction is global.
        for (int t = 0; t < nq; ++t)
            for (int out = 0; out < kOutWidth; ++out)
                for (int i = 0; i < slices[rank].width; ++i)
                    output[t][out] += wout[out][slices[rank].offset+i]*local[t][i];
    }
    return output;
}

void near(const Matrix& a, const Matrix& b, double tolerance) {
    assert(a.size()==b.size());
    for (std::size_t i=0;i<a.size();++i) {
        assert(a[i].size()==b[i].size());
        for (std::size_t j=0;j<a[i].size();++j)
            assert(std::fabs(a[i][j]-b[i][j])<tolerance);
    }
}
void test_attention(bool self, const Shares& shares) {
    const auto slices = attention_slices(kHeads,kHeadDim,shares);
    const int nq=3, nk=self?3:5;
    const Matrix input=matrix(nq,kInputWidth,5);
    const Matrix context=self?input:matrix(nk,kInputWidth,17);
    const Matrix fused_w=matrix(3*kWidth,kInputWidth,29);
    const Vec fused_b=row(3*kWidth,43);
    const Matrix q_w=matrix(kWidth,kInputWidth,31);
    const Vec q_b=row(kWidth,47);
    const Matrix kv_w=matrix(2*kWidth,kInputWidth,37);
    const Vec kv_b=row(2*kWidth,53);
    const Matrix projection_self=project(input,fused_w,fused_b);
    const Matrix projection_q=project(input,q_w,q_b);
    const Matrix projection_kv=project(context,kv_w,kv_b);
    const Matrix q=self?bank(projection_self,0,kWidth):projection_q;
    const Matrix k=self?bank(projection_self,kWidth,kWidth):bank(projection_kv,0,kWidth);
    const Matrix v=self?bank(projection_self,2*kWidth,kWidth):bank(projection_kv,kWidth,kWidth);

    if(self) {
        for (const Slice s:slices) {
            const auto spans=fused_qkv_pieces(kWidth,s);
            for (int t=0;t<nq;++t) {
                for (int bankid=0;bankid<3;++bankid)
                    for (int i=0;i<s.width;++i)
                        assert(projection_self[t][spans[bankid].source_row+i] ==
                               (bankid==0?q[t][s.offset+i]:
                                bankid==1?k[t][s.offset+i]:v[t][s.offset+i]));
            }
        }
    }
    const Vec q_gain=row(kWidth,61), k_gain=row(kWidth,67);
    const auto nq_dense=normalize_dense(q,q_gain);
    const auto nk_dense=normalize_dense(k,k_gain);
    const auto nq_tp=normalize_tp(q,q_gain,slices);
    const auto nk_tp=normalize_tp(k,k_gain,slices);
    near(nq_dense,nq_tp,1e-12);
    near(nk_dense,nk_tp,1e-12);
    const Matrix gates=matrix(nq,kHeads,73);
    const Matrix wo=matrix(kOutWidth,kWidth,79);
    const Vec bo=row(kOutWidth,83);
    const Matrix dense=dense_attention(nq_dense,nk_dense,v,gates,wo,bo);
    const Matrix sharded=sharded_attention(nq_tp,nk_tp,v,gates,wo,bo,slices);
    near(dense,sharded,1e-10);
    // Negative control: add output projection bias on all three ranks.
    for (int t=0;t<nq;++t) for(int j=0;j<kOutWidth;++j)
        if (std::fabs(bo[j])>1e-4) {
            assert(std::fabs(dense[t][j]-(sharded[t][j]+2*bo[j]))>1e-4);
            break;
        }
}
} // namespace

int main() {
    for(const Shares s : {Shares{8,8,16}, Shares{12,12,8}, Shares{10,10,12}}) {
        test_attention(true,s);
        test_attention(false,s);
    }
    std::cout << "LTX TP3 multirow self/cross attention numerical oracle: 6 cases passed\n";
}
