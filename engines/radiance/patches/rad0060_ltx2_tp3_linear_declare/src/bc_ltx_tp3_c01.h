/* BC_LTX3_C01_HEADER: experimental LTX2 IQ8R linear span declaration.
 * Builder-only geometry. Never enables multi-rank LTX execution by itself.
 * Shards are in logical model rows/K; actual three-rank execution also needs
 * global Q/K RMS, collectives, conditioning and output ownership.
 */
#pragma once
#include <cstdint>
#include <limits>
namespace ltx2 {
struct C01Slice {
    enum class Mode { Rows, Columns };
    Mode mode;
    int64_t source_width; // whole fused bank (Rows), or whole input K (Columns)
    int64_t offset;       // rank's start inside each bank or input K
    int64_t width;        // local rows per fused bank, or local input K
    int64_t head_dim;     // Rows: whole-head test; Columns: 0
    int banks;            // Rows: 2 or 3; Columns: 0

    static constexpr C01Slice rows(int64_t full_bank, int64_t start, int64_t local,
                                    int64_t hd, int count) {
        return {Mode::Rows, full_bank, start, local, hd, count};
    }
    static constexpr C01Slice columns(int64_t full_k, int64_t start, int64_t local) {
        return {Mode::Columns, full_k, start, local, 0, 0};
    }
    constexpr bool valid(int64_t n_out, int64_t n_in) const {
        // C01 is deliberately only IQ8R's 256-K group and H128 transform.
        if (n_out <= 0 || n_in <= 0 || source_width <= 0 ||
            offset < 0 || width <= 0 || offset > source_width ||
            width > source_width - offset)
            return false;
        if (mode == Mode::Rows)
            return (banks == 2 || banks == 3) && head_dim > 0 &&
                   width <= std::numeric_limits<int64_t>::max() / banks &&
                   source_width % head_dim == 0 && offset % head_dim == 0 &&
                   width % head_dim == 0 && n_out == banks * width &&
                   n_in % 256 == 0;
        return mode == Mode::Columns && banks == 0 && head_dim == 0 &&
               source_width % 256 == 0 && offset % 256 == 0 &&
               width % 256 == 0 && n_in == width;
    }
};
} // namespace ltx2
