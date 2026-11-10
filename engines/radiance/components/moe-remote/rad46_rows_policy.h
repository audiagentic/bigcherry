#pragma once
#include <cstdint>

// RAD46: observed median kernel latency, in tenths of a microsecond,
// 6900 XT / one production-width rank / five interleaved repeats.
//
// Columns at 64 rows: req-18-3435 and req-18-3439.
// Columns at 2048 rows: req-18-7, req-18-14 and req-18-21.
// The accurate quad uses 16 rows/batch for both stages. The row count is
// already available in the dispatch; no request-specific probing is needed.
namespace r3_moe_gpu {
enum class Rad46RowsForm { OutputLane, Quad16, Quad8, RowLane16, RowLane8, StockReplay };
enum class Rad46Sweep { Production, Stock, OutputLane, Quad, QuadV2, RowLane16, RowLane8, Split };

constexpr int kRad46SmallRows = 64;
constexpr int kRad46LargeRows = 2048;
// At the small endpoint quad is slower; at the large endpoint quad is faster.
// Store every observed difference (quad minus output-lane), with a single
// averaging step per endpoint, and interpolate the signed difference to zero.
constexpr int kGuSmallDiff = ((1874 - 862) + (1248 - 880)) / 2;
constexpr int kGuLargeDiff = ((4997 - 4825) + (12174 - 22588) + (10992 - 15438)) / 3;
constexpr int kDnSmallDiff = ((1001 - 550) + (633 - 372)) / 2;
constexpr int kDnLargeDiff = ((2525 - 2026) + (5229 - 7081) + (4972 - 5020)) / 3;
constexpr int rad46_crossover(int small_difference, int large_difference) {
    return kRad46SmallRows + (kRad46LargeRows - kRad46SmallRows) *
        small_difference / (small_difference - large_difference);
}
static_assert(kGuSmallDiff > 0 && kGuLargeDiff < 0 &&
              kDnSmallDiff > 0 && kDnLargeDiff < 0, "the measured forms must cross");
constexpr uint32_t kRad46GuCrossover = rad46_crossover(kGuSmallDiff, kGuLargeDiff);
constexpr uint32_t kRad46DnCrossover = rad46_crossover(kDnSmallDiff, kDnLargeDiff);
static_assert(kRad46GuCrossover > kRad46SmallRows &&
              kRad46GuCrossover < kRad46LargeRows &&
              kRad46DnCrossover > kRad46SmallRows &&
              kRad46DnCrossover < kRad46LargeRows, "crossover must interpolate measured sizes");

constexpr Rad46RowsForm rad46_rows_form(Rad46Sweep sweep, bool gate_up, uint32_t rows) {
    switch (sweep) {
    case Rad46Sweep::Stock: return Rad46RowsForm::StockReplay;
    case Rad46Sweep::OutputLane: return Rad46RowsForm::OutputLane;
    case Rad46Sweep::Quad:
    case Rad46Sweep::Split: return Rad46RowsForm::Quad16;
    case Rad46Sweep::QuadV2: return gate_up ? Rad46RowsForm::Quad8 : Rad46RowsForm::Quad16;
    case Rad46Sweep::RowLane16:
        if (gate_up) return Rad46RowsForm::RowLane16;
        break;
    case Rad46Sweep::RowLane8:
        if (gate_up) return Rad46RowsForm::RowLane8;
        break;
    case Rad46Sweep::Production: break;
    }
    return rows <= (gate_up ? kRad46GuCrossover : kRad46DnCrossover)
        ? Rad46RowsForm::OutputLane : Rad46RowsForm::Quad16;
}
} // namespace r3_moe_gpu
