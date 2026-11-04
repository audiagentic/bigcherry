#include "rad46_rows_policy.h"
#include <cassert>
#include <initializer_list>

int main() {
    using namespace r3_moe_gpu;
    static_assert(kRad46GuCrossover == 309);
    static_assert(kRad46DnCrossover == 922);
    static_assert(rad46_rows_form(Rad46Sweep::Production, true, 64) == Rad46RowsForm::OutputLane);
    static_assert(rad46_rows_form(Rad46Sweep::Production, false, 64) == Rad46RowsForm::OutputLane);
    static_assert(rad46_rows_form(Rad46Sweep::Production, true, 2048) == Rad46RowsForm::Quad16);
    static_assert(rad46_rows_form(Rad46Sweep::Production, false, 2048) == Rad46RowsForm::Quad16);
    for (bool gate : {false, true}) {
        const auto crossover = gate ? kRad46GuCrossover : kRad46DnCrossover;
        assert(rad46_rows_form(Rad46Sweep::Production, gate, crossover) == Rad46RowsForm::OutputLane);
        assert(rad46_rows_form(Rad46Sweep::Production, gate, crossover + 1) == Rad46RowsForm::Quad16);
        for (auto rows : {1u, 64u, 128u, 512u, 2048u, 4096u}) {
            assert(rad46_rows_form(Rad46Sweep::Stock, gate, rows) == Rad46RowsForm::StockReplay);
            assert(rad46_rows_form(Rad46Sweep::OutputLane, gate, rows) == Rad46RowsForm::OutputLane);
            assert(rad46_rows_form(Rad46Sweep::Quad, gate, rows) == Rad46RowsForm::Quad16);
            assert(rad46_rows_form(Rad46Sweep::QuadV2, gate, rows) ==
                   (gate ? Rad46RowsForm::Quad8 : Rad46RowsForm::Quad16));
            assert(rad46_rows_form(Rad46Sweep::RowLane16, gate, rows) ==
                   (gate ? Rad46RowsForm::RowLane16 : rad46_rows_form(Rad46Sweep::Production, false, rows)));
            assert(rad46_rows_form(Rad46Sweep::RowLane8, gate, rows) ==
                   (gate ? Rad46RowsForm::RowLane8 : rad46_rows_form(Rad46Sweep::Production, false, rows)));
        }
    }
}
