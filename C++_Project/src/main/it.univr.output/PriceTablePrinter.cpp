/*
 * PriceTablePrinter.cpp
 * Price table and exercise-region matrix printed on the console (and shown by the interface).
 *
 * Author: Alberto Oliva Medin
 */
#include "PriceTablePrinter.h"
#include "it.univr.utils/TensorUtils.h"
#include <cstdio>
#include <vector>

namespace it::univr::output {

using it::univr::utils::toVector;

static const char* kLine =
    "---------------------------------------------------------------------------------------------------------";

// Prints a grid (rows x columns) of numbers with row and column labels
static void printGrid(const char* corner, const std::vector<double>& rowVals, const std::vector<double>& colVals,
                      const std::vector<double>& values, bool asInteger) {
    std::printf("%s\t", corner);
    for (double c : colVals) std::printf("%.2f\t", c);
    std::printf("\n%s\n", kLine);
    const size_t nc = colVals.size();
    for (size_t i = 0; i < rowVals.size(); ++i) {
        std::printf("%.2f |\t", rowVals[i]);
        for (size_t j = 0; j < nc; ++j) {
            if (asInteger) std::printf("%d\t", static_cast<int>(values[i * nc + j]));
            else           std::printf("%.4f\t", values[i * nc + j]);
        }
        std::printf("\n");
    }
    std::printf("%s\n", kLine);
}

// d = 1: price P(S, t) on a grid, rows S and columns t, followed by the 0/1 exercise matrix.
void PriceTablePrinter::print1D(neuralnetworks::PINN& pinn, double sMin, double sMax, int sSteps,
                                int tSteps, long long seed, int epochs) {
    const double T = pinn.getOption()->getMaturity();
    auto Saxis = torch::linspace(sMin, sMax, sSteps + 1);
    auto taxis = torch::linspace(0.0, T, tSteps + 1);
    auto grid = torch::meshgrid({Saxis, taxis}, "ij");    // rows: S, columns: t
    auto S = grid[0].reshape({-1, 1});
    auto t = grid[1].reshape({-1, 1});

    auto price = toVector(pinn.price(S, t));
    auto exercise = toVector(pinn.isInContinuationRegion(S, t).logical_not());

    std::printf("\nPrice Table (Seed %lld, epochs %d) - Rows: Asset Price S, Columns: Time t from 0 to %.2f\n",
                seed, epochs, T);
    printGrid("S/t", toVector(Saxis), toVector(taxis), price, false);
    std::printf("\nExercise Region Matrix (1 = Early Exercise, 0 = Continuation Region)\n");
    printGrid("S/t", toVector(Saxis), toVector(taxis), exercise, true);
}

// d = 2: price P(S1, S2, t) at a fixed time, rows S1 and columns S2, with the estimated boundary.
void PriceTablePrinter::print2D(neuralnetworks::PINN& pinn, double sMin, double sMax, int sSteps,
                                double t, long long seed, int epochs) {
    auto axis = torch::linspace(sMin, sMax, sSteps + 1);
    auto grid = torch::meshgrid({axis, axis}, "ij");      // rows: S1, columns: S2
    auto S = torch::cat({grid[0].reshape({-1, 1}), grid[1].reshape({-1, 1})}, 1);
    auto tt = torch::full({S.size(0), 1}, t);

    auto price = toVector(pinn.price(S, tt));
    auto exercise = toVector(pinn.isInContinuationRegion(S, tt).logical_not());
    auto B = toVector(pinn.freeBoundary(torch::full({1, 1}, t)));
    auto vals = toVector(axis);

    std::printf("\nPrice Table 2D (Seed %lld, epochs %d, t = %.2f) - Rows: S1, Columns: S2\n", seed, epochs, t);
    std::printf("Estimated free boundary: B1(t) = %.4f, B2(t) = %.4f\n", B[0], B[1]);
    printGrid("S1/S2", vals, vals, price, false);
    std::printf("\nExercise Region Matrix 2D (t = %.2f, 1 = Early Exercise, 0 = Continuation Region)\n", t);
    printGrid("S1/S2", vals, vals, exercise, true);
}

} // namespace it::univr::output
