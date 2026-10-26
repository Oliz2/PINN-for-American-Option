/*
 * CsvExporter.cpp
 * CSV files read by the Python interface to draw the charts: training history, exercise boundary,
 * price surfaces. Column names are part of the interface: changing them breaks the charts.
 *
 * Author: Alberto Oliva Medin
 */
#include "CsvExporter.h"
#include "it.univr.utils/TensorUtils.h"
#include <cstdio>
#include <fstream>

namespace it::univr::output {

using it::univr::utils::toVector;

// One row per epoch: learning rates and every loss term of the joint step.
void CsvExporter::trainingHistory(const std::vector<neuralnetworks::EpochRecord>& history,
                                  long long seed, const std::string& suffix) {
    const std::string name = "training_history_seed" + std::to_string(seed) + suffix + ".csv";
    std::ofstream f(name);
    f << "epoch,lr_sol,lr_fb,pde,initial,dirichlet,fb_init,fb_dir,fb_neu,free_boundary,solution\n";
    f.precision(10);
    for (const auto& r : history)
        f << r.epoch << "," << std::scientific << r.solutionLearningRate << "," << r.boundaryLearningRate
          << "," << r.pde << "," << r.initial << "," << r.dirichlet << "," << r.fbInitial << ","
          << r.fbDirichlet << "," << r.fbNeumann << "," << r.freeBoundaryLoss << "," << r.solutionLoss
          << "\n" << std::defaultfloat;
    std::printf("[TRACKING] Training history saved to: %s\n", name.c_str());
}

// Exercise boundary B(t) on tSteps + 1 equally spaced times between 0 and T.
void CsvExporter::boundaryCurve(neuralnetworks::PINN& pinn, int tSteps, long long seed,
                                const std::string& suffix) {
    const int d = pinn.getNumberOfAssets();
    auto t = torch::linspace(0.0, pinn.getOption()->getMaturity(), tSteps + 1).unsqueeze(1);
    auto tv = toVector(t);
    auto bv = toVector(pinn.freeBoundary(t));

    const std::string name = "boundary_curve_seed" + std::to_string(seed) + suffix + ".csv";
    std::ofstream f(name);
    f << "t";
    for (int i = 1; i <= d; ++i) f << (d == 1 ? ",B_t" : ",B" + std::to_string(i) + "_t");
    f << "\n";
    f.precision(8);
    for (size_t j = 0; j < tv.size(); ++j) {
        f << std::fixed << tv[j];
        for (int i = 0; i < d; ++i) f << "," << bv[j * d + i];
        f << "\n";
    }
    std::printf("[TRACKING] Free boundary curve saved to: %s\n", name.c_str());
}

/*
 * d = 1: grid (S, t) on [0, sMax] x [0, T]. For every point: V = raw network output, Price = American
 * price (with the payoff floor), B_t = boundary at that time, is_exercise = 1 in the exercise region.
 */
void CsvExporter::priceSurface1D(neuralnetworks::PINN& pinn, double sMax, int sSteps, int tSteps,
                                 long long seed, const std::string& suffix) {
    torch::NoGradGuard noGrad;
    auto option = pinn.getOption();
    auto Saxis = torch::linspace(0.0, sMax, sSteps + 1);
    auto taxis = torch::linspace(0.0, option->getMaturity(), tSteps + 1);
    auto grid = torch::meshgrid({Saxis, taxis}, "ij");   // every (S, t) pair, flattened to columns
    auto S = grid[0].reshape({-1, 1});
    auto t = grid[1].reshape({-1, 1});
    auto B = pinn.freeBoundary(t);

    auto Sv = toVector(S), tv = toVector(t), Bv = toVector(B);
    auto Vv = toVector(pinn.evaluate(S, t));
    auto Pv = toVector(pinn.price(S, t));
    auto Ev = toVector(pinn.isInContinuationRegion(S, t).logical_not());

    const std::string name = "price_surface_seed" + std::to_string(seed) + suffix + ".csv";
    std::ofstream f(name);
    f << "S,t,V,Price,B_t,is_exercise\n";
    f.precision(8);
    for (size_t k = 0; k < Sv.size(); ++k)
        f << std::fixed << Sv[k] << "," << tv[k] << "," << Vv[k] << "," << Pv[k] << "," << Bv[k] << ","
          << static_cast<int>(Ev[k]) << "\n";
    std::printf("[TRACKING] Price/exercise region surface saved to: %s\n", name.c_str());
}

// d = 2: grid (S1, S2) on [0, sMax]^2 at a fixed time t, same columns as the 1D surface without B.
void CsvExporter::priceSurface2D(neuralnetworks::PINN& pinn, double sMax, int sSteps, double t,
                                 long long seed, const std::string& suffix) {
    torch::NoGradGuard noGrad;
    auto option = pinn.getOption();
    auto axis = torch::linspace(0.0, sMax, sSteps + 1);
    auto grid = torch::meshgrid({axis, axis}, "ij");
    auto S = torch::cat({grid[0].reshape({-1, 1}), grid[1].reshape({-1, 1})}, 1);
    auto tt = torch::full({S.size(0), 1}, t);

    auto Sv = toVector(S);
    auto Vv = toVector(pinn.evaluate(S, tt));
    auto Pv = toVector(pinn.price(S, tt));
    auto Ev = toVector(pinn.isInContinuationRegion(S, tt).logical_not());

    char tag[32];
    std::snprintf(tag, sizeof(tag), "%.2f", t);
    const std::string name = "price_surface_2d_seed" + std::to_string(seed) + "_t" + tag + suffix + ".csv";
    std::ofstream f(name);
    f << "S1,S2,V,Price,is_exercise\n";
    f.precision(8);
    for (size_t k = 0; k < Vv.size(); ++k)
        f << std::fixed << Sv[2 * k] << "," << Sv[2 * k + 1] << "," << Vv[k] << "," << Pv[k] << ","
          << static_cast<int>(Ev[k]) << "\n";
    std::printf("[TRACKING] 2D surface (t = %.2f) saved to: %s\n", t, name.c_str());
}

} // namespace it::univr::output
