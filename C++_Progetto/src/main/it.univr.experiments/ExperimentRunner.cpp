/*
 * ExperimentRunner.cpp
 * Runs one PINN experiment from start to end: builds the networks, trains them, computes the prices,
 * writes tables and CSV files and evaluates the losses on a test set.
 * It does not solve anything itself: it coordinates PINN, TrainNeuralNetwork and the output classes.
 *
 * Author: Alberto Oliva Medin
 */
#include "ExperimentRunner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "it.univr.neuralNetworks/FNNSol.h"
#include "it.univr.neuralNetworks/FNNfb.h"
#include "it.univr.output/CsvExporter.h"
#include "it.univr.output/PriceTablePrinter.h"

namespace it::univr::experiments {

using it::univr::output::CsvExporter;
using it::univr::output::OptionPriceReport;
using it::univr::output::PriceTablePrinter;
using it::univr::output::Table3Results;
using it::univr::output::Table3Row;

// "_d<d>_<variant>", with the commas of the variant replaced by dashes so that it is a valid file name.
std::string ExperimentRunner::fileSuffix(const IOption& option, const ExperimentSettings& s) {
    std::string variant = s.variantName;
    std::replace(variant.begin(), variant.end(), ',', '-');
    return "_d" + std::to_string(option.getNumberOfAssets()) + "_" + variant;
}

/*
 * Variant "slowlr": the learning rate must go from its initial value to finalLearningRate exactly at the
 * last epoch. With lr = lr0 * rate^(steps / decaySteps), the number of decays needed is
 * log(final / lr0) / log(rate), and decaySteps = total optimiser steps / that number.
 */
void ExperimentRunner::fitLearningRateDecay(TrainingConfig& training, bool penaltyFormulation,
                                            double finalLearningRate) {
    auto fit = [&](LearningRateSchedule& lr, double stepsPerEpoch) {
        const double totalSteps = training.epochs * stepsPerEpoch;
        const double decays = std::log(finalLearningRate / lr.initialLearningRate) / std::log(lr.decayRate);
        lr.decaySteps = std::max(1, static_cast<int>(totalSteps / decays));
    };
    const int n = training.solutionStepsPerBoundaryStep;
    const bool fbTrained = !penaltyFormulation;
    // optimiser steps per epoch: n - 1 inner + 1 joint (+1 if FNNfb shares the optimiser)
    fit(training.solution, n + ((!training.separateBoundaryOptimizer && fbTrained) ? 1 : 0));
    fit(training.boundary, 1.0);
}

// Runs every seed, then writes the summary CSV files (with mean and median when there are several seeds).
void ExperimentRunner::runAllSeeds(const std::vector<long long>& seeds, std::shared_ptr<IOption> option,
                                   const BlackScholesParameters& market, const ExperimentSettings& s) {
    std::vector<Table3Row> losses;
    std::vector<output::OptionPriceRow> prices;
    for (long long seed : seeds) {
        SeedResult r = runSingleSeed(seed, option, market, s);
        losses.push_back(r.losses);
        prices.push_back(r.prices);
    }
    std::string tag = fileSuffix(*option, s);
    if (seeds.size() == 1)
        tag = "_seed" + std::to_string(seeds[0]) + "_epochs" + std::to_string(s.training.epochs) + tag;
    else {
        Table3Results::printAll(losses);
        OptionPriceReport::printAll(prices, s.spotPrices, s.valuationTime);
    }
    Table3Results::writeCsv(losses, "table3_results" + tag + ".csv");
    OptionPriceReport::writeCsv(prices, s.spotPrices, s.valuationTime, "option_prices" + tag + ".csv");
}

/* ===========================================================================
 * SINGLE SEED: networks, training, output, test-set evaluation
 * ===========================================================================
 */
SeedResult ExperimentRunner::runSingleSeed(long long seed, std::shared_ptr<IOption> option,
                                           const BlackScholesParameters& market, const ExperimentSettings& s) {
    // The seed must be set BEFORE building the networks: the Glorot initialisation depends on it
    torch::manual_seed(static_cast<uint64_t>(seed));

    const int d = option->getNumberOfAssets();
    const double K = option->getStrike();
    const double T = option->getMaturity();
    // Input normalisation: the networks see S/K and t/T instead of S and t
    const double sScale = s.normalizeInputs ? K : 1.0;
    const double tScale = s.normalizeInputs ? T : 1.0;
    auto sol = std::make_shared<FNNSol>(d, s.solHiddenLayers, s.solHiddenUnits, sScale, tScale);
    // With the constrained boundary FNNfb returns the argument of the softplus: no factor K
    const double fbScale = s.pinnOptions.constrainedFreeBoundary ? 1.0 : sScale;
    auto fb  = std::make_shared<FNNfb>(d, s.fbHiddenLayers, s.fbHiddenUnits, tScale, fbScale);
    PINN pinn(option, market, s.sMax, sol, fb, s.weights, s.pinnOptions);

    // The collocation points are sampled once and kept fixed for the whole training
    auto trainingSet = pinn.sampleCollocationPoints(s.nPde, s.nInit, s.nDir, s.nFbDir, s.nFbNeu, s.sampling);

    std::printf("\n=====================================================================\n");
    std::printf("Starting PINN training (%s, K = %.4f, T = %.4f, Seed: %lld, variante: %s)...\n",
                option->getName().c_str(), K, T, seed, s.variantName.c_str());
    std::printf("%s\n%s\n", sol->describe().c_str(), fb->describe().c_str());
    std::fflush(stdout);

    TrainNeuralNetwork trainer(pinn, trainingSet, s.training);
    const auto start = std::chrono::steady_clock::now();
    trainer.train();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Training completed in: %.1f s\n", secs);

    SeedResult result;

    // Option prices at the requested points: one forward pass of the trained network (PINN::price)
    result.prices = OptionPriceReport::evaluate(pinn, s.spotPrices, s.valuationTime, seed);

    // Tables on screen and CSV files
    const std::string suffix = fileSuffix(*option, s);
    const int epochs = s.training.epochs;
    if (s.exportCsv) CsvExporter::trainingHistory(trainer.getHistory(), seed, suffix);

    if (d == 1 && s.printPriceTable)
        PriceTablePrinter::print1D(pinn, s.tableSMin, s.tableSMax, s.tableSSteps, s.tableTSteps, seed, epochs);
    if (d == 2) {
        for (double t : {0.0, 0.5 * T}) {   // price tables and surfaces at t = 0 and t = T/2
            if (s.printPriceTable)
                PriceTablePrinter::print2D(pinn, s.tableSMin, s.tableSMax, s.tableSSteps, t, seed, epochs);
            if (s.exportCsv) CsvExporter::priceSurface2D(pinn, s.surfaceSMax, s.surfaceSSteps, t, seed, suffix);
        }
    }
    if (s.exportCsv) {
        CsvExporter::boundaryCurve(pinn, s.curveTSteps, seed, suffix);
        if (d == 1) CsvExporter::priceSurface1D(pinn, s.surfaceSMax, s.surfaceSSteps, s.surfaceTSteps, seed, suffix);
    }

    // Losses on the test set
    result.losses = evaluateTestSet(pinn, s, seed);
    return result;
}

/* ===========================================================================
 * TEST SET: random points, PDE residual only in the continuation region
 * ===========================================================================
 */
Table3Row ExperimentRunner::evaluateTestSet(PINN& pinn, const ExperimentSettings& s, long long seed) {
    auto option = pinn.getOption();
    const int d = option->getNumberOfAssets();
    const double T = option->getMaturity();

    std::printf("\nEvaluating test set metrics (%d PDE points, %d boundary points)...\n",
                s.testPdePoints, s.testBoundaryPoints);

    // PDE: random points in [0, Smax]^d x [0, T], kept only in the continuation region
    double sumSquares = 0.0;
    long long pdeCount = 0;
    for (int done = 0; done < s.testPdePoints; done += s.testPdeBatchSize) {
        const int n = std::min(s.testPdeBatchSize, s.testPdePoints - done);
        auto S = torch::rand({n, d}) * s.sMax;
        auto t = torch::rand({n, 1}) * T;
        auto continuation = pinn.isInContinuationRegion(S, t).squeeze(1);
        auto Sc = S.index({continuation});
        auto tc = t.index({continuation});
        if (Sc.size(0) == 0) continue;
        // pure Black-Scholes residual (Lambda P)^2, also for the complementarity formulation
        sumSquares += pinn.pdeResidual(Sc, tc).pow(2).sum().item<double>();
        pdeCount += Sc.size(0);
    }

    // Boundary conditions, on new random points (the free boundary is not trained here: trainBoundary = false)
    const int nb = s.testBoundaryPoints;
    auto initS  = torch::rand({nb, d}) * s.sMax;
    auto dirS   = option->projectOnFarBoundary(torch::rand({nb, d}) * s.sMax, s.sMax);
    auto dirT   = torch::rand({nb, 1}) * T;
    auto fbDirT = torch::rand({nb, 1}) * T;
    auto fbNeuT = torch::rand({nb, 1}) * T;

    Table3Row row{};
    row.seed         = seed;
    row.unsupervised = pdeCount > 0 ? sumSquares / pdeCount : 0.0;
    row.initial      = pinn.lossInitial(initS).item<double>();
    row.dirichlet    = pinn.lossDirichlet(dirS, dirT).item<double>();
    row.fbInit       = pinn.lossFreeBoundaryInitial().item<double>();
    row.fbDir        = pinn.lossFreeBoundaryDirichlet(fbDirT, false).item<double>();
    row.fbNeu        = pinn.lossFreeBoundaryNeumann(fbNeuT, false).item<double>();
    row.freeBoundary = row.fbInit + row.fbDir + row.fbNeu;
    row.solution     = row.unsupervised + row.initial + row.dirichlet + row.fbDir + row.fbNeu;

    Table3Results::printRow(row);
    std::printf("[DIAGNOSTICA] Seed %lld: punti PDE di test nella regione di continuazione: %lld su %d\n",
                seed, pdeCount, s.testPdePoints);
    return row;
}

} // namespace it::univr::experiments
