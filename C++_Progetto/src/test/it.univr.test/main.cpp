/*
 * main.cpp
 * Entry point of pinn.exe. Two modes: --config (parameters from a JSON file, used by the Python
 * interface) or stand-alone (parameters written in sections 1-5 below, useful for tests).
 *
 * Author: Alberto Oliva Medin
 */
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "it.univr.neuralNetworks/Putoption.h"
#include "it.univr.experiments/ExperimentSettings.h"
#include "it.univr.experiments/ExperimentRunner.h"
#include "it.univr.experiments/JsonConfiguration.h"
#include "it.univr.output/Table3Results.h"
#include "it.univr.output/OptionPriceReport.h"

using namespace it::univr::neuralnetworks;
using it::univr::experiments::ExperimentRunner;
using it::univr::experiments::ExperimentSettings;
using it::univr::experiments::SeedResult;
using it::univr::output::OptionPriceReport;
using it::univr::output::OptionPriceRow;
using it::univr::output::Table3Results;
using it::univr::output::Table3Row;

/**
 * Entry point of the <b>PINN pricer for American options</b>.
 * <p>
 * Usage:
 * <pre>
 *   pinn.exe                                -> all the seeds listed below
 *   pinn.exe &lt;seed&gt;                         -> one seed
 *   pinn.exe &lt;seed&gt; &lt;epochs&gt;                -> one seed, different number of epochs
 *   pinn.exe &lt;seed&gt; &lt;epochs&gt; &lt;variants&gt;     -> comma-separated variants among:
 *        base     base formulation with the free-boundary network
 *        norm     normalised inputs (S/K, t/T)
 *        refine   30% of the points near the strike
 *        fbcap    constrained boundary: B(t) <= K and B(T) = K by construction
 *        penalty  penalty formulation, without the free-boundary network
 *        slowlr   learning rate decaying to finalLearningRate (set below)
 *        double   double precision
 *        hard     terminal condition by construction
 *        raw      reported price without max(price, payoff)
 *
 *   pinn.exe --config file.json [--out folder]
 *        all the parameters from the JSON file (used by the Python interface);
 *        the results are written to "folder" (created if missing)
 * </pre>
 * Without --config, all the parameters are set in sections 1-5 of this file.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
int main(int argc, char* argv[]) {
    try {
        // Configuration-file mode (Python interface)
        if (argc > 1 && std::string(argv[1]) == "--config") {
            if (argc < 3) throw std::invalid_argument("Uso: pinn.exe --config file.json [--out cartella]");
            const std::string configPath = std::filesystem::absolute(argv[2]).string();
            if (argc > 4 && std::string(argv[3]) == "--out") {
                std::filesystem::create_directories(argv[4]);
                std::filesystem::current_path(argv[4]);   // all the CSV files go to this folder
            }
            std::setvbuf(stdout, nullptr, _IONBF, 0);       // output immediately visible to the interface
            auto def = it::univr::experiments::JsonConfiguration::load(configPath);
            if (def.settings.useDoublePrecision) torch::set_default_dtype(caffe2::TypeMeta::Make<double>());
            std::printf("CONFIG %s\n", configPath.c_str());
            ExperimentRunner::runAllSeeds(def.seeds, def.option, def.market, def.settings);
            std::printf("FINE\n");   // runner.py reads this line to know that the run has finished
            return 0;
        }

        /* ===========================================================================
         * 1) OPTION
         * ===========================================================================
         */
        const int    numberOfAssets = 1;          // d: number of underlyings (any d >= 1)
        const double strike         = 10.0;       // K
        const double maturity       = 3.0;        // T (years)

        // Any class implementing IOption. PutOption: American put on the minimum, payoff (K - min_i S_i)^+
        std::shared_ptr<IOption> option = std::make_shared<PutOption>(strike, maturity, numberOfAssets);

        /* ===========================================================================
         * 2) MARKET (Black-Scholes)
         * ===========================================================================
         */
        const double riskFreeRate = 0.01;                          // r
        const std::vector<double> dividendYields = {0.0};          // q_i, one per underlying
        // Volatility matrix sigma, d x d (the diffusion uses sigma * sigma^T)
        const std::vector<std::vector<double>> volatilityMatrix = {
            {0.05}};

        /* ===========================================================================
         * 3) PRICES TO COMPUTE
         * Each row is an initial point S0 = (S1, ..., Sd); a row with one component
         * is used for all the underlyings.
         * ===========================================================================
         */
        std::vector<std::vector<double>> spotPrices = {{8.0}, {9.0}, {9.5}, {10.0}, {10.5}, {11.0}, {12.0}};
        const double valuationTime = 0.0;          // valuation time (0 = today)

        /* ===========================================================================
         * 4) NEURAL NETWORKS, COLLOCATION POINTS AND TRAINING
         * ===========================================================================
         */
        const int d = numberOfAssets;
        if (d > 4)
            std::printf("ATTENZIONE: d = %d sottostanti, l'addestramento sarà molto lento.\n", d);

        BlackScholesParameters market;
        market.riskFreeRate     = riskFreeRate;
        market.dividendYields   = dividendYields;
        market.volatilityMatrix = volatilityMatrix;

        ExperimentSettings s{};
        for (auto& S0 : spotPrices)
            if (S0.size() == 1) S0.assign(d, S0[0]);
        s.spotPrices    = spotPrices;
        s.valuationTime = valuationTime;

        // Variants (all false = base formulation); overridden by the command-line list
        s.useDoublePrecision = false;                      // float32
        s.normalizeInputs = false;
        s.pinnOptions = PINNOptions{.hardTerminalCondition   = false,
                                    .enforceIntrinsicValue   = true,
                                    .constrainedFreeBoundary = false,
                                    .penaltyFormulation      = false,
                                    .exerciseTolerance       = 1e-3};  // penalty formulation only
        double strikeRefinementFraction = 0.0;
        bool slowLearningRateDecay = false;
        s.variantName = "base";
        if (argc > 3) {
            const std::string list = argv[3];
            s.variantName = list;
            auto has = [&](const std::string& k) { return ("," + list + ",").find("," + k + ",") != std::string::npos; };
            s.useDoublePrecision = has("double");
            s.normalizeInputs = has("norm");
            s.pinnOptions.hardTerminalCondition = has("hard");
            s.pinnOptions.enforceIntrinsicValue = has("intrinsic") || list != "raw";
            s.pinnOptions.constrainedFreeBoundary = has("fbcap");
            s.pinnOptions.penaltyFormulation = has("penalty");
            strikeRefinementFraction = has("refine") ? 0.3 : 0.0;
            slowLearningRateDecay = has("slowlr");
        }
        if (s.useDoublePrecision) torch::set_default_dtype(caffe2::TypeMeta::Make<double>());

        // Domain: S in [0, sMax]^d
        s.sMax = 3.0 * strike;

        // Network architectures
        s.solHiddenLayers = 8;    s.solHiddenUnits = 20;   // FNNSol
        s.fbHiddenLayers  = 3;    s.fbHiddenUnits  = 100;  // FNNfb

        // Collocation points
        s.nPde   = 30000;                                  // interior (PDE)
        s.nInit  = 300;                                    // terminal condition
        s.nDir   = 300;                                    // far boundary
        s.nFbDir = 300;                                    // free boundary, Dirichlet
        s.nFbNeu = 300;                                    // free boundary, Neumann
        s.sampling = SamplingOptions{.method = SamplingMethod::Sobol,
                                     .strikeRefinementFraction = strikeRefinementFraction,
                                     .strikeRefinementHalfWidth = 0.2 * strike};

        // Loss weights
        s.weights = LossWeights{.pde = 1.0, .initial = 1.0, .dirichlet = 1.0,
                                .fbInitial = 1.0, .fbDirichlet = 1.0, .fbNeumann = 1.0};

        // Training
        s.training = TrainingConfig{
            .epochs                       = 1500,
            .solutionStepsPerBoundaryStep = 20,            // n_steps
            .separateBoundaryOptimizer    = false,         // false = one optimiser for both networks
            .solution = {.initialLearningRate = 1e-2, .decaySteps = 500, .decayRate = 0.9},
            .boundary = {.initialLearningRate = 1e-2, .decaySteps = 500, .decayRate = 0.9},
            .rmspropRho                   = 0.9,
            .rmspropEpsilon               = 1e-7,
            .verbose                      = true,          // step-by-step output
            .logEvery                     = 100            // used only with verbose = false
        };
        if (argc > 2) s.training.epochs = std::stoi(argv[2]);

        // Variant slowlr: the learning rate reaches finalLearningRate at the last epoch
        // (decaySteps is recomputed from the number of epochs and of optimiser steps per epoch)
        // (with the base decay it is already ~1e-6 after a few thousand epochs and the network stops learning)
        const double finalLearningRate = 1e-4;
        if (slowLearningRateDecay)
            ExperimentRunner::fitLearningRateDecay(s.training, s.pinnOptions.penaltyFormulation, finalLearningRate);

        /* ===========================================================================
         * 5) OUTPUT
         * ===========================================================================
         */
        // Test set
        s.testPdePoints      = 1000000;
        s.testPdeBatchSize   = 20000;
        s.testBoundaryPoints = 1000;

        // Price table and exercise region matrix (printed for d = 1 and d = 2)
        s.printPriceTable = true;
        s.tableSMin   = 0.8 * strike;
        s.tableSMax   = 1.2 * strike;
        s.tableSSteps = 9;                                 // 10 rows
        s.tableTSteps = 10;                                // 11 columns, t from 0 to T (d = 1 only)

        // CSV files for the charts
        s.exportCsv     = true;
        s.curveTSteps   = 600;
        s.surfaceSMax   = 1.5 * strike;
        s.surfaceSSteps = 150;
        s.surfaceTSteps = 150;

        // Seeds: each one is a complete, reproducible run (the seed fixes the initial weights)
        std::vector<long long> seeds = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        if (argc > 1) seeds = {std::stoll(argv[1])};

        /* ===========================================================================
         * RUN
         * ===========================================================================
         */
        ExperimentRunner::runAllSeeds(seeds, option, market, s);
    } catch (const std::exception& e) {
        std::cerr << "ERRORE: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
