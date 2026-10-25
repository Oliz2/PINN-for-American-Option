#pragma once
#include <memory>
#include <string>
#include <vector>
#include "ExperimentSettings.h"
#include "it.univr.neuralNetworks/IOption.h"
#include "it.univr.output/Table3Results.h"
#include "it.univr.output/OptionPriceReport.h"

namespace it::univr::experiments {

/** Result of one seed: test-set losses and option prices at the requested points. */
struct SeedResult {
    output::Table3Row losses;        // losses on the test set (same columns as Table 3 of the reference paper)
    output::OptionPriceRow prices;   // prices, boundary and exercise flags at the requested S0
};

/**
 * Runs a complete <b>PINN experiment</b> on a given option and market.
 * <p>
 * For every seed: construction of the networks, training, option prices, price tables,
 * CSV files and evaluation of the losses on an independent test set.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class ExperimentRunner {
public:
    /**
     * Runs the experiment for one seed.
     *
     * @param seed   Seed of the random generator (set before the networks are built).
     * @param option The option to price.
     * @param market Black-Scholes market parameters.
     * @param s      Numerical settings.
     * @return Test-set losses and option prices.
     */
    static SeedResult runSingleSeed(long long seed, std::shared_ptr<IOption> option,
                                    const BlackScholesParameters& market, const ExperimentSettings& s);

    /**
     * Runs all the seeds in sequence, then writes the summary files table3_results*.csv and
     * option_prices*.csv (with mean and median when there is more than one seed).
     */
    static void runAllSeeds(const std::vector<long long>& seeds, std::shared_ptr<IOption> option,
                            const BlackScholesParameters& market, const ExperimentSettings& s);

    /**
     * Chooses decaySteps so that the learning rate reaches finalLearningRate at the last epoch
     * (variant "slowlr").
     */
    static void fitLearningRateDecay(TrainingConfig& training, bool penaltyFormulation, double finalLearningRate);

    /** @return The suffix of the output file names, e.g. "_d1_norm-refine-penalty-slowlr". */
    static std::string fileSuffix(const IOption& option, const ExperimentSettings& s);

private:
    /**
     * Evaluates the losses on new random points, not used in the training.
     * Solution = Unsupervised (PDE residual in the continuation region) + Initial + Dirichlet + FB_Dir + FB_Neu,
     * the same definition for both formulations, so the two can be compared.
     */
    static output::Table3Row evaluateTestSet(PINN& pinn, const ExperimentSettings& s, long long seed);
};

} // namespace it::univr::experiments
