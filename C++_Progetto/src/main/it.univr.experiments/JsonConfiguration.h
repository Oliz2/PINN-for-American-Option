#pragma once
#include <memory>
#include <string>
#include <vector>
#include "ExperimentSettings.h"
#include "it.univr.neuralNetworks/IOption.h"

namespace it::univr::experiments {

/** Complete experiment read from file: option, market, numerical settings and seeds. */
struct ExperimentDefinition {
    std::shared_ptr<IOption> option;     // the option to price (currently the put on the minimum)
    BlackScholesParameters market;       // r, q_i and the volatility matrix sigma
    ExperimentSettings settings;         // networks, points, training, output
    std::vector<long long> seeds;        // one complete run per seed
};

/**
 * Reads an experiment from a <b>JSON file</b> (written by the Python interface or by hand).
 * <p>
 * Every field is required: if one is missing, loading stops with a message naming it.
 * Structure of the file (complete example in Grafici_python_PINN/esempio_config.json):
 * <pre>
 *   option     { type, numberOfAssets, strike, maturity }
 *   market     { riskFreeRate, dividendYields[d], volatilityMatrix[d][d] }
 *   pricing    { spotPrices[n][d], valuationTime }
 *   domain     { sMax }
 *   network    { solHiddenLayers, solHiddenUnits, fbHiddenLayers, fbHiddenUnits }
 *   points     { pde, initial, dirichlet, fbDirichlet, fbNeumann, sampling,
 *                strikeRefinementFraction, strikeRefinementHalfWidth }
 *   variants   { name, normalizeInputs, hardTerminalCondition, enforceIntrinsicValue,
 *                constrainedFreeBoundary, penaltyFormulation, exerciseTolerance, doublePrecision }
 *   weights    { pde, initial, dirichlet, fbInitial, fbDirichlet, fbNeumann }
 *   training   { epochs, solutionStepsPerBoundaryStep, separateBoundaryOptimizer, solution{...},
 *                boundary{...}, rmspropRho, rmspropEpsilon, slowDecayToFinalLearningRate,
 *                finalLearningRate, verbose, logEvery }
 *   testSet    { pdePoints, pdeBatchSize, boundaryPoints }
 *   output     { printPriceTable, tableSMin, tableSMax, tableSSteps, tableTSteps,
 *                exportCsv, curveTSteps, surfaceSMax, surfaceSSteps, surfaceTSteps }
 *   seeds      [ ... ]
 * </pre>
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class JsonConfiguration {
public:
    /**
     * @param path Path of the JSON file.
     * @return The experiment definition.
     */
    static ExperimentDefinition load(const std::string& path);
};

} // namespace it::univr::experiments
