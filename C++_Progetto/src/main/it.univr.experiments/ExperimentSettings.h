#pragma once
#include <string>
#include <vector>
#include "it.univr.neuralNetworks.PINN/PINN.h"
#include "it.univr.neuralNetworks.PINN/TrainNeuralNetwork.h"

namespace it::univr::experiments {

using namespace it::univr::neuralnetworks;

/**
 * Numerical settings of an experiment: networks, collocation points, training and output.
 * <p>
 * The option and the market are not part of the settings: they are built separately and passed to
 * ExperimentRunner. There are no default values: every field is set by the caller.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
struct ExperimentSettings {
    bool useDoublePrecision;       // false = float32, true = float64

    double sMax;                   // domain: S in [0, sMax]^d

    // Network architectures (thesis: FNNSol 8 x 20, FNNfb 3 x 100)
    int solHiddenLayers, solHiddenUnits;
    int fbHiddenLayers, fbHiddenUnits;

    // Training collocation points: interior, maturity, far boundary, free-boundary Dirichlet and Neumann
    int nPde, nInit, nDir, nFbDir, nFbNeu;
    SamplingOptions sampling;

    // Variants
    bool normalizeInputs;          // FNNSol sees S/K and t/T; FNNfb sees t/T and returns B in units of K
    PINNOptions pinnOptions;
    std::string variantName;       // label used in the file names

    LossWeights weights;           // weights of the loss terms
    TrainingConfig training;       // epochs, steps, learning rates, RMSprop

    // Prices to compute: every element is an S0 with d components, valued at valuationTime
    std::vector<std::vector<double>> spotPrices;
    double valuationTime;

    // Test set: independent random points, used only to evaluate the losses after the training
    int testPdePoints;             // PDE points (before keeping only the continuation region)
    int testPdeBatchSize;          // PDE points evaluated at a time (limits the memory)
    int testBoundaryPoints;        // points for each boundary condition

    // Price table and exercise region (d = 1: rows S, columns t; d = 2: rows S1, columns S2)
    bool printPriceTable;
    double tableSMin, tableSMax;
    int tableSSteps;               // rows = tableSSteps + 1
    int tableTSteps;               // columns = tableTSteps + 1, t from 0 to T (d = 1 only)

    // CSV files for the charts
    bool exportCsv;
    int curveTSteps;               // resolution of the curve B(t)
    double surfaceSMax;            // price surface on S in [0, surfaceSMax]
    int surfaceSSteps, surfaceTSteps;
};

} // namespace it::univr::experiments
