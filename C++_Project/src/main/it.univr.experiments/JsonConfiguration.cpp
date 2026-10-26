/*
 * JsonConfiguration.cpp
 * Reads the config.json of a run (written by the Python interface) into an ExperimentDefinition.
 * The JSON parser is nlohmann/json (single header, third_party folder).
 *
 * Author: Alberto Oliva Medin
 */
#include "JsonConfiguration.h"

#include <fstream>
#include <stdexcept>

#include "third_party/nlohmann/json.hpp"
#include "it.univr.neuralNetworks/Putoption.h"
#include "ExperimentRunner.h"

namespace it::univr::experiments {

using nlohmann::json;

namespace {

// Required field: if it is missing, the error says exactly which one
const json& field(const json& obj, const std::string& key, const std::string& where) {
    if (!obj.is_object() || !obj.contains(key))
        throw std::invalid_argument("Configuration: missing field '" + where + "." + key + "'");
    return obj.at(key);
}
// Required field converted to type T: a wrong type also gives a message with the name of the field
template <class T>
T get(const json& obj, const std::string& key, const std::string& where) {
    try {
        return field(obj, key, where).get<T>();
    } catch (const json::exception& e) {
        throw std::invalid_argument("Configuration: field '" + where + "." + key + "' has the wrong type (" + e.what() + ")");
    }
}
// Learning-rate schedule { initialLearningRate, decaySteps, decayRate }
LearningRateSchedule schedule(const json& obj, const std::string& where) {
    return LearningRateSchedule{.initialLearningRate = get<double>(obj, "initialLearningRate", where),
                                .decaySteps = get<int>(obj, "decaySteps", where),
                                .decayRate = get<double>(obj, "decayRate", where)};
}

} // namespace

/*
 * Reads the file section by section, in the order of the documentation in the header.
 * Every value is required: there are no hidden defaults, so the JSON file describes the run completely.
 */
ExperimentDefinition JsonConfiguration::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("Cannot open the configuration file: " + path);
    json root;
    try {
        in >> root;
    } catch (const json::exception& e) {
        throw std::invalid_argument(std::string("Invalid configuration file: ") + e.what());
    }

    ExperimentDefinition def;

    // Option
    const json& o = field(root, "option", "");
    const std::string type = get<std::string>(o, "type", "option");
    const int d = get<int>(o, "numberOfAssets", "option");
    const double K = get<double>(o, "strike", "option");
    const double T = get<double>(o, "maturity", "option");
    if (type != "put")
        throw std::invalid_argument("Option type '" + type + "' not available: only 'put' exists for now");
    def.option = std::make_shared<neuralnetworks::PutOption>(K, T, d);

    // Market
    const json& m = field(root, "market", "");
    def.market.riskFreeRate = get<double>(m, "riskFreeRate", "market");
    def.market.dividendYields = get<std::vector<double>>(m, "dividendYields", "market");
    def.market.volatilityMatrix = get<std::vector<std::vector<double>>>(m, "volatilityMatrix", "market");

    ExperimentSettings& s = def.settings;
    s = ExperimentSettings{};

    // Prices to compute (an S0 with one component is repeated on all the d assets)
    const json& p = field(root, "pricing", "");
    s.spotPrices = get<std::vector<std::vector<double>>>(p, "spotPrices", "pricing");
    for (auto& S0 : s.spotPrices)
        if (S0.size() == 1) S0.assign(d, S0[0]);
    s.valuationTime = get<double>(p, "valuationTime", "pricing");

    // Domain and networks
    s.sMax = get<double>(field(root, "domain", ""), "sMax", "domain");
    const json& n = field(root, "network", "");
    s.solHiddenLayers = get<int>(n, "solHiddenLayers", "network");
    s.solHiddenUnits = get<int>(n, "solHiddenUnits", "network");
    s.fbHiddenLayers = get<int>(n, "fbHiddenLayers", "network");
    s.fbHiddenUnits = get<int>(n, "fbHiddenUnits", "network");

    // Collocation points
    const json& pt = field(root, "points", "");
    s.nPde = get<int>(pt, "pde", "points");
    s.nInit = get<int>(pt, "initial", "points");
    s.nDir = get<int>(pt, "dirichlet", "points");
    s.nFbDir = get<int>(pt, "fbDirichlet", "points");
    s.nFbNeu = get<int>(pt, "fbNeumann", "points");
    const std::string sampling = get<std::string>(pt, "sampling", "points");
    if (sampling != "sobol" && sampling != "uniform")
        throw std::invalid_argument("points.sampling must be 'sobol' or 'uniform'");
    s.sampling = SamplingOptions{.method = sampling == "sobol" ? SamplingMethod::Sobol : SamplingMethod::Uniform,
                                 .strikeRefinementFraction = get<double>(pt, "strikeRefinementFraction", "points"),
                                 .strikeRefinementHalfWidth = get<double>(pt, "strikeRefinementHalfWidth", "points")};

    // Variants
    const json& v = field(root, "variants", "");
    s.variantName = get<std::string>(v, "name", "variants");
    s.normalizeInputs = get<bool>(v, "normalizeInputs", "variants");
    s.useDoublePrecision = get<bool>(v, "doublePrecision", "variants");
    s.pinnOptions = PINNOptions{.hardTerminalCondition = get<bool>(v, "hardTerminalCondition", "variants"),
                                .enforceIntrinsicValue = get<bool>(v, "enforceIntrinsicValue", "variants"),
                                .constrainedFreeBoundary = get<bool>(v, "constrainedFreeBoundary", "variants"),
                                .penaltyFormulation = get<bool>(v, "penaltyFormulation", "variants"),
                                .exerciseTolerance = get<double>(v, "exerciseTolerance", "variants")};

    // Loss weights
    const json& w = field(root, "weights", "");
    s.weights = LossWeights{.pde = get<double>(w, "pde", "weights"), .initial = get<double>(w, "initial", "weights"),
                            .dirichlet = get<double>(w, "dirichlet", "weights"),
                            .fbInitial = get<double>(w, "fbInitial", "weights"),
                            .fbDirichlet = get<double>(w, "fbDirichlet", "weights"),
                            .fbNeumann = get<double>(w, "fbNeumann", "weights")};

    // Training
    const json& t = field(root, "training", "");
    s.training = TrainingConfig{
        .epochs = get<int>(t, "epochs", "training"),
        .solutionStepsPerBoundaryStep = get<int>(t, "solutionStepsPerBoundaryStep", "training"),
        .separateBoundaryOptimizer = get<bool>(t, "separateBoundaryOptimizer", "training"),
        .solution = schedule(field(t, "solution", "training"), "training.solution"),
        .boundary = schedule(field(t, "boundary", "training"), "training.boundary"),
        .rmspropRho = get<double>(t, "rmspropRho", "training"),
        .rmspropEpsilon = get<double>(t, "rmspropEpsilon", "training"),
        .verbose = get<bool>(t, "verbose", "training"),
        .logEvery = get<int>(t, "logEvery", "training")};
    // "slowlr": decaySteps is recomputed so that the learning rate reaches finalLearningRate at the last epoch
    if (get<bool>(t, "slowDecayToFinalLearningRate", "training"))
        ExperimentRunner::fitLearningRateDecay(s.training, s.pinnOptions.penaltyFormulation,
                                               get<double>(t, "finalLearningRate", "training"));

    // Test set and output
    const json& ts = field(root, "testSet", "");
    s.testPdePoints = get<int>(ts, "pdePoints", "testSet");
    s.testPdeBatchSize = get<int>(ts, "pdeBatchSize", "testSet");
    s.testBoundaryPoints = get<int>(ts, "boundaryPoints", "testSet");
    const json& out = field(root, "output", "");
    s.printPriceTable = get<bool>(out, "printPriceTable", "output");
    s.tableSMin = get<double>(out, "tableSMin", "output");
    s.tableSMax = get<double>(out, "tableSMax", "output");
    s.tableSSteps = get<int>(out, "tableSSteps", "output");
    s.tableTSteps = get<int>(out, "tableTSteps", "output");
    s.exportCsv = get<bool>(out, "exportCsv", "output");
    s.curveTSteps = get<int>(out, "curveTSteps", "output");
    s.surfaceSMax = get<double>(out, "surfaceSMax", "output");
    s.surfaceSSteps = get<int>(out, "surfaceSSteps", "output");
    s.surfaceTSteps = get<int>(out, "surfaceTSteps", "output");

    // Seeds
    def.seeds = get<std::vector<long long>>(root, "seeds", "");
    if (def.seeds.empty()) throw std::invalid_argument("Configuration: 'seeds' is empty");
    return def;
}

} // namespace it::univr::experiments
