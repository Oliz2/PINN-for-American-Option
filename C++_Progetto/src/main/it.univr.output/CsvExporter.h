#pragma once
#include <string>
#include <vector>
#include "it.univr.neuralNetworks.PINN/PINN.h"
#include "it.univr.neuralNetworks.PINN/TrainNeuralNetwork.h"

namespace it::univr::output {

/**
 * Writes the <b>CSV files</b> used for the charts and for the training history.
 * <p>
 * Every file name contains the seed and the given suffix (e.g. "_d1_norm-refine-penalty-slowlr").
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class CsvExporter {
public:
    /** training_history_seed&lt;seed&gt;&lt;suffix&gt;.csv: all the losses and learning rates at every epoch. */
    static void trainingHistory(const std::vector<neuralnetworks::EpochRecord>& history,
                                long long seed, const std::string& suffix);

    /** boundary_curve_seed&lt;seed&gt;&lt;suffix&gt;.csv: the curve B(t), one column per underlying. */
    static void boundaryCurve(neuralnetworks::PINN& pinn, int tSteps, long long seed, const std::string& suffix);

    /** price_surface_seed&lt;seed&gt;&lt;suffix&gt;.csv: d = 1, price and exercise region on [0, sMax] x [0, T]. */
    static void priceSurface1D(neuralnetworks::PINN& pinn, double sMax, int sSteps, int tSteps,
                               long long seed, const std::string& suffix);

    /** price_surface_2d_seed&lt;seed&gt;_t&lt;t&gt;&lt;suffix&gt;.csv: d = 2, price and exercise region on [0, sMax]^2 at time t. */
    static void priceSurface2D(neuralnetworks::PINN& pinn, double sMax, int sSteps, double t,
                               long long seed, const std::string& suffix);
};

} // namespace it::univr::output
