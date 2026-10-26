#pragma once
#include "it.univr.neuralNetworks.PINN/PINN.h"

namespace it::univr::output {

/**
 * Prints the <b>price table</b> and the <b>exercise region matrix</b> on the standard output.
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class PriceTablePrinter {
public:
    /** d = 1: rows S in [sMin, sMax] (sSteps + 1 values), columns t from 0 to T (tSteps + 1 values). */
    static void print1D(neuralnetworks::PINN& pinn, double sMin, double sMax, int sSteps,
                        int tSteps, long long seed, int epochs);

    /** d = 2: rows S1, columns S2, both in [sMin, sMax], at time t. */
    static void print2D(neuralnetworks::PINN& pinn, double sMin, double sMax, int sSteps,
                        double t, long long seed, int epochs);
};

} // namespace it::univr::output
