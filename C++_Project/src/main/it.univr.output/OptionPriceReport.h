#pragma once
#include <string>
#include <vector>
#include "it.univr.neuralNetworks.PINN/PINN.h"

namespace it::univr::output {

/** Option prices computed by the PINN for one seed. */
struct OptionPriceRow {
    long long seed;
    std::vector<double> prices;        // one price per initial point S0
    std::vector<double> boundary;      // free boundary B(t) at the valuation time (d values)
    std::vector<int> exercise;         // 1 = early-exercise region, 0 = continuation region
};

/**
 * Computes, prints and saves the <b>option prices</b> at the requested initial points.
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class OptionPriceReport {
public:
    /**
     * @param pinn  The trained PINN.
     * @param spots Initial points S0, each with d components.
     * @param t     Valuation time (0 = today).
     * @param seed  Seed of the run (for the printout).
     * @return Prices, free boundary and exercise flags.
     */
    static OptionPriceRow evaluate(neuralnetworks::PINN& pinn, const std::vector<std::vector<double>>& spots,
                                   double t, long long seed);

    /** Prints a table with all the seeds, plus mean and median for every S0. */
    static void printAll(const std::vector<OptionPriceRow>& rows, const std::vector<std::vector<double>>& spots,
                         double t);

    /** Writes the prices of all the seeds to a CSV file. */
    static void writeCsv(const std::vector<OptionPriceRow>& rows, const std::vector<std::vector<double>>& spots,
                         double t, const std::string& fileName);
};

} // namespace it::univr::output
