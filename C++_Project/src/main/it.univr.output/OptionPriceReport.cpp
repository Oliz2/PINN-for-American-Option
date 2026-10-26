/*
 * OptionPriceReport.cpp
 * Final prices of the option at the requested S0: computation, console table and CSV file.
 *
 * Author: Alberto Oliva Medin
 */
#include "OptionPriceReport.h"
#include "it.univr.utils/TensorUtils.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace it::univr::output {

using it::univr::utils::toVector;

// "(10.00; 10.00)": label of an initial point in the tables.
static std::string spotLabel(const std::vector<double>& s) {
    std::string out = "(";
    char buf[32];
    for (size_t i = 0; i < s.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "%s%.2f", i ? "; " : "", s[i]);
        out += buf;
    }
    return out + ")";
}

static double mean(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / v.size();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size(), mid = n / 2;
    return (n % 2 == 0) ? (v[mid - 1] + v[mid]) / 2.0 : v[mid];
}

/*
 * Prices at all the requested S0 in one batch: the S0 are stacked into a tensor [n, d] and passed to
 * PINN::price (forward pass of the trained network + max with the payoff). No gradient is needed.
 */
OptionPriceRow OptionPriceReport::evaluate(neuralnetworks::PINN& pinn,
                                           const std::vector<std::vector<double>>& spots, double t,
                                           long long seed) {
    torch::NoGradGuard noGrad;
    auto option = pinn.getOption();
    const int d = pinn.getNumberOfAssets();

    std::vector<double> flat;
    for (const auto& s : spots) {
        if (static_cast<int>(s.size()) != d)
            throw std::invalid_argument("Every initial price S0 must have " + std::to_string(d) + " components");
        flat.insert(flat.end(), s.begin(), s.end());
    }
    const int64_t n = static_cast<int64_t>(spots.size());
    auto S = torch::tensor(flat).to(torch::get_default_dtype()).reshape({n, d});
    auto tt = torch::full({n, 1}, t);
    auto B = pinn.freeBoundary(tt);

    OptionPriceRow row;
    row.seed = seed;
    row.prices = toVector(pinn.price(S, tt));
    row.boundary = toVector(B.index({0}));
    for (double e : toVector(pinn.isInContinuationRegion(S, tt).logical_not())) row.exercise.push_back(static_cast<int>(e));
    auto payoff = toVector(option->payoff(S));

    std::printf("\n--- PREZZO DELL'OPZIONE (%s, Seed %lld, t = %.4f) ---\n", option->getName().c_str(), seed, t);
    std::printf("Free boundary B(t) =");
    for (double b : row.boundary) std::printf(" %.4f", b);
    int w = 4;
    for (const auto& s : spots) w = std::max(w, static_cast<int>(spotLabel(s).size()) + 2);
    std::printf("\n%-*s %12s %12s   %s\n", w, "S0", "PINN price", "Payoff", "Region");
    for (int64_t k = 0; k < n; ++k)
        std::printf("%-*s %12.6f %12.6f   %s\n", w, spotLabel(spots[k]).c_str(), row.prices[k], payoff[k],
                    row.exercise[k] ? "exercise" : "continuation");
    return row;
}

// Summary over several seeds: one column per seed, then mean and median.
void OptionPriceReport::printAll(const std::vector<OptionPriceRow>& rows,
                                 const std::vector<std::vector<double>>& spots, double t) {
    std::printf("\n\n===================== PREZZI DELL'OPZIONE (ALL SEEDS, t = %.4f) =====================\n", t);
    int w = 12;
    for (const auto& s : spots) w = std::max(w, static_cast<int>(spotLabel(s).size()) + 2);
    std::printf("%-*s", w, "S0 \\ Seed");
    for (const auto& r : rows) std::printf("%10lld", r.seed);
    std::printf("%12s%12s\n", "Mean", "Median");
    for (size_t k = 0; k < spots.size(); ++k) {
        std::vector<double> v;
        for (const auto& r : rows) v.push_back(r.prices[k]);
        std::printf("%-*s", w, spotLabel(spots[k]).c_str());
        for (double p : v) std::printf("%10.4f", p);
        std::printf("%12.6f%12.6f\n", mean(v), median(v));
    }
}

// option_prices*.csv, read by the interface: t, S1..Sd, one column per seed (+ Mean, Median).
void OptionPriceReport::writeCsv(const std::vector<OptionPriceRow>& rows,
                                 const std::vector<std::vector<double>>& spots, double t,
                                 const std::string& fileName) {
    std::ofstream f(fileName);
    const size_t d = spots.empty() ? 0 : spots[0].size();
    f << "t";
    for (size_t i = 1; i <= d; ++i) f << ",S" << i;
    for (const auto& r : rows) f << ",Seed" << r.seed;
    if (rows.size() > 1) f << ",Mean,Median";
    f << "\n";
    f.precision(8);
    for (size_t k = 0; k < spots.size(); ++k) {
        std::vector<double> v;
        for (const auto& r : rows) v.push_back(r.prices[k]);
        f << std::fixed << t;
        for (double s : spots[k]) f << "," << s;
        for (double p : v) f << "," << p;
        if (rows.size() > 1) f << "," << mean(v) << "," << median(v);
        f << "\n";
    }
    std::printf("Prices saved to: %s\n", fileName.c_str());
}

} // namespace it::univr::output
