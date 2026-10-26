/*
 * Table3Results.cpp
 * Test-set losses in the format of Table 3 of the reference paper (Gatta et al., 2023), so that the
 * values can be compared with the published ones.
 *
 * Author: Alberto Oliva Medin
 */
#include "Table3Results.h"
#include <algorithm>
#include <cstdio>
#include <fstream>

namespace it::univr::output {

static const char* kHeader =
    "Seed\tUnsupervised\tInitial\t\tDirichlet\tFB_Init\t\tFB_Dir\t\tFB_Neu\t\tFree boundary\tSolution\n";
static const char* kLine =
    "------------------------------------------------------------------------------------------------------------------------------------------------------------";

// Losses of one seed, printed right after its test-set evaluation.
void Table3Results::printRow(const Table3Row& r) {
    std::printf("\n--- TABLE 3 FORMAT RESULTS (SEED %lld) ---\n", r.seed);
    std::printf("%s", kHeader);
    std::printf("%lld\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n", r.seed, r.unsupervised, r.initial,
                r.dirichlet, r.fbInit, r.fbDir, r.fbNeu, r.freeBoundary, r.solution);
}

// Values of one column (0 = Unsupervised, ..., 7 = Solution) over all the seeds.
std::vector<double> Table3Results::column(const std::vector<Table3Row>& rows, int col) {
    std::vector<double> v;
    for (const auto& r : rows) {
        const double cols[] = {r.unsupervised, r.initial, r.dirichlet, r.fbInit,
                               r.fbDir, r.fbNeu, r.freeBoundary, r.solution};
        v.push_back(cols[col]);
    }
    return v;
}

double Table3Results::mean(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / v.size();
}

double Table3Results::median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size(), mid = n / 2;
    return (n % 2 == 0) ? (v[mid - 1] + v[mid]) / 2.0 : v[mid];
}

// Table with all the seeds, plus mean and median of every column.
void Table3Results::printAll(const std::vector<Table3Row>& rows) {
    std::printf("\n\n===================== TABLE 3 (ALL SEEDS) =====================\n");
    std::printf("%s%s\n", kHeader, kLine);
    for (const auto& r : rows)
        std::printf("%lld\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n", r.seed, r.unsupervised,
                    r.initial, r.dirichlet, r.fbInit, r.fbDir, r.fbNeu, r.freeBoundary, r.solution);
    std::printf("Mean");
    for (int c = 0; c < 8; ++c) std::printf("\t%.6e", mean(column(rows, c)));
    std::printf("\nMedian");
    for (int c = 0; c < 8; ++c) std::printf("\t%.6e", median(column(rows, c)));
    std::printf("\n%s\n", kLine);
}

// table3_results*.csv, read by the interface (test-set table).
void Table3Results::writeCsv(const std::vector<Table3Row>& rows, const std::string& fileName) {
    std::ofstream f(fileName);
    f << "Seed,Unsupervised,Initial,Dirichlet,FB_Init,FB_Dir,FB_Neu,FreeBoundary,Solution\n";
    f.precision(6);
    for (const auto& r : rows)
        f << r.seed << std::scientific << "," << r.unsupervised << "," << r.initial << "," << r.dirichlet
          << "," << r.fbInit << "," << r.fbDir << "," << r.fbNeu << "," << r.freeBoundary << ","
          << r.solution << "\n" << std::defaultfloat;
    f << std::scientific << "Mean";
    for (int c = 0; c < 8; ++c) f << "," << mean(column(rows, c));
    f << "\nMedian";
    for (int c = 0; c < 8; ++c) f << "," << median(column(rows, c));
    f << "\n";
    std::printf("\nResults saved to: %s\n", fileName.c_str());
}

} // namespace it::univr::output
