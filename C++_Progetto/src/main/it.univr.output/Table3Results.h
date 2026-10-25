#pragma once
#include <string>
#include <vector>

namespace it::univr::output {

/**
 * Losses on the test set for one seed (columns of Table 3 of the reference paper).
 * unsupervised = PDE residual in the continuation region, freeBoundary = fbInit + fbDir + fbNeu,
 * solution = unsupervised + initial + dirichlet + fbDir + fbNeu.
 */
struct Table3Row {
    long long seed;
    double unsupervised, initial, dirichlet, fbInit, fbDir, fbNeu, freeBoundary, solution;
};

/**
 * Prints and saves the <b>test-set losses</b> of all the seeds, with mean and median.
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class Table3Results {
public:
    /** Prints the losses of one seed. */
    static void printRow(const Table3Row& row);
    /** Prints all the seeds with mean and median. */
    static void printAll(const std::vector<Table3Row>& rows);
    /** Writes all the seeds, with mean and median, to a CSV file. */
    static void writeCsv(const std::vector<Table3Row>& rows, const std::string& fileName);

private:
    static std::vector<double> column(const std::vector<Table3Row>& rows, int col);
    static double mean(const std::vector<double>& v);
    static double median(std::vector<double> v);
};

} // namespace it::univr::output
