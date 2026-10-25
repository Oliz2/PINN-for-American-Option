#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace it::univr::utils {

/**
 * Generator of <b>Sobol low-discrepancy sequences</b> in [0, 1)^dim.
 * <p>
 * Direction numbers of Joe and Kuo (new-joe-kuo-6.21201), construction with the Gray code.
 * Up to <code>kMaxDimension</code> dimensions. The first point (all zeros) is skipped by default.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class SobolSampler {
public:
    static constexpr int kMaxDimension = 1024;   // size of the table in SobolDirectionNumbers.h

    /** @param dimension Dimension of the points (1 to kMaxDimension). */
    explicit SobolSampler(int dimension);

    /** @return The next point of the sequence (dim values in [0, 1)). */
    std::vector<double> next();

    /**
     * @param n         Number of points.
     * @param dimension Dimension of each point.
     * @param skip      Number of initial points to skip.
     * @return n consecutive points as a tensor of shape [n, dimension].
     */
    static torch::Tensor generate(int64_t n, int dimension, int64_t skip = 1);

private:
    int dimension_;
    uint64_t index_ = 0;                             // index of the next point
    std::vector<uint32_t> state_;                    // current x_j for every dimension
    std::vector<std::vector<uint32_t>> directions_;  // v_{j,k}, k = 1..32
};

} // namespace it::univr::utils
