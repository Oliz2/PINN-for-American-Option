/*
 * SobolSampler.cpp
 * Sobol low-discrepancy sequence: the points fill [0, 1)^d more evenly than random points, so fewer
 * collocation points are needed to cover the domain.
 *
 * Author: Alberto Oliva Medin
 */
#include "SobolSampler.h"
#include "SobolDirectionNumbers.h"
#include <stdexcept>
#include <string>

namespace it::univr::utils {

namespace {
constexpr int kBits = 32;   // every coordinate is a 32-bit binary fraction
}

/*
 * Computes the direction numbers v_{j,k} of every dimension j. For j >= 1 they come from the primitive
 * polynomial of degree s with coefficients a and the initial values m_1..m_s (Joe and Kuo), extended with
 * the recurrence v_k = v_{k-s} xor (v_{k-s} >> s) xor sum_i a_i v_{k-i}.
 */
SobolSampler::SobolSampler(int dimension) : dimension_(dimension) {
    if (dimension < 1 || dimension > kMaxDimension)
        throw std::invalid_argument("SobolSampler: dimensione supportata da 1 a " +
                                    std::to_string(kMaxDimension));

    directions_.assign(dimension, std::vector<uint32_t>(kBits + 1, 0));
    state_.assign(dimension, 0);

    // First dimension: v_k = 2^(32-k) (van der Corput sequence)
    for (int k = 1; k <= kBits; ++k) directions_[0][k] = 1u << (kBits - k);

    for (int j = 1; j < dimension; ++j) {
        const sobol::JoeKuo& p = sobol::kTable[j - 1];
        auto& v = directions_[j];
        for (unsigned k = 1; k <= p.s && k <= kBits; ++k) v[k] = p.m[k - 1] << (kBits - k);
        for (int k = p.s + 1; k <= kBits; ++k) {
            v[k] = v[k - p.s] ^ (v[k - p.s] >> p.s);
            for (unsigned i = 1; i < p.s; ++i)
                if ((p.a >> (p.s - 1 - i)) & 1u) v[k] ^= v[k - i];
        }
    }
}

// Returns the current point and moves to the next one (one xor per dimension).
std::vector<double> SobolSampler::next() {
    std::vector<double> point(dimension_);
    for (int j = 0; j < dimension_; ++j) point[j] = static_cast<double>(state_[j]) / 4294967296.0;

    // Gray code: flip the direction of the least significant zero bit of index_
    int c = 1;
    uint64_t value = index_;
    while (value & 1u) { value >>= 1; ++c; }
    if (c > kBits) throw std::runtime_error("SobolSampler: sequenza esaurita");
    for (int j = 0; j < dimension_; ++j) state_[j] ^= directions_[j][c];
    ++index_;
    return point;
}

// n consecutive points after skipping the first `skip` ones (the origin by default), as a tensor [n, dimension].
torch::Tensor SobolSampler::generate(int64_t n, int dimension, int64_t skip) {
    SobolSampler sampler(dimension);
    for (int64_t i = 0; i < skip; ++i) sampler.next();
    auto out = torch::empty({n, dimension}, torch::kDouble);
    auto acc = out.accessor<double, 2>();
    for (int64_t i = 0; i < n; ++i) {
        auto p = sampler.next();
        for (int j = 0; j < dimension; ++j) acc[i][j] = p[j];
    }
    return out.to(torch::get_default_dtype());
}

} // namespace it::univr::utils
