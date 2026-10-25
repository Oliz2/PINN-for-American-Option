/*
 * Putoption.cpp
 * American put on the minimum of d underlyings: payoff, boundary values and exercise region.
 * Every method returns a tensor, so that the PINN can evaluate the conditions on whole batches
 * of collocation points at once.
 *
 * Author: Alberto Oliva Medin
 */
#include "Putoption.h"
#include <stdexcept>
#include <string>

namespace it::univr::neuralnetworks {

PutOption::PutOption(double strike, double maturity, int numberOfAssets)
    : strike_(strike), maturity_(maturity), numberOfAssets_(numberOfAssets) {
    if (strike <= 0.0) throw std::invalid_argument("Lo strike K deve essere > 0");
    if (maturity <= 0.0) throw std::invalid_argument("La maturity T deve essere > 0");
    if (numberOfAssets < 1) throw std::invalid_argument("Il numero di sottostanti deve essere >= 1");
}

std::string PutOption::getName() const {
    return numberOfAssets_ == 1 ? "American Put"
                                : "American Put on min of " + std::to_string(numberOfAssets_) + " assets";
}

// Minimum of every row of a [N, d] tensor, returned as a column [N, 1].
static torch::Tensor rowMin(const torch::Tensor& X) {
    return std::get<0>(X.min(/*dim=*/1, /*keepdim=*/true));
}

// Payoff psi(S) = (K - min S)^+
torch::Tensor PutOption::payoff(const torch::Tensor& S) const {
    return torch::clamp_min(strike_ - rowMin(S), 0.0);
}

// At maturity the exercise boundary equals the strike: B(T) = (K, ..., K)
torch::Tensor PutOption::freeBoundaryAtMaturity() const {
    return torch::full({1, numberOfAssets_}, strike_);
}

// Value matching: on the boundary the price equals the payoff, P(B, t) = K - min B
torch::Tensor PutOption::valueAtFreeBoundary(const torch::Tensor& B) const {
    return strike_ - rowMin(B);
}

/*
 * Smooth pasting: on the boundary the price has the slope of the payoff.
 * dP/dS_i = -1 for the asset that is the minimum (i = argmin B), 0 for the others (always -1 for d = 1).
 * argmin is taken on B.detach(): the choice of the minimum asset is not differentiated.
 */
torch::Tensor PutOption::deltaAtFreeBoundary(const torch::Tensor& B) const {
    auto target = torch::zeros_like(B);
    target.scatter_(1, B.detach().argmin(1, /*keepdim=*/true), -1.0);
    return target;
}

/*
 * For the put the far boundary is S_i = Smax. The points are split evenly among the d faces
 * in contiguous blocks: the first N/d rows get S_1 = Smax, the next ones S_2 = Smax, and so on.
 * An alternating row-by-row assignment would be correlated with the Sobol sequence and would
 * leave parts of the faces uncovered.
 */
torch::Tensor PutOption::projectOnFarBoundary(const torch::Tensor& S, double Smax) const {
    auto out = S.clone();
    const int64_t n = S.size(0);
    // face[k] = floor(k * d / n): index of the coordinate set to Smax in row k
    auto face = (torch::arange(n, torch::kLong) * numberOfAssets_).div(n, "floor").unsqueeze(1);
    out.scatter_(1, face, Smax);
    return out;
}

/*
 * Value on the far boundary.
 * d = 1: P(Smax, t) = 0, because a put deep out of the money is worth nothing.
 * d > 1: only one asset is at Smax, the others can still be low, so the value is approximated by the
 * payoff with discounted strike, (K e^{-r(T-t)} - min S)^+.
 */
torch::Tensor PutOption::farBoundaryValue(const torch::Tensor& S, const torch::Tensor& t,
                                          double riskFreeRate) const {
    if (numberOfAssets_ == 1) return torch::zeros({S.size(0), 1}, S.options());
    auto discountedStrike = strike_ * torch::exp(-riskFreeRate * (maturity_ - t));
    return torch::clamp_min(discountedStrike - rowMin(S), 0.0);
}

// Continuation region (two-network formulation): S_i > B_i(t) for every i
torch::Tensor PutOption::isInContinuationRegion(const torch::Tensor& S,
                                                const torch::Tensor& B) const {
    return (S > B).all(/*dim=*/1, /*keepdim=*/true);
}

} // namespace it::univr::neuralnetworks
