#pragma once
#include "IOption.h"

namespace it::univr::neuralnetworks {

/**
 * <b>American put on the minimum</b> of d underlyings: psi(S) = (K - min_i S_i)^+.
 * <p>
 * With d = 1 it is the standard American put.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class PutOption : public IOption {
public:
    /**
     * @param strike         Strike K (> 0).
     * @param maturity       Maturity T in years (> 0).
     * @param numberOfAssets Number of underlyings d (>= 1).
     */
    PutOption(double strike, double maturity, int numberOfAssets);

    double getStrike() const override { return strike_; }
    double getMaturity() const override { return maturity_; }
    int getNumberOfAssets() const override { return numberOfAssets_; }
    std::string getName() const override;

    torch::Tensor payoff(const torch::Tensor& S) const override;
    torch::Tensor freeBoundaryAtMaturity() const override;
    torch::Tensor valueAtFreeBoundary(const torch::Tensor& B) const override;
    torch::Tensor deltaAtFreeBoundary(const torch::Tensor& B) const override;
    torch::Tensor projectOnFarBoundary(const torch::Tensor& S, double Smax) const override;
    torch::Tensor farBoundaryValue(const torch::Tensor& S, const torch::Tensor& t,
                                   double riskFreeRate) const override;
    torch::Tensor isInContinuationRegion(const torch::Tensor& S,
                                         const torch::Tensor& B) const override;

private:
    double strike_;         // K
    double maturity_;       // T
    int numberOfAssets_;    // d
};

} // namespace it::univr::neuralnetworks
