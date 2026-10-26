#pragma once
#include <torch/torch.h>
#include <string>

namespace it::univr::neuralnetworks {

/**
 * Interface of an <b>American option</b> on d >= 1 underlyings.
 * <p>
 * Each method corresponds to one condition of the free-boundary problem, so that the PINN
 * contains nothing specific to a put or a call.
 * </p>
 * <p>
 * Shape conventions:
 * <ul>
 * <li>S (underlying prices): [N, d]</li>
 * <li>t (times): [N, 1]</li>
 * <li>B (free boundary): [N, d]</li>
 * </ul>
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class IOption {
public:
    virtual ~IOption() = default;

    virtual double getStrike() const = 0;       // K
    virtual double getMaturity() const = 0;     // T
    virtual int getNumberOfAssets() const = 0;  // d
    virtual std::string getName() const = 0;

    /** @return The payoff psi(S), shape [N, 1]. */
    virtual torch::Tensor payoff(const torch::Tensor& S) const = 0;

    /** @return The free boundary at maturity B(T), shape [1, d]. */
    virtual torch::Tensor freeBoundaryAtMaturity() const = 0;

    /** @return The value on the free boundary P(B(t), t) (Dirichlet condition), shape [N, 1]. */
    virtual torch::Tensor valueAtFreeBoundary(const torch::Tensor& B) const = 0;

    /** @return The deltas on the free boundary dP/dS_i(B(t), t) (Neumann condition), shape [N, d]. */
    virtual torch::Tensor deltaAtFreeBoundary(const torch::Tensor& B) const = 0;

    /**
     * Moves points of [0, Smax]^d onto the far boundary, one coordinate per row,
     * spreading the points evenly over the faces.
     */
    virtual torch::Tensor projectOnFarBoundary(const torch::Tensor& S, double Smax) const = 0;

    /** @return The value on the far boundary, shape [N, 1]. */
    virtual torch::Tensor farBoundaryValue(const torch::Tensor& S, const torch::Tensor& t,
                                           double riskFreeRate) const = 0;

    /** @return true where early exercise is not optimal (continuation region), shape [N, 1]. */
    virtual torch::Tensor isInContinuationRegion(const torch::Tensor& S,
                                                 const torch::Tensor& B) const = 0;
};

} // namespace it::univr::neuralnetworks
