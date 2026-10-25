#pragma once
#include "abstractNeuralNetork.h"

namespace it::univr::neuralnetworks {

/**
 * Neural network approximating the <b>option price</b>: (S_1, ..., S_d, t) -> P(S, t).
 * <p>
 * The inputs are divided by <code>sScale</code> and <code>tScale</code> before entering the network.
 * With sScale = tScale = 1 the network sees the raw inputs; with sScale = K and tScale = T it works
 * on S/K and t/T. Derivatives computed by autograd are always taken with respect to the original S and t.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class FNNSol : public AbstractNeuralNetwork {
public:
    /**
     * @param numberOfAssets Number of underlyings d.
     * @param hiddenLayers   Number of hidden layers.
     * @param hiddenUnits    Number of neurons per hidden layer.
     * @param sScale         Scale of the underlying prices (1 or K).
     * @param tScale         Scale of the time (1 or T).
     */
    FNNSol(int numberOfAssets, int hiddenLayers, int hiddenUnits, double sScale, double tScale);

    /**
     * @param S Underlying prices, shape [N, d].
     * @param t Times, shape [N, 1].
     * @return The price P(S, t), shape [N, 1].
     */
    torch::Tensor forward(const torch::Tensor& S, const torch::Tensor& t);

    /** @return The number of underlyings d. */
    int getNumberOfAssets() const { return numberOfAssets_; }
    std::string getName() const override { return "FNNSol"; }

private:
    int numberOfAssets_;        // d
    double sScale_, tScale_;    // the inputs enter the network as (S / sScale, t / tScale)
};

} // namespace it::univr::neuralnetworks
