#pragma once
#include "abstractNeuralNetork.h"

namespace it::univr::neuralnetworks {

/**
 * Neural network approximating the <b>free boundary</b>: t -> B(t) = (B_1(t), ..., B_d(t)).
 * <p>
 * The input is divided by <code>tScale</code> and the linear output is multiplied by <code>outputScale</code>.
 * With outputScale = K the network works on the order of magnitude of B/K.
 * </p>
 * <p>
 * The network is trained only in the two-network (free-boundary) formulation. With the
 * complementarity formulation the boundary is recovered from the price and FNNfb is not used.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class FNNfb : public AbstractNeuralNetwork {
public:
    /**
     * @param numberOfAssets Number of underlyings d (one boundary component per asset).
     * @param hiddenLayers   Number of hidden layers.
     * @param hiddenUnits    Number of neurons per hidden layer.
     * @param tScale         Scale of the time (1 or T).
     * @param outputScale    Scale of the output (1 or K).
     */
    FNNfb(int numberOfAssets, int hiddenLayers, int hiddenUnits, double tScale, double outputScale);

    /**
     * @param t Times, shape [N, 1].
     * @return The boundary B(t), shape [N, d].
     */
    torch::Tensor forward(const torch::Tensor& t);

    /** @return The number of underlyings d. */
    int getNumberOfAssets() const { return numberOfAssets_; }
    std::string getName() const override { return "FNNfb"; }

private:
    int numberOfAssets_;            // d
    double tScale_, outputScale_;   // input t / tScale, output multiplied by outputScale
};

} // namespace it::univr::neuralnetworks
