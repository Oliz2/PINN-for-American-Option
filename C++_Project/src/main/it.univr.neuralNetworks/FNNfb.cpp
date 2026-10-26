/*
 * FNNfb.cpp
 * Free-boundary network of the PINN: t -> B(t). Used only in the two-network formulation.
 *
 * Author: Alberto Oliva Medin
 */
#include "FNNfb.h"
#include <stdexcept>

namespace it::univr::neuralnetworks {

// One input (the time), d outputs (one boundary component B_i(t) per underlying).
FNNfb::FNNfb(int numberOfAssets, int hiddenLayers, int hiddenUnits, double tScale, double outputScale)
    : AbstractNeuralNetwork(1, numberOfAssets, hiddenLayers, hiddenUnits),
      numberOfAssets_(numberOfAssets), tScale_(tScale), outputScale_(outputScale) {
    if (tScale <= 0.0 || outputScale <= 0.0)
        throw std::invalid_argument("FNNfb: tScale and outputScale must be > 0");
}

/*
 * The time is scaled to t / tScale and the output is multiplied by outputScale (= K with
 * normalisation): the network then works with numbers of order 1, while B(t) comes out in price units.
 */
torch::Tensor FNNfb::forward(const torch::Tensor& t) {
    return outputScale_ * forwardLayers(t / tScale_);
}

} // namespace it::univr::neuralnetworks
