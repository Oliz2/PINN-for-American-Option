/*
 * FNNSol.cpp
 * Price network of the PINN: (S, t) -> P(S, t).
 *
 * Author: Alberto Oliva Medin
 */
#include "FNNSol.h"
#include <stdexcept>

namespace it::univr::neuralnetworks {

// d + 1 inputs (the d underlyings and the time), one output (the price).
FNNSol::FNNSol(int numberOfAssets, int hiddenLayers, int hiddenUnits, double sScale, double tScale)
    : AbstractNeuralNetwork(numberOfAssets + 1, 1, hiddenLayers, hiddenUnits),
      numberOfAssets_(numberOfAssets), sScale_(sScale), tScale_(tScale) {
    if (sScale <= 0.0 || tScale <= 0.0) throw std::invalid_argument("FNNSol: sScale e tScale devono essere > 0");
}

/*
 * Input normalisation: the network sees (S / sScale, t / tScale), i.e. (S/K, t/T) when the
 * normalisation is active. The division is part of the autograd graph, so dP/dS and dP/dt
 * computed later are still derivatives with respect to the original S and t.
 */
torch::Tensor FNNSol::forward(const torch::Tensor& S, const torch::Tensor& t) {
    return forwardLayers(torch::cat({S / sScale_, t / tScale_}, /*dim=*/1));
}

} // namespace it::univr::neuralnetworks
