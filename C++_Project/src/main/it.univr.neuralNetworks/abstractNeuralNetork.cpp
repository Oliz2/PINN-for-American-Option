/*
 * abstractNeuralNetork.cpp
 * Construction, initialisation and forward pass of the feed-forward networks used by the PINN.
 *
 * Author: Alberto Oliva Medin
 */
#include "abstractNeuralNetork.h"
#include <sstream>
#include <stdexcept>

namespace it::univr::neuralnetworks {

/*
 * Builds the fully connected layers and registers them in the torch::nn::Module, so that
 * parameters() returns their weights and biases to the optimiser.
 * The layers are named fc0, fc1, ..., out: the names identify the parameters when saving a model.
 */
AbstractNeuralNetwork::AbstractNeuralNetwork(int inputDim, int outputDim,
                                             int hiddenLayers, int hiddenUnits)
    : inputDim_(inputDim), outputDim_(outputDim),
      hiddenLayers_(hiddenLayers), hiddenUnits_(hiddenUnits) {
    if (inputDim <= 0 || outputDim <= 0 || hiddenLayers <= 0 || hiddenUnits <= 0)
        throw std::invalid_argument("The network dimensions must be > 0");

    // input -> hidden, (hiddenLayers - 1) x hidden -> hidden, hidden -> output
    layers_.push_back(register_module("fc0", torch::nn::Linear(inputDim, hiddenUnits)));
    for (int i = 1; i < hiddenLayers; ++i)
        layers_.push_back(register_module("fc" + std::to_string(i),
                                          torch::nn::Linear(hiddenUnits, hiddenUnits)));
    layers_.push_back(register_module("out", torch::nn::Linear(hiddenUnits, outputDim)));

    initializeWeights();
}

/*
 * Glorot (Xavier) normal initialisation of the weights and zero biases.
 * Xavier keeps the variance of the activations roughly constant from layer to layer, which
 * suits tanh networks: the signal neither vanishes nor saturates at the start of the training.
 */
void AbstractNeuralNetwork::initializeWeights() {
    torch::NoGradGuard noGrad;   // the initialisation must not enter the autograd graph
    for (auto& layer : layers_) {
        torch::nn::init::xavier_normal_(layer->weight);
        torch::nn::init::zeros_(layer->bias);
    }
}

/*
 * Forward pass: tanh after every hidden layer, identity after the last one.
 * The output is linear so that the network can return any real value; the concrete networks
 * (FNNSol, FNNfb) decide how the inputs are scaled and how the output is interpreted.
 */
torch::Tensor AbstractNeuralNetwork::forwardLayers(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == inputDim_,
                getName(), ": expected input of shape [N, ", inputDim_, "]");
    auto h = x;
    for (size_t i = 0; i + 1 < layers_.size(); ++i)   // hidden layers: Linear + tanh
        h = torch::tanh(layers_[i](h));
    return layers_.back()(h);                         // output layer: Linear only
}

// Total number of trainable parameters: sum of the sizes of all weight matrices and bias vectors.
int64_t AbstractNeuralNetwork::numberOfParameters() const {
    int64_t n = 0;
    for (const auto& p : parameters()) n += p.numel();
    return n;
}

// One-line summary of the architecture, printed in the log at the start of a run.
std::string AbstractNeuralNetwork::describe() const {
    std::ostringstream os;
    os << getName() << ": input " << inputDim_ << ", "
       << hiddenLayers_ << " hidden layers x " << hiddenUnits_ << " neurons (tanh), "
       << "output " << outputDim_ << ", " << numberOfParameters() << " parameters";
    return os.str();
}

} // namespace it::univr::neuralnetworks
