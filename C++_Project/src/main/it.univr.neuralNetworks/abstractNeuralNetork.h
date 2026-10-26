#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

namespace it::univr::neuralnetworks {

/**
 * Base class of the <b>feed-forward neural networks</b> used by the PINN.
 * <p>
 * Architecture: input -> [Linear + tanh] x hiddenLayers -> Linear (identity output).
 * Weights are initialised with Glorot (Xavier) normal, biases with zero.
 * </p>
 * <p>
 * The class extends <code>torch::nn::Module</code>, which handles parameters, gradients and optimisers.
 * Concrete networks only decide how the inputs are prepared and how the output is scaled.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class AbstractNeuralNetwork : public torch::nn::Module {
public:
    virtual ~AbstractNeuralNetwork() = default;

    /** @return The name of the network, used in the log. */
    virtual std::string getName() const = 0;

    int getInputDim() const { return inputDim_; }
    int getOutputDim() const { return outputDim_; }
    int getHiddenLayers() const { return hiddenLayers_; }
    int getHiddenUnits() const { return hiddenUnits_; }

    /** @return The total number of trainable parameters (weights and biases). */
    int64_t numberOfParameters() const;

    /** @return A one-line description of the architecture. */
    std::string describe() const;

protected:
    /**
     * Builds the layers and initialises the weights.
     *
     * @param inputDim     Number of inputs.
     * @param outputDim    Number of outputs.
     * @param hiddenLayers Number of hidden layers.
     * @param hiddenUnits  Number of neurons per hidden layer.
     */
    AbstractNeuralNetwork(int inputDim, int outputDim, int hiddenLayers, int hiddenUnits);

    /**
     * Evaluates the network.
     *
     * @param x Input tensor of shape [N, inputDim], already scaled.
     * @return Output tensor of shape [N, outputDim].
     */
    torch::Tensor forwardLayers(const torch::Tensor& x);

private:
    /** Xavier normal weights and zero biases, outside the autograd graph. */
    void initializeWeights();

    int inputDim_;
    int outputDim_;
    int hiddenLayers_;
    int hiddenUnits_;
    std::vector<torch::nn::Linear> layers_;   // hidden layers followed by the output layer
};

} // namespace it::univr::neuralnetworks