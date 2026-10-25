#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>
#include "PINN.h"

namespace it::univr::neuralnetworks {

/**
 * Exponential learning-rate decay (non-staircase):
 * lr = initialLearningRate * decayRate^(optimizerSteps / decaySteps).
 * The step counter advances at every update of the optimiser.
 */
struct LearningRateSchedule {
    double initialLearningRate;   // learning rate at the first update
    int decaySteps;               // number of updates after which the rate is multiplied by decayRate
    double decayRate;             // factor in (0, 1]
};

/**
 * Hyper-parameters of the training. There are no default values: everything is set by the caller.
 */
struct TrainingConfig {
    int epochs;
    int solutionStepsPerBoundaryStep;    // n_steps: steps of FNNSol for every step of FNNfb
    bool separateBoundaryOptimizer;      // false: one RMSprop for both networks; true: one per network
    LearningRateSchedule solution;       // schedule of FNNSol (or of the single optimiser)
    LearningRateSchedule boundary;       // schedule of FNNfb (used only with separateBoundaryOptimizer)
    double rmspropRho;                   // decay of the moving average of the squared gradients
    double rmspropEpsilon;               // added to the denominator of RMSprop to avoid divisions by zero
    bool verbose;                        // true: print every inner step and every loss component
    int logEvery;                        // if verbose = false: summary every N epochs (0 = never)
};

/** Losses recorded at the joint step of each epoch (one row of training_history.csv). */
struct EpochRecord {
    int epoch;
    double solutionLearningRate, boundaryLearningRate;
    double pde, initial, dirichlet;
    double fbInitial, fbDirichlet, fbNeumann;
    double solutionLoss, freeBoundaryLoss;
};

/**
 * <b>Training loop</b> of the PINN, optimised with RMSprop.
 * <p>
 * Each epoch performs:
 * <ol>
 * <li>n_steps - 1 steps of the price network FNNSol only, with the free boundary kept fixed;</li>
 * <li>one joint step: the losses are evaluated once, then FNNfb and FNNSol are updated
 *     with gradients computed on the weights before either update.</li>
 * </ol>
 * With the penalty formulation the joint step updates FNNSol only.
 * The collocation points are sampled once and kept fixed during the training.
 * </p>
 * <p>
 * RMSprop divides each gradient by the square root of a moving average of its squares,
 * so every weight gets its own step size: v = rho v + (1 - rho) g^2, w = w - lr g / (sqrt(v) + eps).
 * </p>
 * <p>
 * The two losses are kept separate: the gradient of L_sol is taken only with respect to the weights
 * of FNNSol and the gradient of L_fb only with respect to the weights of FNNfb.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class TrainNeuralNetwork {
public:
    /**
     * @param pinn        The PINN to train.
     * @param trainingSet The collocation points.
     * @param config      The training hyper-parameters (validated here).
     */
    TrainNeuralNetwork(PINN& pinn, CollocationSet trainingSet, TrainingConfig config);

    /** Runs all the epochs, printing the progress on the standard output. */
    void train();

    /** @return The losses recorded at every epoch. */
    const std::vector<EpochRecord>& getHistory() const { return history_; }

private:
    /** An RMSprop optimiser together with its learning-rate schedule and step counter. */
    struct OptimizerState {
        std::unique_ptr<torch::optim::RMSprop> optimizer;
        std::vector<torch::Tensor> params;   // all the parameters managed by this optimiser
        LearningRateSchedule schedule;
        int64_t steps = 0;                   // number of updates performed
        /** @return The current learning rate of the schedule. */
        double learningRate() const;
    };

    /** Throws std::invalid_argument if a hyper-parameter is out of range. */
    static void validate(const TrainingConfig& c);
    /** One step of FNNSol with the boundary fixed. @return The value of L_sol. */
    double solutionStep();
    /** One joint step of both networks. @return All the loss terms evaluated at this step. */
    PINNLosses jointStep();
    /** Writes the gradients into the given parameters only, sets the learning rate and makes one RMSprop step. */
    static void applyGradients(OptimizerState& state, const std::vector<torch::Tensor>& params,
                               const std::vector<torch::Tensor>& grads);
    /** Detailed print of the joint step (verbose mode). */
    void printJointStep(const EpochRecord& rec, double elapsedSeconds) const;

    PINN& pinn_;
    CollocationSet set_;      // fixed collocation points
    TrainingConfig config_;

    std::vector<torch::Tensor> solParams_;    // weights and biases of FNNSol
    std::vector<torch::Tensor> fbParams_;     // weights and biases of FNNfb
    std::shared_ptr<OptimizerState> solOpt_;
    std::shared_ptr<OptimizerState> fbOpt_;    // = solOpt_ when separateBoundaryOptimizer is false

    std::vector<EpochRecord> history_;        // one record per epoch
};

} // namespace it::univr::neuralnetworks
