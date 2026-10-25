/*
 * TrainNeuralNetwork.cpp
 * Training loop of the PINN: RMSprop optimisers, learning-rate schedule, inner and joint steps,
 * progress output.
 *
 * Author: Alberto Oliva Medin
 */
#include "TrainNeuralNetwork.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace it::univr::neuralnetworks {

// Checks one learning-rate schedule (name is used in the error message).
static void validateSchedule(const LearningRateSchedule& s, const std::string& name) {
    if (s.initialLearningRate <= 0.0)
        throw std::invalid_argument("TrainingConfig." + name + ".initialLearningRate deve essere > 0");
    if (s.decaySteps <= 0)
        throw std::invalid_argument("TrainingConfig." + name + ".decaySteps deve essere > 0");
    if (s.decayRate <= 0.0 || s.decayRate > 1.0)
        throw std::invalid_argument("TrainingConfig." + name + ".decayRate deve essere in (0, 1]");
}

// Checks all the hyper-parameters before the training starts, so that a wrong configuration fails immediately.
void TrainNeuralNetwork::validate(const TrainingConfig& c) {
    if (c.epochs <= 0)
        throw std::invalid_argument("TrainingConfig.epochs deve essere > 0");
    if (c.solutionStepsPerBoundaryStep < 1)
        throw std::invalid_argument("TrainingConfig.solutionStepsPerBoundaryStep deve essere >= 1");
    validateSchedule(c.solution, "solution");
    if (c.separateBoundaryOptimizer) validateSchedule(c.boundary, "boundary");
    if (c.rmspropRho <= 0.0 || c.rmspropRho >= 1.0)
        throw std::invalid_argument("TrainingConfig.rmspropRho deve essere in (0, 1)");
    if (c.rmspropEpsilon <= 0.0)
        throw std::invalid_argument("TrainingConfig.rmspropEpsilon deve essere > 0");
    if (c.logEvery < 0)
        throw std::invalid_argument("TrainingConfig.logEvery deve essere >= 0");
}

// Exponential decay: lr = lr0 * decayRate^(steps / decaySteps), continuous in the number of steps.
double TrainNeuralNetwork::OptimizerState::learningRate() const {
    return schedule.initialLearningRate *
           std::pow(schedule.decayRate, static_cast<double>(steps) / schedule.decaySteps);
}

/*
 * Collects the parameters of the two networks and creates the optimiser(s).
 * alpha in LibTorch is the rho of RMSprop (decay of the moving average of g^2).
 */
TrainNeuralNetwork::TrainNeuralNetwork(PINN& pinn, CollocationSet trainingSet, TrainingConfig config)
    : pinn_(pinn), set_(std::move(trainingSet)), config_(config) {
    validate(config_);

    solParams_ = pinn_.getSolutionNet()->parameters();
    fbParams_  = pinn_.getBoundaryNet()->parameters();

    auto makeState = [&](std::vector<torch::Tensor> params, const LearningRateSchedule& schedule) {
        auto state = std::make_shared<OptimizerState>();
        state->params = params;
        state->schedule = schedule;
        state->optimizer = std::make_unique<torch::optim::RMSprop>(
            params, torch::optim::RMSpropOptions(schedule.initialLearningRate)
                        .alpha(config_.rmspropRho)
                        .eps(config_.rmspropEpsilon));
        return state;
    };

    if (config_.separateBoundaryOptimizer) {
        // One optimiser per network, each with its own schedule
        solOpt_ = makeState(solParams_, config_.solution);
        fbOpt_  = makeState(fbParams_, config_.boundary);
    } else {
        // A single RMSprop for both networks, with one step counter driving the learning rate
        std::vector<torch::Tensor> all = solParams_;
        all.insert(all.end(), fbParams_.begin(), fbParams_.end());
        solOpt_ = makeState(all, config_.solution);
        fbOpt_  = solOpt_;
    }
}

// Updates ONLY the given parameters with the given gradients and advances the step counter
void TrainNeuralNetwork::applyGradients(OptimizerState& state, const std::vector<torch::Tensor>& params,
                                        const std::vector<torch::Tensor>& grads) {
    // The other parameters keep no gradient: RMSprop skips them and leaves their state untouched
    for (auto& p : state.params) p.mutable_grad() = torch::Tensor();
    for (size_t i = 0; i < params.size(); ++i) params[i].mutable_grad() = grads[i];

    const double lr = state.learningRate();
    for (auto& group : state.optimizer->param_groups())
        static_cast<torch::optim::RMSpropOptions&>(group.options()).lr(lr);

    state.optimizer->step();
    ++state.steps;
}

/*
 * Free boundary fixed: only FNNSol is updated, with its full loss.
 * computeLosses(..., false) detaches B(t), and the gradient is requested only for solParams_.
 */
double TrainNeuralNetwork::solutionStep() {
    auto L = pinn_.computeLosses(set_, /*trainBoundary=*/false);
    auto loss = L.solutionLoss();
    auto grads = torch::autograd::grad({loss}, solParams_);
    applyGradients(*solOpt_, solParams_, grads);
    return loss.item<double>();
}

/*
 * One evaluation of the losses, then update of FNNfb and of FNNSol,
 * both with gradients computed on the weights before either update.
 * Each network receives only the gradient of its own loss: d L_fb / d(weights of FNNfb) and
 * d L_sol / d(weights of FNNSol). retain_graph = true on the first call keeps the graph alive
 * for the second one, because the two losses share part of it (FB_Dir and FB_Neu).
 */
PINNLosses TrainNeuralNetwork::jointStep() {
    auto L = pinn_.computeLosses(set_, /*trainBoundary=*/true);

    // Penalty formulation: there is no free boundary to train, only FNNSol is updated
    if (!pinn_.usesFreeBoundaryNetwork()) {
        auto gradSol = torch::autograd::grad({L.solutionLoss()}, solParams_);
        applyGradients(*solOpt_, solParams_, gradSol);
        return L;
    }

    auto gradFb  = torch::autograd::grad({L.freeBoundaryLoss()}, fbParams_, {}, /*retain_graph=*/true);
    auto gradSol = torch::autograd::grad({L.solutionLoss()}, solParams_);

    applyGradients(*fbOpt_, fbParams_, gradFb);
    applyGradients(*solOpt_, solParams_, gradSol);
    return L;
}

// Verbose output of one epoch: every loss component of both networks (the interface runs with verbose = false).
void TrainNeuralNetwork::printJointStep(const EpochRecord& rec, double elapsedSeconds) const {
    const int E = config_.epochs;
    const int n = config_.solutionStepsPerBoundaryStep;
    const int e = rec.epoch;
    const bool monitorOnly = !pinn_.usesFreeBoundaryNetwork();
    std::cout << "Epoch [" << e << "/" << E << "] - Joint Step [" << n << "/" << n << "]\n"
              << "Epoch [" << e << "/" << E << "] -> Loss FNN^fb (Boundary): " << rec.freeBoundaryLoss
              << (monitorOnly ? "   (penalty: frontiera ricavata dal prezzo, solo monitoraggio)" : "") << "\n"
              << "    [FB_Init component]: " << rec.fbInitial   << "\n"
              << "    [FB_Dir component]:  " << rec.fbDirichlet << "\n"
              << "    [FB_Neu component]:  " << rec.fbNeumann   << "\n"
              << "Epoch [" << e << "/" << E << "] - Final Loss -> Loss FNN^sol: " << rec.solutionLoss << "\n"
              << "    [PDE component]:       " << rec.pde        << "\n"
              << "    [Initial component]:   " << rec.initial    << "\n"
              << "    [Dirichlet component]: " << rec.dirichlet
              << (monitorOnly ? "   (bordo lontano + bordo S = 0)" : "") << "\n"
              << "    [FB_Dir component]:    " << rec.fbDirichlet << (monitorOnly ? "   (non nella loss)" : "") << "\n"
              << "    [FB_Neu component]:    " << rec.fbNeumann  << (monitorOnly ? "   (non nella loss)" : "") << "\n"
              << "Tempo trascorso: " << std::fixed << std::setprecision(1) << elapsedSeconds << " s\n"
              << std::defaultfloat << std::setprecision(16)
              << "---------------------------------------------------" << std::endl;
}

/*
 * Main loop. Every epoch: (n_steps - 1) inner steps of FNNSol, then one joint step whose losses are
 * stored in the history.
 * The first line ("Addestramento ... epoche") and the summary line printed every logEvery epochs are
 * parsed by the Python interface (runner.py) to draw the progress: keep their format unchanged.
 */
void TrainNeuralNetwork::train() {
    const int nSteps = config_.solutionStepsPerBoundaryStep;
    const int E = config_.epochs;
    const auto start = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };

    std::cout << std::setprecision(16);
    std::cout << "Addestramento " << pinn_.getOption()->getName() << ": " << E << " epoche, "
              << nSteps << " passi di FNNSol per ogni passo di FNNfb, "
              << (config_.separateBoundaryOptimizer ? "due ottimizzatori" : "un ottimizzatore")
              << std::endl;

    for (int epoch = 1; epoch <= E; ++epoch) {
        if (config_.verbose) {
            std::cout << "[lr schedule] epoch " << epoch << " -> learning rate = " << solOpt_->learningRate();
            if (config_.separateBoundaryOptimizer) std::cout << " (FNNfb: " << fbOpt_->learningRate() << ")";
            std::cout << std::endl;
        }

        // n_steps - 1 steps of the price network only
        for (int k = 1; k < nSteps; ++k) {
            const double loss = solutionStep();
            if (config_.verbose)
                std::cout << "Epoch [" << epoch << "/" << E << "] - Inner Step [" << k << "/" << nSteps
                          << "] -> Loss FNN^sol: " << loss << std::endl;
        }

        // Joint step (the learning rates are read before the step, i.e. the ones actually used)
        const double lrSol = solOpt_->learningRate();
        const double lrFb = fbOpt_->learningRate();
        PINNLosses L = jointStep();

        EpochRecord rec{epoch, lrSol, lrFb,
                        L.pde.item<double>(), L.initial.item<double>(), L.dirichlet.item<double>(),
                        L.fbInitial.item<double>(), L.fbDirichlet.item<double>(), L.fbNeumann.item<double>(),
                        L.solutionLoss().item<double>(), L.freeBoundaryLoss().item<double>()};
        history_.push_back(rec);

        if (config_.verbose) {
            printJointStep(rec, elapsed());
        } else if (config_.logEvery > 0 && (epoch % config_.logEvery == 0 || epoch == E)) {
            std::cout << "Epoch [" << epoch << "/" << E << "] -> Loss FNN^sol: " << rec.solutionLoss
                      << " | Loss FNN^fb: " << rec.freeBoundaryLoss << " | lr: " << rec.solutionLearningRate
                      << " | " << std::fixed << std::setprecision(1) << elapsed() << " s"
                      << std::defaultfloat << std::setprecision(16) << std::endl;
        }
    }

    std::cout << "Addestramento completato in " << std::fixed << std::setprecision(1) << elapsed()
              << " s" << std::defaultfloat << std::endl;
}

} // namespace it::univr::neuralnetworks
