#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>
#include "it.univr.neuralNetworks/IOption.h"
#include "it.univr.neuralNetworks/FNNSol.h"
#include "it.univr.neuralNetworks/FNNfb.h"

namespace it::univr::neuralnetworks {

/**
 * Parameters of the multi-asset <b>Black-Scholes model</b>.
 * <p>
 * The diffusion matrix entering the PDE is sigma sigma^T, i.e. sum_k sigma_ik sigma_jk.
 * For d = 1, volatilityMatrix = {{sigma}}.
 * </p>
 */
struct BlackScholesParameters {
    double riskFreeRate;                                  // r
    std::vector<double> dividendYields;                   // q_i, one per underlying
    std::vector<std::vector<double>> volatilityMatrix;    // sigma (d x d)
};

/** Method used to sample the collocation points. */
enum class SamplingMethod { Sobol, Uniform };

/**
 * Sampling of the collocation points.
 * <p>
 * With strikeRefinementFraction = 0 the points are uniform in [0, Smax]^d. With a positive value,
 * that fraction of the PDE and terminal points is drawn in [K - w, K + w]^d
 * (w = strikeRefinementHalfWidth), where the payoff has its kink.
 * </p>
 */
struct SamplingOptions {
    SamplingMethod method;              // Sobol (low-discrepancy) or uniform random points
    double strikeRefinementFraction;    // fraction of the points concentrated near K (0 = none)
    double strikeRefinementHalfWidth;   // half width w of the refinement box [K - w, K + w]^d
};

/**
 * Variants of the method. All false gives the base formulation with the free-boundary network.
 */
struct PINNOptions {
    /** P(S, t) = payoff(S) + (T - t) FNNSol(S, t): the terminal condition holds by construction. */
    bool hardTerminalCondition;

    /** Reported price = max(price, payoff): an American option is worth at least its payoff. */
    bool enforceIntrinsicValue;

    /** B_i(t) = K exp(-(T - t)/T softplus(FNNfb(t))): guarantees 0 < B_i(t) <= K and B(T) = K. */
    bool constrainedFreeBoundary;

    /**
     * Complementarity formulation (called "penalty" in the code and in the interface): the American
     * price solves the linear complementarity problem
     * Lambda P <= 0, P - payoff >= 0, Lambda P (P - payoff) = 0,
     * written as the single equation min(-Lambda P, P - payoff) = 0. The interior loss is the mean of
     * min(-Lambda P, P - payoff)^2 on all the points and FNNfb is not trained: the exercise region is
     * where P - payoff <= exerciseTolerance, and the boundary is recovered from the price.
     */
    bool penaltyFormulation;
    double exerciseTolerance;   // epsilon of the exercise test P - payoff <= epsilon (e.g. 1e-3)
};

/** Collocation points of one training set. Shapes: S [N, d], t [N, 1]. */
struct CollocationSet {
    torch::Tensor pdeS, pdeT;   // interior of [0, Smax]^d x [0, T]
    torch::Tensor initS;        // terminal condition at t = T
    torch::Tensor dirS, dirT;   // far boundary
    torch::Tensor zeroS, zeroT; // boundary S_i = 0 (penalty formulation only)
    torch::Tensor fbDirT;       // Dirichlet condition on the free boundary
    torch::Tensor fbNeuT;       // Neumann condition on the free boundary
};

/** Weights of the loss terms (all equal to 1 in the experiments of the thesis). */
struct LossWeights {
    double pde;
    double initial;
    double dirichlet;
    double fbInitial;
    double fbDirichlet;
    double fbNeumann;
};

/**
 * Values of the individual loss terms (mean squared errors) and their combinations.
 * <p>
 * The two networks have two <b>separate</b> losses: solutionLoss() is minimised only with respect to
 * the weights of FNNSol, freeBoundaryLoss() only with respect to the weights of FNNfb. They are never
 * summed. fbDirichlet and fbNeumann appear in both, and are what couples the two networks.
 * </p>
 */
struct PINNLosses {
    torch::Tensor pde, initial, dirichlet;            // depend only on FNNSol
    torch::Tensor fbInitial, fbDirichlet, fbNeumann;  // involve the free boundary
    LossWeights w;
    /** false with the penalty formulation: the free-boundary terms are only monitored. */
    bool freeBoundaryInLoss;

    /**
     * @return The loss minimised by the price network FNNSol:
     *         L_sol = PDE + Initial + Dirichlet (+ FB_Dir + FB_Neu with two networks).
     */
    torch::Tensor solutionLoss() const {
        auto loss = w.pde * pde + w.initial * initial + w.dirichlet * dirichlet;
        if (freeBoundaryInLoss) loss = loss + w.fbDirichlet * fbDirichlet + w.fbNeumann * fbNeumann;
        return loss;
    }

    /** @return The loss minimised by the free-boundary network FNNfb: L_fb = FB_Init + FB_Dir + FB_Neu. */
    torch::Tensor freeBoundaryLoss() const {
        return w.fbInitial * fbInitial + w.fbDirichlet * fbDirichlet + w.fbNeumann * fbNeumann;
    }
};

/**
 * <b>Physics-Informed Neural Network</b> for the American option free-boundary problem.
 * <p>
 * The class is general in the number of underlyings: d = 1 is the single-asset case,
 * d > 1 the multi-asset case. It combines:
 * <ul>
 * <li>an option (IOption), which provides payoff and boundary conditions;</li>
 * <li>the price network FNNSol, P(S, t);</li>
 * <li>the free-boundary network FNNfb, B(t) (not trained with the penalty formulation).</li>
 * </ul>
 * The losses are mean squared errors of the PDE residual and of the terminal, boundary and
 * free-boundary conditions, evaluated on the collocation points. All derivatives are computed
 * with autograd.
 * </p>
 * <p>
 * The class defines the problem (residual, losses, price); the optimisation itself is done by
 * TrainNeuralNetwork.
 * </p>
 *
 * @author Alberto Oliva Medin
 * @version 1.0
 */
class PINN {
public:
    /**
     * @param option      The American option to price.
     * @param params      Black-Scholes market parameters.
     * @param Smax        Upper bound of the spatial domain [0, Smax]^d (must be > K).
     * @param solutionNet Price network.
     * @param boundaryNet Free-boundary network.
     * @param weights     Weights of the loss terms.
     * @param options     Variants of the method.
     */
    PINN(std::shared_ptr<IOption> option,
         BlackScholesParameters params,
         double Smax,
         std::shared_ptr<FNNSol> solutionNet,
         std::shared_ptr<FNNfb> boundaryNet,
         LossWeights weights,
         PINNOptions options);

    /* ===========================================================================
     * COLLOCATION POINTS AND EVALUATION
     * ===========================================================================
     */

    /**
     * Samples a new set of collocation points.
     *
     * @param nPde     Interior points.
     * @param nInit    Points at maturity.
     * @param nDir     Points on the far boundary (and on S_i = 0).
     * @param nFbDir   Times for the Dirichlet condition on the free boundary.
     * @param nFbNeu   Times for the Neumann condition on the free boundary.
     * @param sampling Sampling method and refinement near the strike.
     * @return The collocation set.
     */
    CollocationSet sampleCollocationPoints(int nPde, int nInit, int nDir, int nFbDir, int nFbNeu,
                                           const SamplingOptions& sampling) const;

    /**
     * Value of the solution P(S, t) on which all the losses are computed: the output of FNNSol,
     * or payoff + (T - t) FNNSol when hardTerminalCondition is active.
     */
    torch::Tensor evaluate(const torch::Tensor& S, const torch::Tensor& t);

    /**
     * Black-Scholes operator applied to P:
     * Lambda P = P_t + 1/2 sum_ij (sigma sigma^T)_ij S_i S_j P_ij + sum_i (r - q_i) S_i P_i - r P.
     * The first and second derivatives are computed with torch::autograd::grad (create_graph = true,
     * so that the residual can be differentiated again with respect to the weights).
     */
    torch::Tensor pdeResidual(const torch::Tensor& S, const torch::Tensor& t);

    /* ===========================================================================
     * LOSS TERMS
     * trainBoundary = false: B(t) is a constant (steps of FNNSol only);
     * trainBoundary = true : the gradients also reach the weights of FNNfb (joint step).
     * ===========================================================================
     */

    /** Interior loss: mean of min(-Lambda P, P - payoff)^2 (penalty) or of (Lambda P)^2 in the continuation region. */
    torch::Tensor lossPDE(const torch::Tensor& S, const torch::Tensor& t);
    /** Terminal condition: P(S, T) = payoff(S). */
    torch::Tensor lossInitial(const torch::Tensor& S);
    /** Far boundary: P = value given by the option (0 for the put with d = 1). */
    torch::Tensor lossDirichlet(const torch::Tensor& S, const torch::Tensor& t);
    /** Boundary S_i = 0: P = payoff. Used only by the penalty formulation, which has no free boundary as anchor. */
    torch::Tensor lossZeroBoundary(const torch::Tensor& S, const torch::Tensor& t);
    /** Free boundary at maturity: B(T) = K. */
    torch::Tensor lossFreeBoundaryInitial();
    /** Value matching on the free boundary: P(B(t), t) = payoff(B(t)). */
    torch::Tensor lossFreeBoundaryDirichlet(const torch::Tensor& t, bool trainBoundary);
    /** Smooth pasting on the free boundary: dP/dS(B(t), t) = -1 (on the minimum asset). */
    torch::Tensor lossFreeBoundaryNeumann(const torch::Tensor& t, bool trainBoundary);

    /** @return All the loss terms on the given collocation set. */
    PINNLosses computeLosses(const CollocationSet& set, bool trainBoundary);

    /* ===========================================================================
     * PRICING
     * ===========================================================================
     */

    /** @return The American price: network in the continuation region, payoff in the exercise region. */
    torch::Tensor price(const torch::Tensor& S, const torch::Tensor& t);

    /** @return The free boundary B(t), shape [N, d]. */
    torch::Tensor freeBoundary(const torch::Tensor& t);

    /** @return true where the PINN estimates the continuation region, shape [N, 1]. */
    torch::Tensor isInContinuationRegion(const torch::Tensor& S, const torch::Tensor& t);

    /** @return false with the penalty formulation, where FNNfb is not trained. */
    bool usesFreeBoundaryNetwork() const { return !options_.penaltyFormulation; }

    std::shared_ptr<FNNSol> getSolutionNet() const { return solutionNet_; }
    std::shared_ptr<FNNfb> getBoundaryNet() const { return boundaryNet_; }
    std::shared_ptr<IOption> getOption() const { return option_; }
    const BlackScholesParameters& getParameters() const { return params_; }
    double getSmax() const { return Smax_; }
    int getNumberOfAssets() const { return d_; }

private:
    /** Mean squared error of a residual tensor (0 if the tensor is empty). */
    static torch::Tensor mse(const torch::Tensor& x);
    /** B(t): from FNNfb (two networks) or recovered from the price (penalty formulation). */
    torch::Tensor boundaryAt(const torch::Tensor& t);
    /** B(t) = max{ S in [0, K] : P(S, t) - payoff(S) <= exerciseTolerance }, evaluated on a grid. */
    torch::Tensor boundaryFromSolution(const torch::Tensor& t);
    /** n points in [0, 1]^dimension, Sobol or uniform. */
    torch::Tensor samplePoints(int64_t n, int dimension, SamplingMethod method) const;
    /** n underlying prices in [0, Smax]^d (and times in [0, T] if times != nullptr). */
    torch::Tensor sampleSpot(int64_t n, const SamplingOptions& sampling, torch::Tensor* times) const;

    std::shared_ptr<IOption> option_;
    BlackScholesParameters params_;
    double Smax_;
    std::shared_ptr<FNNSol> solutionNet_;
    std::shared_ptr<FNNfb> boundaryNet_;
    LossWeights weights_;
    PINNOptions options_;
    int d_;                                        // number of underlyings
    std::vector<std::vector<double>> diffusion_;   // sum_k sigma_ik sigma_jk
};

} // namespace it::univr::neuralnetworks
