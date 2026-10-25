/*
 * PINN.cpp
 * Physics-Informed Neural Network for American options: collocation points, Black-Scholes residual,
 * loss terms, exercise boundary and final price.
 *
 * Two formulations share this class (switch PINNOptions::penaltyFormulation):
 *  - two networks (free boundary): FNNSol for the price and FNNfb for the exercise boundary B(t);
 *    the PDE holds only in the continuation region, and value matching / smooth pasting are imposed
 *    on B(t) as extra losses;
 *  - complementarity ("penalty" in the code): a single network FNNSol and the residual
 *    min(-Lambda P, P - payoff) on all the points; B(t) is recovered afterwards from the price.
 *
 * Author: Alberto Oliva Medin
 */
#include "PINN.h"
#include "it.univr.utils/SobolSampler.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace it::univr::neuralnetworks {

using torch::indexing::Slice;

/*
 * Checks that option, networks and market parameters have consistent dimensions and precomputes the
 * diffusion matrix sigma sigma^T used by the Black-Scholes operator.
 */
PINN::PINN(std::shared_ptr<IOption> option, BlackScholesParameters params, double Smax,
           std::shared_ptr<FNNSol> solutionNet, std::shared_ptr<FNNfb> boundaryNet,
           LossWeights weights, PINNOptions options)
    : option_(std::move(option)), params_(std::move(params)), Smax_(Smax),
      solutionNet_(std::move(solutionNet)), boundaryNet_(std::move(boundaryNet)),
      weights_(weights), options_(options) {
    if (!option_ || !solutionNet_ || !boundaryNet_)
        throw std::invalid_argument("Opzione e reti non possono essere nulle");
    d_ = option_->getNumberOfAssets();
    if (solutionNet_->getNumberOfAssets() != d_ || boundaryNet_->getNumberOfAssets() != d_)
        throw std::invalid_argument("Le reti e l'opzione devono avere lo stesso numero di sottostanti");
    if (static_cast<int>(params_.dividendYields.size()) != d_)
        throw std::invalid_argument("dividendYields deve avere d elementi");
    if (static_cast<int>(params_.volatilityMatrix.size()) != d_)
        throw std::invalid_argument("volatilityMatrix deve essere d x d");
    for (const auto& row : params_.volatilityMatrix)
        if (static_cast<int>(row.size()) != d_)
            throw std::invalid_argument("volatilityMatrix deve essere d x d");
    if (Smax_ <= option_->getStrike()) throw std::invalid_argument("Smax deve essere > K");

    // Diffusion coefficients (sigma sigma^T)_ij = sum_k sigma_ik sigma_jk
    diffusion_.assign(d_, std::vector<double>(d_, 0.0));
    for (int i = 0; i < d_; ++i)
        for (int j = 0; j < d_; ++j)
            for (int k = 0; k < d_; ++k)
                diffusion_[i][j] += params_.volatilityMatrix[i][k] * params_.volatilityMatrix[j][k];
}

// Mean squared error: every loss term is the mean of the squared residual over its points.
torch::Tensor PINN::mse(const torch::Tensor& x) {
    if (x.numel() == 0) return torch::zeros({});   // no points: zero contribution
    return x.pow(2).mean();
}

// Points in the unit hypercube [0, 1]^dimension: Sobol (low discrepancy, covers the space evenly) or uniform.
torch::Tensor PINN::samplePoints(int64_t n, int dimension, SamplingMethod method) const {
    if (method == SamplingMethod::Sobol) return utils::SobolSampler::generate(n, dimension);
    return torch::rand({n, dimension});
}

// n points S in [0, Smax]^d (and t in [0, T] if times != nullptr); a fraction can be
// concentrated near the strike (SamplingOptions)
torch::Tensor PINN::sampleSpot(int64_t n, const SamplingOptions& sampling, torch::Tensor* times) const {
    const double T = option_->getMaturity();
    const double K = option_->getStrike();
    const int64_t nRef = static_cast<int64_t>(std::llround(sampling.strikeRefinementFraction * n));
    const int64_t nUni = n - nRef;
    const int dim = d_ + (times ? 1 : 0);   // d prices, plus the time if requested

    // Uniform part: the unit hypercube is stretched to [0, Smax]^d (and [0, T] for the time)
    auto uni = samplePoints(nUni, dim, sampling.method);
    auto S = uni.index({Slice(), Slice(0, d_)}) * Smax_;
    torch::Tensor t = times ? uni.index({Slice(), Slice(d_, d_ + 1)}) * T : torch::Tensor();
    // Refined part: points in [K - w, K + w]^d, where the payoff has its kink and the price changes fastest
    if (nRef > 0) {
        const double lo = std::max(0.0, K - sampling.strikeRefinementHalfWidth);
        const double hi = std::min(Smax_, K + sampling.strikeRefinementHalfWidth);
        // separate sequence (dimension dim + 1) so as not to reuse the points of the uniform part
        auto ref = samplePoints(nRef, dim + 1, sampling.method);
        S = torch::cat({S, lo + (hi - lo) * ref.index({Slice(), Slice(0, d_)})}, 0);
        if (times) t = torch::cat({t, ref.index({Slice(), Slice(d_, d_ + 1)}) * T}, 0);
    }
    if (times) *times = t;
    return S;
}

/*
 * Builds all the collocation points of one training set. They are sampled once, before the training,
 * and kept fixed for all the epochs.
 */
CollocationSet PINN::sampleCollocationPoints(int nPde, int nInit, int nDir, int nFbDir,
                                             int nFbNeu, const SamplingOptions& sampling) const {
    const double T = option_->getMaturity();
    const SamplingMethod method = sampling.method;
    CollocationSet set;

    // PDE: pairs (S, t) in the whole domain; terminal condition: prices S only (t = T is set later)
    set.pdeS  = sampleSpot(nPde, sampling, &set.pdeT);
    set.initS = sampleSpot(nInit, sampling, nullptr);

    // Far boundary: random points moved onto the faces S_i = Smax by the option
    auto dir = samplePoints(nDir, d_ + 1, method);
    set.dirS = option_->projectOnFarBoundary(dir.index({Slice(), Slice(0, d_)}) * Smax_, Smax_);
    set.dirT = dir.index({Slice(), Slice(d_, d_ + 1)}) * T;

    // Boundary S_i = 0: one coordinate per row set to zero, faces assigned in blocks
    auto zero = samplePoints(nDir, d_ + 1, method);
    set.zeroS = zero.index({Slice(), Slice(0, d_)}) * Smax_;
    auto face = (torch::arange(nDir) * d_).div(nDir, "floor");
    set.zeroS.index_put_({torch::arange(nDir), face}, 0.0);
    set.zeroT = zero.index({Slice(), Slice(d_, d_ + 1)}) * T;

    // Free-boundary conditions: only times, the price is the boundary B(t) itself
    set.fbDirT = samplePoints(nFbDir, 1, method) * T;
    set.fbNeuT = samplePoints(nFbNeu, 1, method) * T;
    return set;
}

/*
 * The function P(S, t) used everywhere (losses, boundary, price).
 * With hardTerminalCondition, P = payoff + (T - t) z: at t = T it equals the payoff exactly.
 */
torch::Tensor PINN::evaluate(const torch::Tensor& S, const torch::Tensor& t) {
    auto z = solutionNet_->forward(S, t);
    if (!options_.hardTerminalCondition) return z;
    return option_->payoff(S) + (option_->getMaturity() - t) * z;
}

/*
 * Lambda P = P_t + 1/2 sum_ij (sigma sigma^T)_ij S_i S_j P_ij + sum_i (r - q_i) S_i P_i - r P
 *
 * Derivatives with autograd:
 *  1. S and t become leaves of the graph (requires_grad), so that P can be differentiated with respect to them;
 *  2. one grad call gives P_S (all the first derivatives in S) and P_t;
 *  3. for every i, a grad call on P_i = dP/dS_i gives the row of second derivatives P_ij.
 * In grad(outputs, inputs, grad_outputs, retain_graph, create_graph):
 *  - ones_like(P) gives the derivative of every point separately (one row per collocation point);
 *  - retain_graph = true keeps the graph, which is needed again for the second derivatives;
 *  - create_graph = true builds a graph for the derivatives themselves, so that they can be
 *    differentiated again: first with respect to S (second derivatives), then with respect to the weights.
 */
torch::Tensor PINN::pdeResidual(const torch::Tensor& Sin, const torch::Tensor& tin) {
    const double r = params_.riskFreeRate;
    auto S = Sin.detach().requires_grad_(true);
    auto t = tin.detach().requires_grad_(true);
    auto P = evaluate(S, t);

    // First derivatives: dP/dS_i for every asset and dP/dt, in a single call
    auto first = torch::autograd::grad({P}, {S, t}, {torch::ones_like(P)}, true, true);
    auto P_S = first[0];   // [N, d]
    auto P_t = first[1];   // [N, 1]

    auto residual = P_t - r * P;
    for (int i = 0; i < d_; ++i) {
        auto Si = S.index({Slice(), Slice(i, i + 1)});
        auto P_i = P_S.index({Slice(), Slice(i, i + 1)});
        residual = residual + (r - params_.dividendYields[i]) * Si * P_i;   // drift term

        // Second derivatives: differentiating P_i again gives d2P/dS_i dS_j for every j
        auto P_iS = torch::autograd::grad({P_i}, {S}, {torch::ones_like(P_i)}, true, true)[0];
        for (int j = 0; j < d_; ++j) {
            if (diffusion_[i][j] == 0.0) continue;   // uncorrelated pair: no mixed term
            auto Sj = S.index({Slice(), Slice(j, j + 1)});
            auto P_ij = P_iS.index({Slice(), Slice(j, j + 1)});
            residual = residual + 0.5 * diffusion_[i][j] * Si * Sj * P_ij;   // diffusion term
        }
    }
    return residual;
}

/*
 * Interior loss.
 * Free-boundary formulation: the PDE holds only in the continuation region estimated by FNNfb,
 * so at every step only the sampled points in that region are kept.
 * Penalty formulation: residual min(-Lambda P, P - payoff) on all the points.
 *
 * Why the min: -Lambda P >= 0 and P - payoff >= 0 must both hold, and at least one of them must be zero
 * (continuation: Lambda P = 0; exercise: P = payoff). The minimum of two non-negative numbers is zero
 * exactly when one of them is zero, and it is non-zero (positive or negative) when a condition is violated.
 */
torch::Tensor PINN::lossPDE(const torch::Tensor& S, const torch::Tensor& t) {
    if (options_.penaltyFormulation) {
        auto residual = pdeResidual(S, t);
        auto gap = evaluate(S, t) - option_->payoff(S);   // time value P - payoff
        return mse(torch::minimum(-residual, gap));
    }
    // Two networks: the mask is a selection, not a function to differentiate
    torch::Tensor mask;
    {
        torch::NoGradGuard noGrad;
        mask = option_->isInContinuationRegion(S, boundaryAt(t)).squeeze(1);
    }
    auto Sc = S.index({mask});
    auto tc = t.index({mask});
    if (Sc.size(0) == 0) return torch::zeros({});
    return mse(pdeResidual(Sc, tc));
}

// Terminal condition: P(S, T) = payoff(S)
torch::Tensor PINN::lossInitial(const torch::Tensor& S) {
    auto T = torch::full({S.size(0), 1}, option_->getMaturity());   // every point at t = T
    return mse(evaluate(S, T) - option_->payoff(S));
}

// Far-boundary condition (value given by the option)
torch::Tensor PINN::lossDirichlet(const torch::Tensor& S, const torch::Tensor& t) {
    auto target = option_->farBoundaryValue(S, t, params_.riskFreeRate);
    return mse(evaluate(S, t) - target);
}

// Boundary S_i = 0: immediate exercise is optimal, so the price equals the payoff (K for the put)
torch::Tensor PINN::lossZeroBoundary(const torch::Tensor& S, const torch::Tensor& t) {
    return mse(evaluate(S, t) - option_->payoff(S));
}

// Free boundary at maturity: B(T) = (K, ..., K)
torch::Tensor PINN::lossFreeBoundaryInitial() {
    auto B = boundaryAt(torch::full({1, 1}, option_->getMaturity()));
    return mse(B - option_->freeBoundaryAtMaturity());
}

/*
 * Dirichlet condition on the free boundary (value matching): P(B(t), t) = exercise value.
 * The points are (B(t), t): the price network is evaluated on the boundary itself.
 * Fixed boundary: detach() blocks the gradient towards FNNfb, so only FNNSol learns.
 * Trained boundary: the gradient flows through B(t) and also moves the weights of FNNfb.
 */
torch::Tensor PINN::lossFreeBoundaryDirichlet(const torch::Tensor& t, bool trainBoundary) {
    auto B = boundaryAt(t);
    if (!trainBoundary) B = B.detach();
    return mse(evaluate(B, t) - option_->valueAtFreeBoundary(B));
}

// Neumann condition on the free boundary: dP/dS_i(B(t), t) = exercise delta (put: -1 on the minimum asset)
torch::Tensor PINN::lossFreeBoundaryNeumann(const torch::Tensor& t, bool trainBoundary) {
    auto B = boundaryAt(t);
    // Fixed boundary: a leaf tensor is needed to differentiate with respect to.
    // Trained boundary: differentiate directly with respect to the output of FNNfb,
    // so that the gradient of the Neumann condition also reaches the weights of FNNfb.
    auto Sb = trainBoundary ? B : B.detach().requires_grad_(true);
    auto P = evaluate(Sb, t);
    // The only derivative of the free-boundary losses: dP/dS at S = B(t)
    auto P_S = torch::autograd::grad({P}, {Sb}, {torch::ones_like(P)}, true, true)[0];
    return mse(P_S - option_->deltaAtFreeBoundary(Sb));
}

/*
 * Evaluates every loss term on the collocation set.
 * Two networks: all six terms take part in the training (L_sol for FNNSol, L_fb for FNNfb).
 * Complementarity: only PDE, Initial and Dirichlet are trained; the free-boundary terms are
 * computed with detach() just to be plotted.
 */
PINNLosses PINN::computeLosses(const CollocationSet& set, bool trainBoundary) {
    PINNLosses L;
    L.pde       = lossPDE(set.pdeS, set.pdeT);
    L.initial   = lossInitial(set.initS);
    L.dirichlet = lossDirichlet(set.dirS, set.dirT);
    L.freeBoundaryInLoss = !options_.penaltyFormulation;
    if (options_.penaltyFormulation) {
        // Boundary S_i = 0 added to the Dirichlet term (far boundary + near boundary)
        L.dirichlet = L.dirichlet + lossZeroBoundary(set.zeroS, set.zeroT);
        // The free-boundary conditions are already inside min(...): here they are computed only
        // to monitor them, on the boundary obtained from the price, without gradient
        L.fbInitial   = lossFreeBoundaryInitial().detach();
        L.fbDirichlet = lossFreeBoundaryDirichlet(set.fbDirT, false).detach();
        L.fbNeumann   = lossFreeBoundaryNeumann(set.fbNeuT, false).detach();
    } else {
        L.fbInitial   = lossFreeBoundaryInitial();
        L.fbDirichlet = lossFreeBoundaryDirichlet(set.fbDirT, trainBoundary);
        L.fbNeumann   = lossFreeBoundaryNeumann(set.fbNeuT, trainBoundary);
    }
    L.w = weights_;
    return L;
}

/*
 * Exercise boundary B(t).
 * Complementarity: recovered from the price (boundaryFromSolution).
 * Two networks: output of FNNfb, optionally constrained to B(t) = K exp(-(T - t)/T softplus(z)),
 * which is always in (0, K] and equals K at maturity.
 */
torch::Tensor PINN::boundaryAt(const torch::Tensor& t) {
    if (options_.penaltyFormulation) return boundaryFromSolution(t);
    auto z = boundaryNet_->forward(t);
    if (!options_.constrainedFreeBoundary) return z;
    const double T = option_->getMaturity();
    auto BT = option_->freeBoundaryAtMaturity();                      // [1, d]
    return BT * torch::exp(-(T - t) / T * torch::nn::functional::softplus(z));
}

/*
 * Free boundary obtained from the solution (penalty formulation): for every asset i, with the other
 * underlyings at Smax, B_i(t) is the largest S_i in [0, K] where exercise occurs, i.e. where
 * P - payoff <= exerciseTolerance with payoff > 0 (0 if exercise never occurs).
 *
 * In formula: B(t) = max{ S in [0, K] : P(S, t) - payoff(S) <= epsilon }, computed on a grid of G + 1
 * prices for all the requested times at once. The other underlyings are set to Smax so that the asset
 * being scanned is the minimum.
 */
torch::Tensor PINN::boundaryFromSolution(const torch::Tensor& t) {
    torch::NoGradGuard noGrad;
    const int64_t G = 400;
    const int64_t N = t.size(0);
    const double K = option_->getStrike();
    auto x = torch::linspace(0.0, K, G + 1);                           // [G+1]
    auto out = torch::empty({N, d_});
    for (int i = 0; i < d_; ++i) {
        // N blocks of G + 1 rows: block n scans S_i over the grid at time t_n
        auto S = torch::full({N * (G + 1), d_}, Smax_);
        S.index_put_({Slice(), i}, x.repeat({N}));
        auto tt = t.repeat_interleave(G + 1, 0);
        auto payoff = option_->payoff(S);
        // Exercise test on every grid point: price equal to the payoff (within the tolerance), payoff > 0
        auto exercised = ((evaluate(S, tt) - payoff <= options_.exerciseTolerance) & (payoff > 0.0))
                             .view({N, G + 1});
        // Keep the grid value where exercised, 0 elsewhere; the row maximum is the boundary
        auto B = torch::where(exercised, x.unsqueeze(0).expand({N, G + 1}), torch::zeros({N, G + 1}));
        out.index_put_({Slice(), i}, std::get<0>(B.max(1)));
    }
    return out;
}

// Penalty formulation: exercise where the price equals the payoff (within exerciseTolerance) and the
// payoff is positive; with a zero payoff there is never exercise, even if the price is about zero.
torch::Tensor PINN::isInContinuationRegion(const torch::Tensor& S, const torch::Tensor& t) {
    torch::NoGradGuard noGrad;
    if (options_.penaltyFormulation) {
        auto payoff = option_->payoff(S);
        auto exercised = (evaluate(S, t) - payoff <= options_.exerciseTolerance) & (payoff > 0.0);
        return exercised.logical_not();
    }
    return option_->isInContinuationRegion(S, boundaryAt(t));
}

/*
 * Price of the American option after the training: one forward pass of the network.
 * Complementarity: max(P, payoff), because the option is never worth less than immediate exercise.
 * Two networks: the network where FNNfb says "continue", the payoff where it says "exercise".
 */
torch::Tensor PINN::price(const torch::Tensor& S, const torch::Tensor& t) {
    torch::NoGradGuard noGrad;
    auto P = evaluate(S, t);
    auto payoff = option_->payoff(S);
    auto price = options_.penaltyFormulation ? torch::maximum(P, payoff)
                                             : torch::where(isInContinuationRegion(S, t), P, payoff);
    if (options_.enforceIntrinsicValue) price = torch::maximum(price, payoff);
    return price;
}

// Exercise boundary for output (tables, CSV files), without gradient.
torch::Tensor PINN::freeBoundary(const torch::Tensor& t) {
    torch::NoGradGuard noGrad;
    return boundaryAt(t);
}

} // namespace it::univr::neuralnetworks
