package it.univr.option;

import java.util.function.DoubleUnaryOperator;
import it.univr.model.MarketEnvironmentForModel;

/**
 * Represents a Bermudan (or simplified American-style) option contract,
 * evaluating path-dependent early exercise decisions by comparing immediate
 * intrinsic values against estimated local continuation values along simulated
 * trajectories.
 *
 * Fix 1: getPayoff(double[][] path) (used ONLY by the naive pricer MonteCarloPricer;
 * LongstaffSchwartzPricer never calls this overload) estimated the continuation value by averaging
 * the FUTURE payoffs realised on the same path. This is look-ahead bias (it uses information not
 * available at time i) and it inflated the price by about 25% (checked against LSM and an
 * independent CRR binomial tree). It has been replaced by the classic naive rule "exercise as soon
 * as the option is in the money", which uses only the current state of the path. It is a
 * suboptimal policy (hence "naive"), but a valid non-anticipating stopping rule, so by construction
 * it gives a legitimate lower bound of the American price.
 *
 * Fix 2: the loop below used to start at i = 0. But i = 0 is the valuation time (t = 0): path[0][0]
 * is ALWAYS S0, the same deterministic value on ALL paths (nothing has been simulated yet), so it
 * is not a real exercise opportunity. Checking it as a decision time made the option "exercise at
 * t = 0" whenever S0 was in the money (S0 < K for a put), collapsing the price to the deterministic
 * value (K - S0), identical for ANY volatility, which is exactly the observed symptom (NaiveMC
 * prices not changing with sigma). The loop now starts at the first real exercise date
 * (i = timetoExcersize), not at t = 0.
 */
public class BermudaOption implements IOption {

	// Option maturity in years
	private final double maturity;
	// Functional interface defining the option payoff structure (e.g., put or call)
	private final DoubleUnaryOperator payoff;
	// Frequency or interval step count for early exercise opportunities
	private final int timetoExcersize;

	/**
	 * Constructs a Bermudan option with specified maturity, payoff, and exercise
	 * frequency.
	 *
	 * @param maturity        Option maturity in years.
	 * @param payoff          Functional operator representing the option payoff.
	 * @param timetoExcersize Interval steps for discrete early exercise points.
	 */
	public BermudaOption(double maturity, DoubleUnaryOperator payoff, int timetoExcersize) {
		this.maturity = maturity;
		this.payoff = payoff;
		this.timetoExcersize = timetoExcersize;
	}

	/**
	 * Evaluates the option payoff over a simulated multi-step path matrix
	 * (Bermudan style), with the naive non-anticipating rule "exercise as soon as in the money"
	 * (see the notes at the top of the class).
	 *
	 * @param path The 2D matrix representing a single simulation path:
	 *             [timeSteps][assets].
	 * @return The discounted cash flow resulting from early exercise or
	 *         terminal maturity.
	 */
	@Override
	public double getPayoff(double[][] path) {

		int numberOfSteps = path.length - 1;
		// Retrieves the weighted risk-free rate from the global market environment
		double riskFreeRateMedium = MarketEnvironmentForModel.getWeightedRate();

		// Start at timetoExcersize, not at 0: i = 0 is t = 0 (deterministic S0, the same on all
		// paths), not a real exercise date. See Fix 2 at the top of the class.
		for (int i = timetoExcersize; i < path.length; i += timetoExcersize) {
			double timeYears = ((double) i / numberOfSteps) * maturity;
			// Immediate intrinsic exercise value at the current time step for the primary
			// asset (path[i][0]): no information about the future is used here.
			double valueAtTime = payoff.applyAsDouble(path[i][0]);

			// Naive non-anticipating rule: exercise as soon as the immediate payoff is positive
			// (in the money), instead of comparing it with a continuation value computed from
			// the realised future.
			if (valueAtTime > 0.0) {
				// Return immediate exercise payoff discounted from the exercise time back to t
				// = 0
				return valueAtTime * Math.exp(-riskFreeRateMedium * timeYears);
			}
		}

		// Fallback: if never in-the-money at any exercise date, return terminal
		// payoff at maturity discounted to t = 0
		return payoff.applyAsDouble(path[numberOfSteps][0]) * Math.exp(-riskFreeRateMedium * maturity);
	}

	/**
	 * Returns the option maturity in years.
	 *
	 * @return The maturity time horizon.
	 */
	@Override
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Evaluates the payoff for a single flat path array (European-style fallback).
	 *
	 * @param path Array of asset prices along a simulation path.
	 * @return The option payoff evaluated at the final path element.
	 */
	@Override
	public double getPayoff(double[] path) {
		// Returns the payoff evaluated strictly at the terminal element of the path
		// array
		return payoff.applyAsDouble(path[path.length - 1]);
	}
}