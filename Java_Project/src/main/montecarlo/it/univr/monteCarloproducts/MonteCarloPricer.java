package it.univr.monteCarloproducts;

import it.univr.model.*;
import it.univr.option.IOption;

/**
 * Class responsible for pricing standard European-style options using direct
 * forward Monte Carlo simulation by averaging discounted terminal payoffs
 * across all simulated paths.
 *
 * Fix: getValue(...) used to discount the result TWICE. The payoff returned by
 * option.getPayoff(double[][] path), as implemented in BermudaOption, is ALREADY discounted to
 * t = 0 at the actual exercise time (line "valueAtTime * Math.exp(-riskFreeRateMedium * timeYears)"
 * in BermudaOption.getPayoff). This method took that already discounted value and multiplied it
 * again by Math.exp(-riskFreeMedium * maturity), discounting it a second time to a time (maturity)
 * that is often different from the actual exercise time. The bug was already there with the old
 * "look-ahead" rule, where it partly hid that bias. Checked numerically: an in-the-money option
 * exercised "at t = 0" returned (K - S0) from getPayoff (no discount, since timeYears was 0, see
 * the fix in BermudaOption), but the printed price was always lower by exactly exp(-r * maturity)
 * (e.g. T = 1: 4 * exp(-0.06 * 1) = 3.767; T = 2: 4 * exp(-0.06 * 2) = 3.548, both exactly the
 * observed values). The second discount has been removed: the discounted value from getPayoff is
 * averaged over the simulations as it is.
 */
public class MonteCarloPricer {

	// Cached fair price value of the option
	private double priceValue;

	/**
	 * Default constructor for the Monte Carlo pricer.
	 */
	public MonteCarloPricer() {
		// Default initialization
	}

	/**
	 * Computes the fair price of a standard option using direct forward simulation
	 * of asset paths. Each path's payoff, as returned by option.getPayoff(...),
	 * is already discounted back to t = 0 at its own (possibly early) exercise
	 * time: this method simply averages those already-discounted payoffs across
	 * simulations — it must NOT discount again.
	 *
	 * @param option The option contract defining the payoff structure.
	 * @param model  The stochastic model providing the simulated asset paths.
	 * @return The expected standard Monte Carlo option price.
	 */
	public double getValue(IOption option, ModelToUse model) {

		// Retrieve the 3D asset paths matrix from the stochastic model
		double[][][] path = model.getAssetPaths();

		int numberOfSimulation = path.length;

		double totalPayoff = 0.0;

		// Evaluate the (already time-0-discounted) payoff for each independent
		// simulation trajectory
		for (int i = 0; i < numberOfSimulation; i++) {
			double finalAssetPrice = option.getPayoff(path[i]);
			totalPayoff += finalAssetPrice;
		}

		// No second discount here: option.getPayoff(...) has already discounted each payoff to
		// t = 0 at its own exercise time. Only the average over the simulations is taken
		// (Monte Carlo estimate of the discounted expected value).
		this.priceValue = totalPayoff / numberOfSimulation;

		return priceValue;
	}

	/**
	 * Retrieves the cached option price value.
	 *
	 * @return The computed option price.
	 */
	public double getPriceValue() {
		return priceValue;
	}
}