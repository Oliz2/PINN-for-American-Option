package it.univr.model;

import it.univr.utils.FinanceUtils;

/**
 * Represents a global market environment utility that manages aggregated
 * financial parameters, such as the weighted average risk-free rate, shared
 * across multi-asset pricing and simulation models.
 * 
 */
public final class MarketEnvironmentForModel {

	// Cached weighted average risk-free rate
	private static double weightedRate;
	// Initialization flag tracking whether the market environment has been set up
	private static boolean initialized = false;

	/**
	 * Private constructor to prevent instantiation of this utility class.
	 */
	private MarketEnvironmentForModel() {
	}

	/**
	 * Initializes the market environment by computing the weighted average of the
	 * provided risk-free rates based on the specified asset weights.
	 * 
	 * @param rates   Array of risk-free interest rates.
	 * @param weights Array of asset weights corresponding to each rate.
	 */
	public static void getValue(double[] rates, double[] weights) {
		weightedRate = FinanceUtils.calculatesPoderateWeight(rates, weights);
		initialized = true;
	}

	/**
	 * Retrieves the pre-calculated weighted average risk-free rate.
	 * 
	 * @return The weighted average risk-free rate.
	 * @throws IllegalStateException if the market environment has not been
	 *                               initialized yet.
	 */
	public static double getWeightedRate() {
		if (!initialized) {
			throw new IllegalStateException("Market environment has not been initialized!");
		}
		return weightedRate;
	}
}