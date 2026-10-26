package it.univr.utils;

/**
 * Utility class providing fundamental financial calculation methods, including
 * weighted rate aggregations and path-dependent expected value estimations.
 * 
 */
public class FinanceUtils {

	/**
	 * Private constructor to prevent instantiation of this utility class.
	 */
	private FinanceUtils() {
	}

	/**
	 * Computes the weighted average (weighted mean) of a set of interest rates
	 * based on corresponding weight coefficients.
	 * 
	 * @param rates  Array of interest rates.
	 * @param weight Array of corresponding weights.
	 * @return The weighted average rate.
	 * @throws IllegalArgumentException if the rates and weights arrays have
	 *                                  different sizes.
	 */
	public static double calculatesPoderateWeight(double[] rates, double[] weight) {

		if (rates.length != weight.length) {
			throw new IllegalArgumentException("Rates and weights must have the same size");
		} else {

			double sum = 0;
			double totalWeight = 0;
			for (int i = 0; i < rates.length; i++) {
				sum += rates[i] * weight[i];
				totalWeight += weight[i];
			}
			return sum / totalWeight;

		}
	}

}