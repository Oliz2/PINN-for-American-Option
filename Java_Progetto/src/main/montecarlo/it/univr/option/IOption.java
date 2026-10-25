package it.univr.option;

/**
 * Interface representing a generic financial derivative option. Defines the
 * contract maturity and the payoff function.
 * 
 */
public interface IOption {

	/**
	 * Calculates the immediate exercise payoff given a specific asset price. For
	 * example, for a Put option it returns max(Strike - S, 0).
	 * 
	 * @param assetPrice The current price of the underlying asset
	 * @return The immediate execution payoff value
	 */

	/**
	 * Returns the maturity (time to expiration) of the option in years.
	 * 
	 * @return The option maturity (T)
	 */
	public double getMaturity();

	public double getPayoff(double[][] path);

	public double getPayoff(double[] path);

}