package it.univr.model;

/**
 * Interface representing a generic financial stochastic model used to generate
 * asset price paths.
 * 
 * @author Alberto Oliva Medin
 */
public interface ModelToUse {

	/**
	 * Generates and returns the matrix of simulated asset price paths.
	 * 
	 * @return A matrix of dimensions [numberOfPaths][numberOfTimeSteps + 1]
	 */
	public double[][][] getAssetPaths();

	/**
	 * Returns the risk-free interest rate configured in the model.
	 * 
	 * @return The risk-free rate (r)
	 */
	public double[] getRiskFreeRate();

	/**
	 * Returns the time step size (delta t) used in the path discretization.
	 * 
	 * @return The value of dt
	 */
	public double getDt();

	/**
	 * Returns the number of time steps used for the discretization.
	 */
	public int getNumberOfTimes();
}