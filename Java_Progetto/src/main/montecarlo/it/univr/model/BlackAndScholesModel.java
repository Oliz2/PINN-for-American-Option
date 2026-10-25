package it.univr.model;

import it.univr.monteCarloproducts.MonteCarloGenerator;

/**
 * 
 * Implements the multi-asset Black-Scholes stochastic model to simulate asset
 * price trajectories, with correlated Brownian motions derived via Cholesky
 * decomposition.
 * 
 * 
 */
public class BlackAndScholesModel implements ModelToUse {

	// Initial asset spot prices
	private final double[] initialValue;
	// Risk-free interest rates for each asset
	private final double[] riskFreeRate;
	// Volatility parameters for each asset
	private final double[] sigma;
	// Total number of simulated Monte Carlo paths
	private final int numberOfSimulation;
	// Number of discrete time steps
	private int numberOfTimes;
	// Time increment size (dt)
	private final double dt;

	// 3D storage matrix for simulated asset paths: [simulation][time + 1][asset]
	private double[][][] assetPaths;
	// Number of underlying assets in the model
	private int numberOfAsset;
	// Correlation matrix between assets
	private double[][] rho;

	/**
	 * Constructs a multi-asset Black-Scholes simulation model.
	 * 
	 * @param initialValue       Array of starting spot prices.
	 * @param riskFreeRate       Array of risk-free interest rates.
	 * @param maturity           Total time horizon in years.
	 * @param numberOfSimulation Total number of simulated paths.
	 * @param sigma              Array of volatilities.
	 * @param numberOfTimes      Number of discrete time intervals.
	 * @param rho                Correlation matrix of the assets.
	 */
	public BlackAndScholesModel(double[] initialValue, double[] riskFreeRate, double maturity, int numberOfSimulation,
			double[] sigma, int numberOfTimes, double[][] rho) {

		this.initialValue = initialValue;
		this.riskFreeRate = riskFreeRate;
		this.numberOfSimulation = numberOfSimulation;
		this.sigma = sigma;
		this.numberOfTimes = numberOfTimes;
		this.dt = maturity / numberOfTimes;
		this.numberOfAsset = initialValue.length;
		this.rho = rho;
	}

	/**
	 * Generates asset price paths using the Euler-Maruyama scheme on log-prices.
	 * Standard normal innovations are correlated using the Cholesky lower
	 * triangular matrix.
	 */
	private void generateAssetPaths() {

		// Initialize the random noise generator for all simulations, time steps, and
		// assets
		MonteCarloGenerator noiseGenerator = new MonteCarloGenerator(
				numberOfSimulation * numberOfTimes * numberOfAsset);

		// Generate the flat array of standard normal random innovations
		double[] brownianMotion = noiseGenerator.generateNormalNumber();

		// Allocate the 3D storage matrix: [simulation][time + 1][asset]
		assetPaths = new double[numberOfSimulation][numberOfTimes + 1][numberOfAsset];

		// Compute the Cholesky decomposition of the correlation matrix (rho)
		double[][] choleskyDecomposition = computeCholeskyDecomposition(rho);

		// Loop over each independent Monte Carlo simulation path
		for (int simulation = 0; simulation < numberOfSimulation; simulation++) {

			// Initialize starting spot prices (at t = 0) for all assets
			for (int asset = 0; asset < numberOfAsset; asset++) {
				assetPaths[simulation][0][asset] = initialValue[asset];
			}

			// Iterate forward across the discrete time horizon
			for (int time = 0; time < numberOfTimes; time++) {

				// Extract independent standard normal draws (Z) for all assets at the current
				// step
				double[] Z = new double[numberOfAsset];
				for (int asset = 0; asset < numberOfAsset; asset++) {
					Z[asset] = brownianMotion[simulation * numberOfTimes * numberOfAsset + time * numberOfAsset
							+ asset];
				}

				// Apply Cholesky matrix to transform independent Z into correlated innovations
				// (epsilon)
				double[] epsilon = new double[numberOfAsset];
				for (int i = 0; i < numberOfAsset; i++) {
					for (int j = 0; j <= i; j++) {
						epsilon[i] += choleskyDecomposition[i][j] * Z[j];
					}
				}

				// Advance asset prices to the next time step using the Euler-Maruyama scheme
				for (int asset = 0; asset < numberOfAsset; asset++) {
					double currentPrice = assetPaths[simulation][time][asset];
					double drift = (riskFreeRate[asset] - 0.5 * sigma[asset] * sigma[asset]) * dt;
					double vol = sigma[asset] * Math.sqrt(dt);

					assetPaths[simulation][time + 1][asset] = currentPrice * Math.exp(drift + vol * epsilon[asset]);
				}
			}
		}
	}

	/**
	 * Computes the Cholesky decomposition of a symmetric positive-definite
	 * correlation matrix.
	 * 
	 * @param rho The input correlation matrix.
	 * @return The lower triangular Cholesky matrix.
	 */
	private double[][] computeCholeskyDecomposition(double[][] rho) {

		double[][] choleskyMatrices = new double[numberOfAsset][numberOfAsset];

		for (int i = 0; i < numberOfAsset; i++) {
			for (int j = 0; j <= i; j++) {

				double sum = 0;

				// Subtract the product of previous row elements
				for (int k = 0; k < j; k++) {
					sum += choleskyMatrices[i][k] * choleskyMatrices[j][k];
				}

				if (i == j) {
					// Diagonal elements: square root of variance adjustment
					choleskyMatrices[i][j] = Math.sqrt(rho[i][i] - sum);
				} else {
					// Off-diagonal elements
					choleskyMatrices[i][j] = (1.0 / choleskyMatrices[j][j] * (rho[i][j] - sum));
				}
			}
		}
		return choleskyMatrices;
	}

	@Override
	public double getDt() {
		return dt;
	}

	@Override
	public double[] getRiskFreeRate() {
		return riskFreeRate;
	}

	@Override
	public double[][][] getAssetPaths() {
		if (assetPaths == null) {
			generateAssetPaths();
		}
		return assetPaths;
	}

	@Override
	public int getNumberOfTimes() {
		return numberOfTimes;
	}
}