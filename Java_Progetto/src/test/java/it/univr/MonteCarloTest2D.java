package it.univr;

import it.univr.LongstaffSchwartzPricer.*;
import it.univr.model.*;
import it.univr.monteCarloproducts.MonteCarloPricer;
import it.univr.option.BermudaOption;
import java.util.function.DoubleUnaryOperator;

/**
 * <p>
 * <strong>2D Monte Carlo Simulation and Validation Test Suite.</strong>
 * </p>
 * <p>
 * This class serves as a multi-asset (2D) extension test harness for evaluating
 * American and Bermudan style options under correlated Black-Scholes dynamics,
 * comparing Least-Squares Monte Carlo (LSM) against a naive forward Monte Carlo
 * pricer.
 * </p>
 * 
 * @author Alberto Oliva Medin
 */
public class MonteCarloTest2D {

	public static void main(String[] args) {

		// =======================================================================
		// 1. 2D MODEL & OPTION PARAMETERS SETUP
		// =======================================================================
		double[] initialValue = { 36.0, 36.0 };
		double strike = 40.0;
		double[] riskFreeRate = { 0.06, 0.06 };
		double[] sigma = { 0.2, 0.2 };
		double[] weights = { 0.5, 0.5 };
		double maturity = 1.0;
		double timeStep = 0.03;
		int numberOfSimulation = 1000000;
		int numberOfTimes = (int) (maturity / timeStep);
		int timeToExercise = 1;
		double[][] rho = { { 1.0, 0.5 }, { 0.5, 1.0 } };

		// Initialize the global market environment with multi-asset rates and weights
		MarketEnvironmentForModel.getValue(riskFreeRate, weights);

		// Define the option payoff function (Put Option evaluated on the first asset:
		// max(K - S, 0))
		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		// Instantiate the 2D Black-Scholes stochastic model with correlation matrix
		// (rho)
		BlackAndScholesModel blackAndScholes = new BlackAndScholesModel(initialValue, riskFreeRate, maturity,
				numberOfSimulation, sigma, numberOfTimes, rho);

		// Instantiate the Bermudan option contract
		BermudaOption americanPutOption = new BermudaOption(maturity, put, timeToExercise);

		// Instantiate and execute the Longstaff-Schwartz 2D pricing engine
		LongstaffSchwartzPricer americanOtionWithRegression = new LongstaffSchwartzPricer(numberOfSimulation,
				numberOfTimes, blackAndScholes, americanPutOption);

		americanOtionWithRegression.calculateOptionPrice();

		// Instantiate standard forward Monte Carlo pricer for baseline comparison
		MonteCarloPricer priceWithMonteCarlo = new MonteCarloPricer();

		System.out.println("The price of a 2D american option with only Monte Carlo is: "
				+ priceWithMonteCarlo.getValue(americanPutOption, blackAndScholes));
		System.out.println("The price with Monte Carlo with Longstaff-Schwartz (2D): "
				+ americanOtionWithRegression.getFinalPrice());

		System.out.println();

		// =======================================================================
		// 2. BATCH VALIDATION TEST (2D LSM Benchmark Configurations)
		// =======================================================================
		System.out.println("2D Monte Carlo test with Least Squares Monte Carlo (LSM):");

		double[] sigmaArray = { 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20,
				0.40, 0.40, 0.20, 0.20, 0.40, 0.40 };
		double[] InitialValueArray = { 36.0, 36.0, 36.0, 36.0, 38.0, 38.0, 38.0, 38.0, 40.0, 40.0, 40.0, 40.0, 42.0,
				42.0, 42.0, 42.0, 44.0, 44.0, 44.0, 44.0 };
		double[] maturityArray = { 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
				2.0, 1.0, 2.0 };

		double[] priceWithLSM = new double[sigmaArray.length];

		// Loop through all 20 configurations using the 2D LSM model
		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = InitialValueArray[i];
			double sigmoide = sigmaArray[i];
			double M = maturityArray[i];

			double[] initialValues = { s0, s0 };
			double[] sigmas = { sigmoide, sigmoide };

			BlackAndScholesModel blackAndScholesTest = new BlackAndScholesModel(initialValues, riskFreeRate, M,
					numberOfSimulation, sigmas, numberOfTimes, rho);

			BermudaOption americanPutOptionTest = new BermudaOption(M, put, timeToExercise);

			LongstaffSchwartzPricer americanOtionWithRegressionTest = new LongstaffSchwartzPricer(numberOfSimulation,
					numberOfTimes, blackAndScholesTest, americanPutOptionTest);

			americanOtionWithRegressionTest.calculateOptionPrice();

			priceWithLSM[i] = americanOtionWithRegressionTest.getFinalPrice();

			System.out.printf("S0_1: %.1f, S0_2: %.1f | Vol: %.2f | T: %.1f | Prezzo LSM (2D): %.3f\n",
					initialValues[0], initialValues[1], sigmoide, M, priceWithLSM[i]);
		}

		System.out.println();
		System.out.println("-------------------------------------------------------------");

		// =======================================================================
		// 3. BATCH VALIDATION TEST (2D Naive Monte Carlo Pricer)
		// =======================================================================
		System.out.println("2D Monte Carlo test (Naive Pricer):");

		double[] priceWithMC = new double[sigmaArray.length];

		// Loop through all 20 configurations using direct forward Monte Carlo
		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = InitialValueArray[i];
			double sigmoide = sigmaArray[i];
			double M = maturityArray[i];

			double[] initialValues = { s0, s0 };
			double[] sigmas = { sigmoide, sigmoide };

			BlackAndScholesModel blackAndScholesTest = new BlackAndScholesModel(initialValues, riskFreeRate, M,
					numberOfSimulation, sigmas, numberOfTimes, rho);

			BermudaOption americanPutOptionTest = new BermudaOption(M, put, timeToExercise);

			MonteCarloPricer priceWithMonteCarloTest = new MonteCarloPricer();

			priceWithMC[i] = priceWithMonteCarloTest.getValue(americanPutOptionTest, blackAndScholesTest);

			System.out.printf("S0_1: %.1f, S0_2: %.1f | Vol: %.2f | T: %.1f | Prezzo Monte Carlo (2D): %.3f\n",
					initialValues[0], initialValues[1], sigmoide, M, priceWithMC[i]);
		}

		System.out.println();
		System.out.println("-------------------------------");

		// =======================================================================
		// 4. COMPARATIVE ANALYSIS (2D Absolute Differences)
		// =======================================================================
		System.out.println("Difference Analysis (2D LSM vs Naive Monte Carlo):");

		for (int i = 0; i < 20; i++) {
			double s0 = InitialValueArray[i];
			double sigmoide = sigmaArray[i];
			double M = maturityArray[i];

			double diff = Math.abs(((priceWithLSM[i] - priceWithMC[i])/priceWithLSM[i])* 100);
			System.out.printf("S0: %.1f | Vol: %.2f | T: %.1f | Difference: %.3f%%\n", s0, sigmoide, M, diff);
		}
	}
}