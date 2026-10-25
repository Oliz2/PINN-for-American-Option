package it.univr;


import java.util.function.DoubleUnaryOperator;
import it.univr.LongstaffSchwartzPricer.LongstaffSchwartzPricer;
import it.univr.model.BlackAndScholesModel;
import it.univr.model.MarketEnvironmentForModel;
import it.univr.monteCarloproducts.MonteCarloPricer;
import it.univr.option.BermudaOption;

/**
 * <p>
 * <strong>Monte Carlo Simulation and Validation Test Suite.</strong>
 * </p>
 * <p>
 * This class serves as the primary execution and testing harness for validating
 * the Least-Squares Monte Carlo (LSM) pricing engine against standard direct
 * forward Monte Carlo and established reference benchmarks (such as Table 1
 * from Longstaff & Schwartz, 2001).
 * </p>
 * 
 * @author Alberto Oliva Medin
 */
public class MonteCarloTest {

	public static void main(String[] args) {

		// =======================================================================
		// 1. MODEL & OPTION PARAMETERS SETUP
		// =======================================================================
		double[] initialValue = { 36.0 };
		double strike = 40.0;
		double[] riskFreeRate = { 0.06 };
		double[] sigma = { 0.2 };
		double[] weights = { 1.0 };
		double maturity = 1.0;
		double timeStep = 0.02;
		int numberOfSimulation = 100000;
		int numberOfTimes = (int) (maturity / timeStep);
		int timeToExercise = 1;
		double[][] rho = { { 1.0 } };

		// Initialize the global market environment with rates and asset weights
		MarketEnvironmentForModel.getValue(riskFreeRate, weights);

		// Define the option payoff function (Standard Put Option: max(K - S, 0))
		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		// Instantiate the Black-Scholes stochastic model
		BlackAndScholesModel blackAndScholes = new BlackAndScholesModel(initialValue, riskFreeRate, maturity,
				numberOfSimulation, sigma, numberOfTimes, rho);

		// Instantiate the Bermudan / American-style option contract
		BermudaOption americanPutOption = new BermudaOption(maturity, put, timeToExercise);

		// Instantiate and execute the Longstaff-Schwartz pricing engine
		LongstaffSchwartzPricer americanOtionWithRegression = new LongstaffSchwartzPricer(numberOfSimulation,
				numberOfTimes, blackAndScholes, americanPutOption);

		americanOtionWithRegression.calculateOptionPrice();

		// Instantiate standard forward Monte Carlo pricer for comparison
		MonteCarloPricer priceWithMonteCarlo = new MonteCarloPricer();

		System.out.println("The price of an american option with only Monte Carlo is: "
				+ priceWithMonteCarlo.getValue(americanPutOption, blackAndScholes));
		System.out.println(
				"The price with Monte Carlo with Longstaff-Schwartz: " + americanOtionWithRegression.getFinalPrice());

		System.out.println();

		// =======================================================================
		// 2. BATCH VALIDATION TEST AGAINST TABLE 1 (LSM vs Benchmark)
		// =======================================================================
		System.out.println("Monte Carlo test with Least Squares Monte Carlo (LSM):");

		// Benchmark arrays representing different test configurations from Longstaff &
		// Schwartz (2001)
		double[] sigmaArray = { 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20,
				0.40, 0.40, 0.20, 0.20, 0.40, 0.40 };
		double[] InitialValueArray = { 36.0, 36.0, 36.0, 36.0, 38.0, 38.0, 38.0, 38.0, 40.0, 40.0, 40.0, 40.0, 42.0,
				42.0, 42.0, 42.0, 44.0, 44.0, 44.0, 44.0 };
		double[] maturityArray = { 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
				2.0, 1.0, 2.0 };

		double[] priceWithLSM = new double[sigmaArray.length];

		// Loop through all 20 benchmark test configurations using LSM
		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = InitialValueArray[i];
			double sigmoide = sigmaArray[i];
			double M = maturityArray[i];

			double[] initialValues = { s0 };
			double[] sigmas = { sigmoide };

			BlackAndScholesModel blackAndScholesTest = new BlackAndScholesModel(initialValues, riskFreeRate, M,
					numberOfSimulation, sigmas, numberOfTimes, rho);

			BermudaOption americanPutOptionTest = new BermudaOption(M, put, timeToExercise);

			LongstaffSchwartzPricer americanOtionWithRegressionTest = new LongstaffSchwartzPricer(numberOfSimulation,
					numberOfTimes, blackAndScholesTest, americanPutOptionTest);

			americanOtionWithRegressionTest.calculateOptionPrice();

			priceWithLSM[i] = americanOtionWithRegressionTest.getFinalPrice();

			System.out.printf("S0: %.1f, Vol: %.2f, T: %.1f | Prezzo LSM: %.3f\n", s0, sigmoide, M, priceWithLSM[i]);
		}

		System.out.println();
		System.out.println("-------------------------------------------------------------");

		// =======================================================================
		// 3. BATCH VALIDATION TEST (Naive Monte Carlo Pricer)
		// =======================================================================
		System.out.println("Monte Carlo test (Naive Pricer):");

		double[] priceWithMC = new double[sigmaArray.length];

		// Loop through all 20 benchmark configurations using direct forward Monte Carlo
		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = InitialValueArray[i];
			double sigmoide = sigmaArray[i];
			double M = maturityArray[i];

			double[] initialValues = { s0 };
			double[] sigmas = { sigmoide };

			BlackAndScholesModel blackAndScholesTest = new BlackAndScholesModel(initialValues, riskFreeRate, M,
					numberOfSimulation, sigmas, numberOfTimes, rho);

			BermudaOption americanPutOptionTest = new BermudaOption(M, put, timeToExercise);

			MonteCarloPricer priceWithMonteCarloTest = new MonteCarloPricer();

			priceWithMC[i] = priceWithMonteCarloTest.getValue(americanPutOptionTest, blackAndScholesTest);

			System.out.printf("S0: %.1f, Vol: %.2f, T: %.1f | Prezzo Monte Carlo: %.3f\n", s0, sigmoide, M,
					priceWithMC[i]);
		}

		System.out.println();
		System.out.println("-------------------------------");

		// =======================================================================
		// 4. COMPARATIVE ANALYSIS (Absolute Differences)
		// =======================================================================
		System.out.println("Difference Analysis (LSM vs Naive Monte Carlo):");

		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = InitialValueArray[i];
			double sigmoide = sigmaArray[i];
			double M = maturityArray[i];

			double diff = Math.abs(((priceWithLSM[i] - priceWithMC[i])/priceWithLSM[i])*100);
			System.out.printf("S0: %.1f, Vol: %.2f, T: %.1f | Difference: %.3f%%\n", s0, sigmoide, M, diff);
		}
	}
}