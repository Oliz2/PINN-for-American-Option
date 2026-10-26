package it.univr;

import it.univr.BinomialModel.BinomialModel;
import it.univr.LongstaffSchwartzPricer.LongstaffSchwartzPricer;
import it.univr.model.BlackAndScholesModel;
import it.univr.model.MarketEnvironmentForModel;
import it.univr.monteCarloproducts.MonteCarloPricer;
import it.univr.option.BermudaBinomialOption;
import it.univr.option.BermudaOption;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.function.DoubleUnaryOperator;

/**
 * <p>
 * <strong>Monte Carlo Convergence Export (for Python plots).</strong>
 * </p>
 * <p>
 * New standalone class that does NOT modify any existing pricing class.
 * For an increasing number of simulations, it recomputes the price with both
 * Monte Carlo methods (Naive and Longstaff-Schwartz) and writes the results
 * to CSV, to show that both converge to the same value as N grows.
 * </p>
 * <p>
 * [ADDED 2026-10-08] Added a third method, "BinomialTree": it uses the
 * {@code BinomialModel}/{@code BermudaBinomialOption} classes (already in the
 * project, not touched here) to compute the price with the CRR binomial tree.
 * Unlike NaiveMC and LSM it has no statistical noise (no random
 * simulation), so it acts as a deterministic reference: its price is already
 * stable after a few hundred/thousand steps — see
 * {@code BinomialTreeTest.java}, where this has already been checked (4.4866
 * at 5000 steps, in line with the literature value 4.486).
 * </p>
 * <p>
 * [IMPORTANT] Unlike NaiveMC/LSM, the binomial tree of this project stores
 * full N x N matrices (O(N^2) memory, not O(N) as in a tree with "rolling"
 * storage): at N=10000 it already needs a few GB of heap, and at N=100000+
 * (the largest values used for NaiveMC/LSM) it would need terabytes of RAM,
 * which is simply impossible. For this reason the BinomialTree method does
 * NOT reuse the {@code simulationCounts} array of NaiveMC/LSM, but its own
 * {@code binomialSteps} array, denser in the low range (where the curve
 * flattens) and with a maximum of 5000 steps. In the CSV/plot the
 * BinomialTree curve will therefore have more closely spaced points but will
 * stop before the other two on the x axis — which is natural: it is a
 * deterministic method and does not need as many "steps" as Monte Carlo
 * needs to reduce the statistical noise.
 * </p>
 * <p>
 * [FIX 2026-10-08] Added {@code Locale.US} to the printf calls: without it,
 * on a JVM with an Italian locale numbers are written with a comma as the
 * decimal separator, which corrupts the CSV (it adds to the comma that
 * separates the columns). See the same fix already applied in
 * {@code MonteCarloPathAndStoppingExport.java}.
 * </p>
 * <p>
 * Default parameters: same benchmark case used in {@code MonteCarloTest}
 * (S0=36, K=40, r=0.06, sigma=0.20, T=1) — it is one of the cases of
 * Longstaff &amp; Schwartz (2001), Table 1, for which a published reference
 * price computed with a finite-difference method exists (see the comment in
 * the Python script for the exact value and the source).
 * </p>
 *
 * @author Alberto Oliva Medin
 */
public class MonteCarloConvergenceExport {

	public static void main(String[] args) throws IOException {

		// =======================================================================
		// PARAMETERS (same benchmark case as MonteCarloTest — can be changed)
		// =======================================================================
		double[] initialValue = { 36.0 };
		double strike = 40.0;
		double[] riskFreeRate = { 0.06 };
		double[] sigma = { 0.2 };
		double[] weights = { 1.0 };
		double maturity = 1.0;
		double timeStep = 0.02;
		int numberOfTimes = (int) (maturity / timeStep);
		int timeToExercise = 1;
		double[][] rho = { { 1.0 } };

		// Values of N (number of simulations) used to measure the convergence of
		// NaiveMC and LSM
		int[] simulationCounts = { 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000 };

		// [ADDED 2026-10-08] Values of STEPS (not simulations) used to measure the
		// convergence of the binomial tree: a separate, denser array, since the
		// tree converges with very few steps compared with Monte Carlo (see the
		// [IMPORTANT] note in the Javadoc above on the O(steps^2) memory cost, which
		// in any case imposes a maximum here of 5000).
		int[] binomialSteps = { 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000 };

		MarketEnvironmentForModel.getValue(riskFreeRate, weights);

		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		File outDir = new File("plot_data");
		outDir.mkdirs();
		File outFile = new File(outDir, "mc_convergence.csv");

		try (PrintWriter writer = new PrintWriter(new FileWriter(outFile))) {
			writer.println("method,numberOfSimulation,price");

			for (int n : simulationCounts) {

				BlackAndScholesModel modelForNaive = new BlackAndScholesModel(initialValue, riskFreeRate, maturity, n,
						sigma, numberOfTimes, rho);
				BermudaOption optionForNaive = new BermudaOption(maturity, put, timeToExercise);
				MonteCarloPricer naivePricer = new MonteCarloPricer();
				double naivePrice = naivePricer.getValue(optionForNaive, modelForNaive);
				writer.printf(Locale.US, "NaiveMC,%d,%.6f%n", n, naivePrice);
				System.out.printf("NaiveMC      N=%7d -> price=%.4f%n", n, naivePrice);

				BlackAndScholesModel modelForLsm = new BlackAndScholesModel(initialValue, riskFreeRate, maturity, n,
						sigma, numberOfTimes, rho);
				BermudaOption optionForLsm = new BermudaOption(maturity, put, timeToExercise);
				LongstaffSchwartzPricer lsmPricer = new LongstaffSchwartzPricer(n, numberOfTimes, modelForLsm,
						optionForLsm);
				double lsmPrice = lsmPricer.getFinalPrice();
				writer.printf(Locale.US, "LSM,%d,%.6f%n", n, lsmPrice);
				System.out.printf("LSM          N=%7d -> price=%.4f%n", n, lsmPrice);
			}

			// [ADDED 2026-10-08] Third, deterministic method: CRR binomial tree.
			// Separate loop over its own array of steps (binomialSteps), denser in the
			// low range, since the tree converges with few steps (see Javadoc). The
			// "numberOfSimulation" values written to the CSV for this method are
			// actually the number of tree STEPS: they still share the x axis (log
			// scale) of the Python plot, so no change is needed there.
			for (int steps : binomialSteps) {
				try {
					BinomialModel binomialModel = new BinomialModel(initialValue[0], riskFreeRate[0], sigma[0],
							steps, maturity);
					BermudaBinomialOption binomialOption = new BermudaBinomialOption(put, steps, steps,
							binomialModel);
					double binomialPrice = binomialOption.getOptionValues()[0][0];
					writer.printf(Locale.US, "BinomialTree,%d,%.6f%n", steps, binomialPrice);
					System.out.printf("BinomialTree steps=%7d -> price=%.4f%n", steps, binomialPrice);
				} catch (Exception e) {
					// BermudaBinomialOption declares "throws Exception" in its constructor (for
					// the check numberOfStepsOption <= numberOfStepsStock, always true here
					// since we pass them equal): it should never fire, but we log it instead
					// of stopping the whole export if it does.
					System.out.println("BinomialTree steps=" + steps + " -> error: " + e.getMessage());
				}
			}
		}

		System.out.println("Written: " + outFile.getAbsolutePath());
	}
}