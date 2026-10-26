package it.univr;

import it.univr.BinomialModel.BinomialModel;
import it.univr.LongstaffSchwartzPricer.LongstaffSchwartzPricer;
import it.univr.model.BlackAndScholesModel;
import it.univr.model.MarketEnvironmentForModel;
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
 * <strong>Monte Carlo Path &amp; Early-Exercise Export (for Python plots).</strong>
 * </p>
 * <p>
 * New standalone class that does NOT modify any existing pricing class.
 * It exports to CSV: (1) a subset of the paths simulated by the
 * Black-Scholes model, (2) the early-exercise time estimated by LSM for each
 * of those same paths, and (3) the early-exercise boundary computed with the
 * binomial tree — so that the paths, the LSM stopping-time points and the
 * "theoretical" boundary of the tree can all be overlaid on the same Python
 * plot.
 * </p>
 * <p>
 * [ADDED 2026-10-08] Export 3, {@code mc_exercise_boundary.csv}: it uses the
 * {@code BinomialModel}/{@code BermudaBinomialOption} classes (already in the
 * project, not touched here) to compute, at every time t, the critical price
 * S*(t) below which it is optimal to exercise the put immediately (that is,
 * where the immediate payoff exceeds the continuation value). This is the
 * early-exercise boundary: a curve that moves monotonically in time
 * (typically from a value close to K at maturity towards a lower value going
 * back in time) and delimits the "stopping region" — the red points of the
 * path plot (the times at which LSM decides to exercise) should fall near or
 * below this curve, not above it.
 * </p>
 * <p>
 * The computation uses only the public API already available on
 * {@code BinomialModel} ({@code getValuesAtGivenTimeIndex},
 * {@code getTransformedValuesAtGivenTimeIndex},
 * {@code getConditionalExpectation}) and on {@code BermudaBinomialOption}
 * ({@code getOptionValues}, i.e. the Snell envelope already computed): for
 * every t it rebuilds the comparison between exercise value and continuation
 * value, using the fact (standard result for an American put on a lognormal
 * underlying) that the exercise region is a connected interval
 * S &lt;= S*(t): it therefore looks for the highest price, among the tree
 * nodes at that time, at which exercising is still optimal.
 * </p>
 * <p>
 * Default parameters: same benchmark case used in {@code MonteCarloTest}
 * (S0=36, K=40, r=0.06, sigma=0.20, T=1, 50 exercise dates) — change them
 * freely below to export another configuration.
 * </p>
 *
 * [NEW file, 2026-10-08 — no change to the existing classes, it uses only
 * public APIs already available (BlackAndScholesModel.getAssetPaths(),
 * LongstaffSchwartzPricer.getExerciseTimes(), BinomialModel.*,
 * BermudaBinomialOption.getOptionValues())]
 *
 * @author Alberto Oliva Medin
 */
public class MonteCarloPathAndStoppingExport {

	public static void main(String[] args) throws Exception {

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
		int numberOfSimulation = 1000000;
		int numberOfTimes = (int) (maturity / timeStep);
		int timeToExercise = 1;
		double[][] rho = { { 1.0 } };

		// How many paths to export for the plot (not all the simulations: only a
		// readable subset)
		int pathsToExport = 40;

		// [ADDED 2026-10-08] Binomial tree steps used ONLY to compute the exercise
		// boundary (a huge number is not needed: it is for drawing the curve, not
		// for a very high precision price — see the note on the O(steps^2) memory
		// cost in MonteCarloConvergenceExport.java).
		int numberOfStepsBoundary = 500;

		MarketEnvironmentForModel.getValue(riskFreeRate, weights);

		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		BlackAndScholesModel model = new BlackAndScholesModel(initialValue, riskFreeRate, maturity,
				numberOfSimulation, sigma, numberOfTimes, rho);

		BermudaOption option = new BermudaOption(maturity, put, timeToExercise);

		LongstaffSchwartzPricer pricer = new LongstaffSchwartzPricer(numberOfSimulation, numberOfTimes, model,
				option);

		// Compute the price: this generates (and caches inside "model") the path
		// matrix and computes the exerciseTime values — exactly the same random
		// realisation used below for the export.
		pricer.calculateOptionPrice();
		System.out.println("LSM price (exported case): " + pricer.getFinalPrice());

		double[][][] paths = model.getAssetPaths();
		int[] exerciseTimes = pricer.getExerciseTimes();

		double dt = maturity / numberOfTimes;

		File outDir = new File("plot_data");
		outDir.mkdirs();

		// -----------------------------------------------------------------------
		// Export 1: simulated paths (for the "simulations over time" plot)
		// -----------------------------------------------------------------------
		File pathsFile = new File(outDir, "mc_paths.csv");
		try (PrintWriter writer = new PrintWriter(new FileWriter(pathsFile))) {
			writer.println("simulation_id,time_step,time_years,price");
			for (int sim = 0; sim < pathsToExport; sim++) {
				for (int t = 0; t <= numberOfTimes; t++) {
					double timeYears = t * dt;
					double price = paths[sim][t][0];
					// [FIX 2026-10-08] Locale.US forces the dot as the decimal separator,
					// whatever the Windows/JVM language (otherwise, on an Italian
					// system, printf writes a comma and breaks the CSV).
					writer.printf(Locale.US, "%d,%d,%.6f,%.6f%n", sim, t, timeYears, price);
				}
			}
		}
		System.out.println("Written: " + pathsFile.getAbsolutePath());

		// -----------------------------------------------------------------------
		// Export 2: early-exercise time per path (for the stopping-time region
		// plot, to be overlaid on the path plot)
		// -----------------------------------------------------------------------
		File exerciseFile = new File(outDir, "mc_exercise.csv");
		try (PrintWriter writer = new PrintWriter(new FileWriter(exerciseFile))) {
			writer.println("simulation_id,exercise_time_step,exercise_time_years,exercise_price,exercised_early");
			for (int sim = 0; sim < pathsToExport; sim++) {
				int et = exerciseTimes[sim];
				double timeYears = et * dt;
				double price = paths[sim][et][0];
				boolean exercisedEarly = et < numberOfTimes;
				// [FIX 2026-10-08] Locale.US, same reason as above.
				writer.printf(Locale.US, "%d,%d,%.6f,%.6f,%b%n", sim, et, timeYears, price, exercisedEarly);
			}
		}
		System.out.println("Written: " + exerciseFile.getAbsolutePath());

		// -----------------------------------------------------------------------
		// Export 3: early-exercise boundary from the binomial tree
		// -----------------------------------------------------------------------
		File boundaryFile = new File(outDir, "mc_exercise_boundary.csv");
		try (PrintWriter writer = new PrintWriter(new FileWriter(boundaryFile))) {
			writer.println("time_years,boundary_price");

			BinomialModel binomialModel = new BinomialModel(initialValue[0], riskFreeRate[0], sigma[0],
					numberOfStepsBoundary, maturity);
			BermudaBinomialOption binomialOption = new BermudaBinomialOption(put, numberOfStepsBoundary,
					numberOfStepsBoundary, binomialModel);
			// Force the computation of the Snell envelope (otherwise it is null until it
			// is requested for the first time)
			double[][] snellEnvelope = binomialOption.getOptionValues();

			double dtBoundary = maturity / numberOfStepsBoundary;

			for (int t = 0; t < numberOfStepsBoundary - 1; t++) {

				double timeYears = t * dtBoundary;

				// Possible prices of the underlying at this time (index 0 = highest price,
				// i.e. all up moves; index t = lowest price, i.e. all down moves)
				double[] assetValues = binomialModel.getValuesAtGivenTimeIndex(t);
				// Immediate exercise value at each node
				double[] exerciseValues = binomialModel.getTransformedValuesAtGivenTimeIndex(t, put);
				// Continuation value (discounted by one step) at each node, obtained from the
				// option values already computed at the next step (snellEnvelope[t+1])
				double[] continuationValues = binomialModel.getConditionalExpectation(snellEnvelope[t + 1], t);

				// The exercise region of an American put is a connected interval
				// S <= S*(t) (standard result): we scan from the lowest price (highest
				// index) upwards and take the highest price at which immediate exercise is
				// still optimal. If exercising is never optimal at this time (far from
				// maturity, OTM option), the boundary does not exist: we write NaN, which
				// pandas/matplotlib skip automatically when drawing the line.
				Double boundaryPrice = null;
				for (int i = assetValues.length - 1; i >= 0; i--) {
					if (exerciseValues[i] > continuationValues[i] + 1e-12) {
						boundaryPrice = assetValues[i];
					} else {
						break;
					}
				}

				if (boundaryPrice != null) {
					writer.printf(Locale.US, "%.6f,%.6f%n", timeYears, boundaryPrice);
				} else {
					writer.printf(Locale.US, "%.6f,%s%n", timeYears, "NaN");
				}
			}
			// At maturity the boundary of the put is by definition the strike: exercise
			// if and only if S(T) < K.
			writer.printf(Locale.US, "%.6f,%.6f%n", maturity, strike);
		}
		System.out.println("Written: " + boundaryFile.getAbsolutePath());
	}
}