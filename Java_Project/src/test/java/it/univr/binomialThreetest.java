package it.univr;

import java.util.function.DoubleUnaryOperator;

import it.univr.BinomialModel.BinomialModel;
import it.univr.LongstaffSchwartzPricer.LongstaffSchwartzPricer;
import it.univr.model.BlackAndScholesModel;
import it.univr.model.MarketEnvironmentForModel;
import it.univr.option.BermudaBinomialOption;
import it.univr.option.BermudaOption;

/**
 * <p>
 * <strong>Binomial Tree (CRR) Test &amp; Cross-Validation vs LSM.</strong>
 * </p>
 * <p>
 * Uses the binomial tree classes already in the project
 * ({@code BinomialModel}, {@code BermudaBinomialOption}, backward induction
 * through the Snell envelope) to: (1) check that the price converges as the
 * number of tree steps grows, and (2) compare the tree price with the
 * Longstaff-Schwartz price on the same 20 benchmark cases used in
 * {@code MonteCarloTest.java} (Longstaff &amp; Schwartz, 2001, Table 1).
 * </p>
 * <p>
 * Why this comparison makes sense: the binomial tree and LSM are two
 * independent numerical methods (dynamic programming on a tree vs.
 * regression on Monte Carlo simulations) that estimate the SAME theoretical
 * quantity, the price of the American/Bermudan put. If both are implemented
 * correctly they must converge to the same value. Here the tree
 * (deterministic, no statistical noise) acts as the "exact" reference to
 * validate LSM, instead of relying only on a literature value copied by
 * hand.
 * </p>
 * <p>
 * [NOTE] Checked locally (sandbox, parameters S0=36,K=40,r=0.06,
 * sigma=0.20,T=1): {@code BermudaBinomialOption} converges correctly,
 * from 4.4818 (50 steps) to 4.4866 (5000 steps), in line both with the
 * literature value (~4.486) and with an independently written CRR tree used
 * as a check. A minor detail, visible only with very many steps: both
 * {@code BinomialModel} and {@code BermudaBinomialOption} store full
 * {@code numberOfTimes x numberOfTimes} matrices (O(N^2) memory instead of
 * O(N) as in a tree with "rolling" storage), so with many steps (e.g.
 * 10000+) the JVM may need more heap (Eclipse: Run Configurations -&gt;
 * Arguments -&gt; VM arguments -&gt; {@code -Xmx2g} or more); for this
 * reason the convergence test below stops at 5000 steps, where the price is
 * already stable to the third decimal place.
 * </p>
 *
 * [NEW file, 2026-10-08 — no change to the existing pricing classes]
 *
 * @author Alberto Oliva Medin
 */
public class binomialThreetest {

	public static void main(String[] args) throws Exception {

		// =======================================================================
		// 1. PARAMETERS (same benchmark case as MonteCarloTest.java)
		// =======================================================================
		double initialValue = 36.0;
		double strike = 40.0;
		double riskFreeRate = 0.06;
		double sigma = 0.20;
		double maturity = 1.0;

		// Tree steps used for the "single" price and for the comparison with LSM
		// below. 2000 is already well converged (see the convergence test) and
		// stays fast even when repeated 20 times in the comparison loop.
		int numberOfStepsBinomial = 2000;

		MarketEnvironmentForModel.getValue(new double[] { riskFreeRate }, new double[] { 1.0 });

		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		// =======================================================================
		// 2. SINGLE PRICE WITH THE BINOMIAL TREE
		// =======================================================================
		BinomialModel model = new BinomialModel(initialValue, riskFreeRate, sigma, numberOfStepsBinomial, maturity);
		BermudaBinomialOption binomialOption = new BermudaBinomialOption(put, numberOfStepsBinomial,
				numberOfStepsBinomial, model);
		double priceBinomial = binomialOption.getOptionValues()[0][0];

		System.out.println(
				"Price with the binomial tree (" + numberOfStepsBinomial + " steps): " + priceBinomial);

		// =======================================================================
		// 3. CONVERGENCE TEST: price as the number of tree steps grows
		// =======================================================================
		System.out.println();
		System.out.println("Convergence of the binomial tree as the number of steps grows:");
		int[] stepsToTest = { 50, 100, 500, 1000, 2000, 5000 };
		for (int steps : stepsToTest) {
			BinomialModel modelStep = new BinomialModel(initialValue, riskFreeRate, sigma, steps, maturity);
			BermudaBinomialOption optionStep = new BermudaBinomialOption(put, steps, steps, modelStep);
			double priceStep = optionStep.getOptionValues()[0][0];
			System.out.printf("  steps=%6d -> price=%.6f%n", steps, priceStep);
		}

		// =======================================================================
		// 4. BINOMIAL TREE vs LSM on the 20 cases of Longstaff & Schwartz
		// (2001), Table 1 — same parameters/loop as MonteCarloTest.java
		// =======================================================================
		System.out.println();
		System.out.println("-------------------------------------------------------------");
		System.out.println("Binomial Tree vs Longstaff-Schwartz (LSM):");

		double[] sigmaArray = { 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20,
				0.40, 0.40, 0.20, 0.20, 0.40, 0.40 };
		double[] initialValueArray = { 36.0, 36.0, 36.0, 36.0, 38.0, 38.0, 38.0, 38.0, 40.0, 40.0, 40.0, 40.0, 42.0,
				42.0, 42.0, 42.0, 44.0, 44.0, 44.0, 44.0 };
		double[] maturityArray = { 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
				2.0, 1.0, 2.0 };

		// LSM parameters (identical to MonteCarloTest.java, for a fair comparison)
		int numberOfSimulationLSM = 100000;
		double timeStepLSM = 0.02;
		int timeToExerciseLSM = 1;
		double[] riskFreeRateArr = { riskFreeRate };
		double[][] rho = { { 1.0 } };

		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = initialValueArray[i];
			double vol = sigmaArray[i];
			double M = maturityArray[i];

			// --- Price with the binomial tree (deterministic reference) ---
			BinomialModel modelI = new BinomialModel(s0, riskFreeRate, vol, numberOfStepsBinomial, M);
			BermudaBinomialOption optionI = new BermudaBinomialOption(put, numberOfStepsBinomial,
					numberOfStepsBinomial, modelI);
			double priceBinomialI = optionI.getOptionValues()[0][0];

			// --- Price with LSM ---
			int numberOfTimesLSM = (int) (M / timeStepLSM);
			double[] initialValues = { s0 };
			double[] sigmas = { vol };
			BlackAndScholesModel bsModel = new BlackAndScholesModel(initialValues, riskFreeRateArr, M,
					numberOfSimulationLSM, sigmas, numberOfTimesLSM, rho);
			BermudaOption lsmOption = new BermudaOption(M, put, timeToExerciseLSM);
			LongstaffSchwartzPricer lsmPricer = new LongstaffSchwartzPricer(numberOfSimulationLSM, numberOfTimesLSM,
					bsModel, lsmOption);
			lsmPricer.calculateOptionPrice();
			double priceLSM = lsmPricer.getFinalPrice();

			// Percentage difference from the binomial tree, which here acts as the
			// "exact" reference (deterministic, no statistical noise) — same logic
			// already used in MonteCarloTest.java to compare LSM with the reference.
			double diffPct = Math.abs((priceBinomialI - priceLSM) / priceBinomialI) * 100;

			System.out.printf(
					"S0: %.1f, Vol: %.2f, T: %.1f | Binomial: %.3f | LSM: %.3f | Difference: %.3f%%\n",
					s0, vol, M, priceBinomialI, priceLSM, diffPct);
		}
	}
}