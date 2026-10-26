package it.univr;

import it.univr.LongstaffSchwartzPricer.LongstaffSchwartzPricer;
import it.univr.model.BlackAndScholesModel;
import it.univr.model.MarketEnvironmentForModel;
import it.univr.model.ModelToUse;
import it.univr.option.BermudaOption;
import it.univr.option.IOption;
import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.Locale;
import java.util.function.DoubleUnaryOperator;

/**
 * Validation test suite for the Price Table of PINN1D (seed 2, run after the
 * non-negativity fix, 2026-10-12) against the Longstaff-Schwartz Monte Carlo
 * engine already in the project (package it.univr, montecarlo folder), plus
 * a closed-form analytical reference (European Black-Scholes).
 *
 * Parameters used in training, read directly from PINNTest1D.java lines
 * 228-234 (K=10, r=0.01, sigma=0.05, T=3, dividend=0). If you change them in
 * training, change them here too.
 *
 * To be placed under src/main/montecarlo/it/univr/validation/ (or src/test/...)
 * and run as a standalone class (it has a main()).
 */
public class MonteCarloValidationPINN1D {

	// ====== Contract / model parameters (CHECK that they match the training) ======
	private static final double K = 10.0;      // strike
	private static final double R = 0.01;      // risk-free rate
	private static final double SIGMA = 0.05;  // volatility
	private static final double T = 3.0;       // maturity (years)
	private static final double Q = 0.0;       // dividend (not used here, q=0)

	// ====== Monte Carlo parameters (increase them for more precision) ======
	private static final int DEFAULT_NUM_SIMULATIONS = 200_000;
	private static final int DEFAULT_NUM_TIME_STEPS = 100; // discretisation steps for tau=T; scaled pro rata for smaller tau

	// PINN Price Table, run 2 (after the fix), seed 2 — for the direct comparison.
	// Rows = S (8.00..12.00), columns = t (0.00..3.00 step 0.30).
	private static final double[] S_GRID = { 8.00, 8.44, 8.89, 9.33, 9.78, 10.22, 10.67, 11.11, 11.56, 12.00 };
	private static final double[] T_GRID = { 0.00, 0.30, 0.60, 0.90, 1.20, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00 };
	private static final double[][] PINN_PRICE_TABLE = {
			{ 3.8885, 3.7946, 3.6878, 3.5518, 3.3719, 3.1014, 2.7894, 2.5999, 2.4381, 2.2962, 2.1768 },
			{ 3.8001, 3.6603, 3.4502, 3.1414, 2.7805, 2.5733, 2.3850, 2.1983, 1.9999, 1.8669, 1.7638 },
			{ 3.4538, 3.2392, 2.8620, 2.5925, 2.3608, 2.1210, 1.9030, 1.7349, 1.5393, 1.3517, 1.2262 },
			{ 2.2902, 2.1867, 1.9675, 1.7475, 1.4708, 1.2650, 1.1294, 1.0137, 0.9099, 0.8108, 0.7113 },
			{ 0.5836, 0.4859, 0.4428, 0.4303, 0.4345, 0.4424, 0.4445, 0.4338, 0.4046, 0.3500, 0.2605 },
			{ 0.1593, 0.1243, 0.1104, 0.1033, 0.0993, 0.0973, 0.0957, 0.0926, 0.0866, 0.0767, 0.0647 },
			{ 0.0825, 0.0747, 0.0669, 0.0602, 0.0556, 0.0529, 0.0514, 0.0507, 0.0503, 0.0501, 0.0500 },
			{ 0.0535, 0.0519, 0.0511, 0.0506, 0.0502, 0.0501, 0.0500, 0.0499, 0.0500, 0.0500, 0.0500 },
			{ 0.0503, 0.0501, 0.0500, 0.0500, 0.0499, 0.0499, 0.0499, 0.0499, 0.0499, 0.0499, 0.0498 },
			{ 0.0499, 0.0499, 0.0499, 0.0499, 0.0499, 0.0499, 0.0498, 0.0496, 0.0494, 0.0493, 0.0487 },
	};

	public static void main(String[] args) {
		MarketEnvironmentForModel.getValue(new double[] { R }, new double[] { 1.0 });

		System.out.println("=== TEST 0 — Terminal condition (tau=0): must match max(K-S,0) EXACTLY ===");
		test0_TerminalCondition();

		System.out.println("\n=== TEST 1 — Full grid: PINN vs Longstaff-Schwartz (American) vs closed-form Black-Scholes (European) ===");
		test1_FullGridComparison();

		System.out.println("\n=== TEST 2 — Monte Carlo convergence at the most delicate point (S=9.78, t=0.00, right on the boundary) ===");
		test2_ConvergenceCheck(9.78, 0.00);

		System.out.println("\n=== TEST 3 — Early-exercise premium (American - European) on the deep OTM rows (S=10.67..12.00) ===");
		test3_EarlyExercisePremiumDeepOTM();
	}

	// ------------------------------------------------------------------
	// TEST 0: at tau=0 the price is deterministic and must be EXACTLY
	// max(K-S,0), with no simulation needed. It answers the question whether
	// the t=3.00 column of the Price Table really is the maturity.
	// ------------------------------------------------------------------
	private static void test0_TerminalCondition() {
		double[] testSpots = { 8.00, 8.44, 8.89, 9.33, 9.78, 10.22, 11.11, 12.00 };
		for (double s : testSpots) {
			double intrinsic = Math.max(K - s, 0.0);
			double mc = americanPutLSM(s, 1e-6, 1000, 1); // tau almost zero, nothing really to simulate
			System.out.printf(Locale.ROOT, "S=%.2f | intrinsic=%.4f | MC(tau~0)=%.4f | PINN at t=3.00=%.4f%n",
					s, intrinsic, mc, pinnPriceAt(s, 3.00));
		}
		System.out.println("(If the MC(tau~0) column matches the intrinsic value here, the terminal condition is confirmed:");
		System.out.println(" the difference from \"PINN at t=3.00\" is then a residual error of the network, not an interpretation error.)");
	}

	// ------------------------------------------------------------------
	// TEST 1: cell-by-cell comparison over the whole Price Table grid.
	// Note: with 10x11=110 cells and DEFAULT_NUM_SIMULATIONS=200000 this can
	// take a few minutes. Reduce the grid (e.g. only t=0.00/1.50/3.00) for a
	// quick first run, if you want.
	// ------------------------------------------------------------------
	private static void test1_FullGridComparison() {
		System.out.printf(Locale.ROOT, "%-6s %-6s | %-10s %-10s %-10s | %-10s%n",
				"S", "t", "PINN", "MC_Amer", "BS_Euro", "PINN-MC");
		for (int i = 0; i < S_GRID.length; i++) {
			double s = S_GRID[i];
			for (int j = 0; j < T_GRID.length; j++) {
				double t0 = T_GRID[j];
				double tau = T - t0;
				double pinnPrice = PINN_PRICE_TABLE[i][j];
				double mcPrice = (tau <= 1e-9) ? Math.max(K - s, 0.0)
						: americanPutLSM(s, tau, DEFAULT_NUM_SIMULATIONS, scaledSteps(tau));
				double bsEuro = europeanPutClosedForm(s, tau);
				System.out.printf(Locale.ROOT, "%-6.2f %-6.2f | %-10.4f %-10.4f %-10.4f | %+-10.4f%n",
						s, t0, pinnPrice, mcPrice, bsEuro, pinnPrice - mcPrice);
			}
		}
	}

	// ------------------------------------------------------------------
	// TEST 2: repeat the same cell with an increasing number of simulations to
	// see when the MC price stabilises (how much noise there is in the "truth"
	// the PINN is compared with).
	// ------------------------------------------------------------------
	private static void test2_ConvergenceCheck(double s, double t0) {
		double tau = T - t0;
		int[] simCounts = { 5_000, 20_000, 50_000, 100_000, 300_000, 600_000 };
		for (int n : simCounts) {
			double price = americanPutLSM(s, tau, n, scaledSteps(tau));
			System.out.printf(Locale.ROOT, "numSim=%7d -> price=%.5f%n", n, price);
		}
		System.out.printf(Locale.ROOT, "PINN on this cell (S=%.2f, t=%.2f): %.4f%n", s, t0, pinnPriceAt(s, t0));
		System.out.println("(If the MC price still changes a lot even at 300k-600k simulations, increase numSim further");
		System.out.println(" before trusting the comparison on this cell: it is the hardest one, right on the exercise boundary.)");
	}

	// ------------------------------------------------------------------
	// TEST 3: measures the early-exercise premium (American - European) in the
	// deep OTM tail, where the PINN Price Table shows the almost constant
	// plateau ~0.05 that worries us. If the Monte Carlo here shows values that
	// clearly DECAY with S and t, the PINN plateau is probably a symptom of
	// under-training (or of the saturating sigmoid constraint), not the
	// economic reality.
	// ------------------------------------------------------------------
	private static void test3_EarlyExercisePremiumDeepOTM() {
		double[] deepOtmSpots = { 10.67, 11.11, 11.56, 12.00 };
		double[] slices = { 0.00, 1.50, 3.00 };
		for (double s : deepOtmSpots) {
			for (double t0 : slices) {
				double tau = T - t0;
				double american = (tau <= 1e-9) ? Math.max(K - s, 0.0)
						: americanPutLSM(s, tau, DEFAULT_NUM_SIMULATIONS, scaledSteps(tau));
				double european = europeanPutClosedForm(s, tau);
				System.out.printf(Locale.ROOT,
						"S=%.2f t=%.2f | American(MC)=%.5f | European(closed)=%.5f | premium=%.5f | PINN=%.4f%n",
						s, t0, american, european, american - european, pinnPriceAt(s, t0));
			}
		}
	}

	// ------------------------------------------------------------------
	// Helpers
	// ------------------------------------------------------------------

	/** Prices an American put with Longstaff-Schwartz, spot S0 and time to maturity tau. */
	private static double americanPutLSM(double s0, double tau, int numSimulations, int numTimeSteps) {
		ModelToUse model = new BlackAndScholesModel(new double[] { s0 }, new double[] { R }, tau, numSimulations,
				new double[] { SIGMA }, numTimeSteps, new double[][] { { 1.0 } });
		DoubleUnaryOperator payoff = s -> Math.max(K - s, 0.0);
		IOption put = new BermudaOption(tau, payoff, 1);
		LongstaffSchwartzPricer pricer = new LongstaffSchwartzPricer(numSimulations, numTimeSteps, model, put);
		return pricer.getFinalPrice();
	}

	/** Black-Scholes closed formula for a European put (q=0). No MC noise. */
	private static double europeanPutClosedForm(double s0, double tau) {
		if (tau <= 1e-9) {
			return Math.max(K - s0, 0.0);
		}
		NormalDistribution normal = new NormalDistribution();
		double d1 = (Math.log(s0 / K) + (R - Q + 0.5 * SIGMA * SIGMA) * tau) / (SIGMA * Math.sqrt(tau));
		double d2 = d1 - SIGMA * Math.sqrt(tau);
		return K * Math.exp(-R * tau) * normal.cumulativeProbability(-d2)
				- s0 * Math.exp(-Q * tau) * normal.cumulativeProbability(-d1);
	}

	/** Scales the number of time steps in proportion to tau, with a minimum of 5. */
	private static int scaledSteps(double tau) {
		int steps = (int) Math.round(DEFAULT_NUM_TIME_STEPS * (tau / T));
		return Math.max(steps, 5);
	}

	/** Reads the PINN Price Table value at the closest (S,t) cell of the grid. */
	private static double pinnPriceAt(double s, double t0) {
		int iBest = 0, jBest = 0;
		double bestDi = Double.MAX_VALUE, bestDj = Double.MAX_VALUE;
		for (int i = 0; i < S_GRID.length; i++) {
			double d = Math.abs(S_GRID[i] - s);
			if (d < bestDi) {
				bestDi = d;
				iBest = i;
			}
		}
		for (int j = 0; j < T_GRID.length; j++) {
			double d = Math.abs(T_GRID[j] - t0);
			if (d < bestDj) {
				bestDj = d;
				jBest = j;
			}
		}
		return PINN_PRICE_TABLE[iBest][jBest];
	}
}