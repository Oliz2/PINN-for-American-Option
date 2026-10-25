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
 * Batteria di test di validazione del Price Table del PINN1D (seed 2, run
 * post-fix non-negativita', 2026-10-12) contro il motore Monte Carlo
 * Longstaff-Schwartz gia' presente nel progetto (package it.univr, cartella
 * montecarlo), piu' un riferimento analitico chiuso (Black-Scholes europea).
 *
 * Parametri usati in training, letti direttamente da PINNTest1D.java righe
 * 228-234 (K=10, r=0.01, sigma=0.05, T=3, dividendo=0). Se li cambi nel
 * training, cambiali anche qui.
 *
 * Da posizionare sotto src/main/montecarlo/it/univr/validation/ (o src/test/...)
 * e lanciare come classe standalone (ha un main()).
 */
public class MonteCarloValidationPINN1D {

	// ====== Parametri del contratto / modello (CONFERMA che combacino col training) ======
	private static final double K = 10.0;      // strike
	private static final double R = 0.01;      // risk-free rate
	private static final double SIGMA = 0.05;  // volatilita'
	private static final double T = 3.0;       // maturity (anni)
	private static final double Q = 0.0;       // dividendo (non usato qui, q=0)

	// ====== Parametri del Monte Carlo (puoi aumentarli se vuoi piu' precisione) ======
	private static final int DEFAULT_NUM_SIMULATIONS = 200_000;
	private static final int DEFAULT_NUM_TIME_STEPS = 100; // passi di discretizzazione per tau=T; scalati pro-rata per tau piu' piccoli

	// Price Table del PINN, run 2 (post-fix), seed 2 — per il confronto diretto.
	// Righe = S (8.00..12.00), colonne = t (0.00..3.00 passo 0.30).
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

		System.out.println("=== TEST 0 — Condizione terminale (tau=0): deve combaciare ESATTAMENTE con max(K-S,0) ===");
		test0_TerminalCondition();

		System.out.println("\n=== TEST 1 — Griglia completa: PINN vs Longstaff-Schwartz (americana) vs Black-Scholes chiusa (europea) ===");
		test1_FullGridComparison();

		System.out.println("\n=== TEST 2 — Convergenza Monte Carlo sul punto piu' delicato (S=9.78, t=0.00, proprio sul confine) ===");
		test2_ConvergenceCheck(9.78, 0.00);

		System.out.println("\n=== TEST 3 — Premio di esercizio anticipato (Americana - Europea) sulla riga OTM profonda (S=10.67..12.00) ===");
		test3_EarlyExercisePremiumDeepOTM();
	}

	// ------------------------------------------------------------------
	// TEST 0: a tau=0 il prezzo e' deterministico e deve essere ESATTAMENTE
	// max(K-S,0), senza bisogno di nessuna simulazione. Risolve la domanda se
	// la colonna t=3.00 della Price Table sia davvero la scadenza.
	// ------------------------------------------------------------------
	private static void test0_TerminalCondition() {
		double[] testSpots = { 8.00, 8.44, 8.89, 9.33, 9.78, 10.22, 11.11, 12.00 };
		for (double s : testSpots) {
			double intrinsic = Math.max(K - s, 0.0);
			double mc = americanPutLSM(s, 1e-6, 1000, 1); // tau quasi zero, niente da simulare davvero
			System.out.printf(Locale.ROOT, "S=%.2f | intrinseco=%.4f | MC(tau~0)=%.4f | PINN a t=3.00=%.4f%n",
					s, intrinsic, mc, pinnPriceAt(s, 3.00));
		}
		System.out.println("(Se la colonna MC(tau~0) combacia con l'intrinseco qui, la condizione terminale e' confermata:");
		System.out.println(" la differenza che vedi rispetto a \"PINN a t=3.00\" e' allora un errore residuo della rete, non un mio errore di interpretazione.)");
	}

	// ------------------------------------------------------------------
	// TEST 1: confronto cella per cella su tutta la griglia della Price Table.
	// Nota: con 10x11=110 celle e DEFAULT_NUM_SIMULATIONS=200000 questo puo'
	// richiedere qualche minuto. Riduci la griglia (es. solo t=0.00/1.50/3.00)
	// per un primo giro rapido, se vuoi.
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
	// TEST 2: ripeti la stessa cella con un numero crescente di simulazioni per
	// vedere quando il prezzo MC si stabilizza (quanto rumore c'e' nella "verita'"
	// con cui stai confrontando il PINN).
	// ------------------------------------------------------------------
	private static void test2_ConvergenceCheck(double s, double t0) {
		double tau = T - t0;
		int[] simCounts = { 5_000, 20_000, 50_000, 100_000, 300_000, 600_000 };
		for (int n : simCounts) {
			double price = americanPutLSM(s, tau, n, scaledSteps(tau));
			System.out.printf(Locale.ROOT, "numSim=%7d -> prezzo=%.5f%n", n, price);
		}
		System.out.printf(Locale.ROOT, "PINN su questa cella (S=%.2f, t=%.2f): %.4f%n", s, t0, pinnPriceAt(s, t0));
		System.out.println("(Se il prezzo MC continua a cambiare parecchio anche a 300k-600k simulazioni, allarga numSim ulteriormente");
		System.out.println(" prima di fidarti del confronto su questa cella: e' la piu' difficile, proprio sul confine di esercizio.)");
	}

	// ------------------------------------------------------------------
	// TEST 3: quantifica il premio di esercizio anticipato (americana - europea)
	// nella coda OTM profonda, dove la Price Table del PINN mostra quel plateau
	// quasi costante ~0.05 che ci preoccupa. Se qui il Monte Carlo mostra invece
	// valori che DECADONO chiaramente con S e con t, il plateau del PINN e'
	// probabilmente un sintomo di sotto-allenamento (o del vincolo sigmoid che
	// satura), non la realta' economica.
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
						"S=%.2f t=%.2f | Americana(MC)=%.5f | Europea(chiusa)=%.5f | premio=%.5f | PINN=%.4f%n",
						s, t0, american, european, american - european, pinnPriceAt(s, t0));
			}
		}
	}

	// ------------------------------------------------------------------
	// Helpers
	// ------------------------------------------------------------------

	/** Prezza una put americana con Longstaff-Schwartz, spot S0 e tempo residuo tau. */
	private static double americanPutLSM(double s0, double tau, int numSimulations, int numTimeSteps) {
		ModelToUse model = new BlackAndScholesModel(new double[] { s0 }, new double[] { R }, tau, numSimulations,
				new double[] { SIGMA }, numTimeSteps, new double[][] { { 1.0 } });
		DoubleUnaryOperator payoff = s -> Math.max(K - s, 0.0);
		IOption put = new BermudaOption(tau, payoff, 1);
		LongstaffSchwartzPricer pricer = new LongstaffSchwartzPricer(numSimulations, numTimeSteps, model, put);
		return pricer.getFinalPrice();
	}

	/** Formula chiusa di Black-Scholes per una put europea (q=0). Nessun rumore MC. */
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

	/** Scala il numero di passi temporali proporzionalmente a tau, con un minimo di 5. */
	private static int scaledSteps(double tau) {
		int steps = (int) Math.round(DEFAULT_NUM_TIME_STEPS * (tau / T));
		return Math.max(steps, 5);
	}

	/** Legge il valore della Price Table del PINN alla cella (S,t) piu' vicina nella griglia. */
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