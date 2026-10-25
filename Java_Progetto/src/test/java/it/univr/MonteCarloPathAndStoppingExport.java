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
 * <strong>Monte Carlo Path &amp; Early-Exercise Export (per grafici Python).</strong>
 * </p>
 * <p>
 * Classe standalone, nuova, che NON modifica nessuna classe di calcolo
 * esistente. Serve a esportare su CSV: (1) un sottoinsieme di traiettorie
 * simulate dal modello Black-Scholes, (2) l'istante di esercizio anticipato
 * stimato da LSM per ciascuna di quelle stesse traiettorie, e (3) la
 * frontiera di esercizio anticipato (exercise boundary) calcolata con
 * l'albero binomiale — in modo che il grafico delle traiettorie, i punti di
 * stopping time di LSM e la frontiera "teorica" dell'albero siano tutti
 * sovrapponibili sullo stesso grafico in Python.
 * </p>
 * <p>
 * [AGGIUNTA 2026-10-08] Export 3, {@code mc_exercise_boundary.csv}: usa le
 * tue classi {@code BinomialModel}/{@code BermudaBinomialOption} (gia'
 * presenti nel progetto, non toccate qui) per calcolare, ad ogni istante t,
 * il prezzo critico S*(t) sotto al quale conviene esercitare subito la put
 * (cioe' dove il payoff immediato supera il valore di continuazione). Questa
 * e' la frontiera di esercizio anticipato: una curva decrescente nel tempo
 * (tipicamente da un valore vicino a K a scadenza, verso un valore piu' basso
 * andando indietro nel tempo) che delimita la "regione di stopping" — i punti
 * rossi del grafico delle traiettorie (gli istanti in cui LSM decide di
 * esercitare) dovrebbero cadere vicino o sotto questa curva, non sopra.
 * </p>
 * <p>
 * Il calcolo usa solo l'API pubblica gia' presente su {@code BinomialModel}
 * ({@code getValuesAtGivenTimeIndex}, {@code getTransformedValuesAtGivenTimeIndex},
 * {@code getConditionalExpectation}) e su {@code BermudaBinomialOption}
 * ({@code getOptionValues}, cioe' l'inviluppo di Snell gia' calcolato):
 * per ogni t si ricostruisce il confronto valore-di-esercizio vs
 * valore-di-continuazione, sfruttando il fatto (teorema standard per una put
 * americana su un sottostante log-normale) che la regione di esercizio e' un
 * intervallo connesso S &lt;= S*(t): si cerca quindi il prezzo piu' alto, tra
 * i nodi dell'albero a quell'istante, per cui conviene ancora esercitare.
 * </p>
 * <p>
 * Parametri di default: stesso caso benchmark usato in {@code MonteCarloTest}
 * (S0=36, K=40, r=0.06, sigma=0.20, T=1, 50 date di esercizio) — cambiali
 * liberamente qui sotto se vuoi esportare un'altra configurazione.
 * </p>
 *
 * [NUOVO file, 2026-10-08 — nessuna modifica alle classi esistenti, usa solo
 * API pubbliche gia' presenti (BlackAndScholesModel.getAssetPaths(),
 * LongstaffSchwartzPricer.getExerciseTimes(), BinomialModel.*,
 * BermudaBinomialOption.getOptionValues())]
 *
 * @author Alberto Oliva Medin
 */
public class MonteCarloPathAndStoppingExport {

	public static void main(String[] args) throws Exception {

		// =======================================================================
		// PARAMETRI (stesso caso benchmark di MonteCarloTest — modificabile)
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

		// Quante traiettorie esportare per il grafico (non tutte le simulazioni: solo
		// un sottoinsieme leggibile)
		int pathsToExport = 40;

		// [AGGIUNTA 2026-10-08] Passi dell'albero binomiale usati SOLO per calcolare
		// la frontiera di esercizio (non serve un numero enorme: e' per il disegno
		// della curva, non per un prezzo ad altissima precisione — vedi la nota sul
		// costo O(passi^2) in memoria in MonteCarloConvergenceExport.java).
		int numberOfStepsBoundary = 500;

		MarketEnvironmentForModel.getValue(riskFreeRate, weights);

		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		BlackAndScholesModel model = new BlackAndScholesModel(initialValue, riskFreeRate, maturity,
				numberOfSimulation, sigma, numberOfTimes, rho);

		BermudaOption option = new BermudaOption(maturity, put, timeToExercise);

		LongstaffSchwartzPricer pricer = new LongstaffSchwartzPricer(numberOfSimulation, numberOfTimes, model,
				option);

		// Calcola il prezzo: questo genera (e mette in cache dentro "model") la
		// matrice dei path, e calcola gli exerciseTime — stessa identica
		// realizzazione casuale usata sotto per l'export.
		pricer.calculateOptionPrice();
		System.out.println("Prezzo LSM (caso esportato): " + pricer.getFinalPrice());

		double[][][] paths = model.getAssetPaths();
		int[] exerciseTimes = pricer.getExerciseTimes();

		double dt = maturity / numberOfTimes;

		File outDir = new File("plot_data");
		outDir.mkdirs();

		// -----------------------------------------------------------------------
		// Export 1: traiettorie simulate (per il grafico "simulazioni nel tempo")
		// -----------------------------------------------------------------------
		File pathsFile = new File(outDir, "mc_paths.csv");
		try (PrintWriter writer = new PrintWriter(new FileWriter(pathsFile))) {
			writer.println("simulation_id,time_step,time_years,price");
			for (int sim = 0; sim < pathsToExport; sim++) {
				for (int t = 0; t <= numberOfTimes; t++) {
					double timeYears = t * dt;
					double price = paths[sim][t][0];
					// [FIX 2026-10-08] Locale.US forza il punto come separatore decimale,
					// indipendentemente dalla lingua di Windows/JVM (altrimenti su un
					// sistema in italiano printf scrive la virgola e rompe il CSV).
					writer.printf(Locale.US, "%d,%d,%.6f,%.6f%n", sim, t, timeYears, price);
				}
			}
		}
		System.out.println("Scritto: " + pathsFile.getAbsolutePath());

		// -----------------------------------------------------------------------
		// Export 2: istante di esercizio anticipato per path (per il grafico della
		// regione di stopping time, da sovrapporre al grafico delle traiettorie)
		// -----------------------------------------------------------------------
		File exerciseFile = new File(outDir, "mc_exercise.csv");
		try (PrintWriter writer = new PrintWriter(new FileWriter(exerciseFile))) {
			writer.println("simulation_id,exercise_time_step,exercise_time_years,exercise_price,exercised_early");
			for (int sim = 0; sim < pathsToExport; sim++) {
				int et = exerciseTimes[sim];
				double timeYears = et * dt;
				double price = paths[sim][et][0];
				boolean exercisedEarly = et < numberOfTimes;
				// [FIX 2026-10-08] Locale.US, stesso motivo di sopra.
				writer.printf(Locale.US, "%d,%d,%.6f,%.6f,%b%n", sim, et, timeYears, price, exercisedEarly);
			}
		}
		System.out.println("Scritto: " + exerciseFile.getAbsolutePath());

		// -----------------------------------------------------------------------
		// Export 3: frontiera di esercizio anticipato dall'albero binomiale
		// -----------------------------------------------------------------------
		File boundaryFile = new File(outDir, "mc_exercise_boundary.csv");
		try (PrintWriter writer = new PrintWriter(new FileWriter(boundaryFile))) {
			writer.println("time_years,boundary_price");

			BinomialModel binomialModel = new BinomialModel(initialValue[0], riskFreeRate[0], sigma[0],
					numberOfStepsBoundary, maturity);
			BermudaBinomialOption binomialOption = new BermudaBinomialOption(put, numberOfStepsBoundary,
					numberOfStepsBoundary, binomialModel);
			// Forza il calcolo dell'inviluppo di Snell (altrimenti e' null finche' non lo
			// si chiede la prima volta)
			double[][] snellEnvelope = binomialOption.getOptionValues();

			double dtBoundary = maturity / numberOfStepsBoundary;

			for (int t = 0; t < numberOfStepsBoundary - 1; t++) {

				double timeYears = t * dtBoundary;

				// Prezzi possibili del sottostante a questo istante (indice 0 = prezzo piu'
				// alto, cioe' tutti rialzi; indice t = prezzo piu' basso, cioe' tutti ribassi)
				double[] assetValues = binomialModel.getValuesAtGivenTimeIndex(t);
				// Valore di esercizio immediato ad ogni nodo
				double[] exerciseValues = binomialModel.getTransformedValuesAtGivenTimeIndex(t, put);
				// Valore di continuazione (scontato di un passo) ad ogni nodo, ricavato dai
				// valori dell'opzione gia' calcolati al passo successivo (snellEnvelope[t+1])
				double[] continuationValues = binomialModel.getConditionalExpectation(snellEnvelope[t + 1], t);

				// La regione di esercizio di una put americana e' un intervallo connesso
				// S <= S*(t) (risultato standard): scansioniamo dal prezzo piu' basso (indice
				// piu' alto) verso l'alto e prendiamo il prezzo piu' alto per cui conviene
				// ancora esercitare subito. Se non conviene mai esercitare a questo istante
				// (lontano dalla scadenza, opzione OTM), la frontiera non esiste: scriviamo
				// NaN, che pandas/matplotlib ignorano automaticamente nel disegno della linea.
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
			// A scadenza la frontiera della put e' per definizione lo strike: si esercita
			// se e solo se S(T) < K.
			writer.printf(Locale.US, "%.6f,%.6f%n", maturity, strike);
		}
		System.out.println("Scritto: " + boundaryFile.getAbsolutePath());
	}
}