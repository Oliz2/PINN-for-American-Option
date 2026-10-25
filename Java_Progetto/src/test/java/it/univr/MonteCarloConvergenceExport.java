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
 * <strong>Monte Carlo Convergence Export (per grafici Python).</strong>
 * </p>
 * <p>
 * Classe standalone, nuova, che NON modifica nessuna classe di calcolo
 * esistente. Per un numero crescente di simulazioni, ricalcola il prezzo con
 * entrambi i metodi Monte Carlo (Naive e Longstaff-Schwartz) e scrive i
 * risultati su CSV, per mostrare che entrambi convergono verso lo stesso
 * valore al crescere di N.
 * </p>
 * <p>
 * [AGGIUNTA 2026-10-08] Aggiunto un terzo metodo, "BinomialTree": usa le tue
 * classi {@code BinomialModel}/{@code BermudaBinomialOption} (gia' presenti
 * nel progetto, non toccate qui) per calcolare il prezzo con l'albero
 * binomiale CRR. A differenza di NaiveMC e LSM non ha rumore statistico
 * (nessuna simulazione casuale), quindi funge da riferimento deterministico:
 * il suo prezzo si stabilizza gia' a poche centinaia/migliaia di passi — vedi
 * {@code BinomialTreeTest.java}, dove questo e' gia' stato verificato (4.4866
 * a 5000 passi, in linea con 4.486 di letteratura).
 * </p>
 * <p>
 * [IMPORTANTE] A differenza di NaiveMC/LSM, l'albero binomiale di questo
 * progetto memorizza matrici piene N x N (memoria O(N^2), non O(N) come un
 * albero con storage "a rullo"): a N=10000 servono gia' alcuni GB di heap, e
 * a N=100000+ (i valori piu' grandi usati per NaiveMC/LSM) servirebbero
 * terabyte di RAM, semplicemente impossibile. Per questo il metodo
 * BinomialTree NON riusa l'array {@code simulationCounts} di NaiveMC/LSM, ma
 * un proprio array {@code binomialSteps}, piu' fitto nella zona bassa (dove
 * la curva si appiattisce) e con un tetto massimo di 5000 passi. Nel
 * CSV/grafico la curva BinomialTree avra' quindi piu' punti ravvicinati ma
 * si fermera' prima delle altre due sull'asse x — cosa naturale: e' un
 * metodo deterministico, non ha bisogno di tanti "passi" quanti ne servono a
 * Monte Carlo per ridurre il rumore statistico.
 * </p>
 * <p>
 * [FIX 2026-10-08] Aggiunto {@code Locale.US} alle printf: senza, su una JVM
 * con locale italiano i numeri vengono scritti con la virgola come separatore
 * decimale, che corrompe il CSV (si somma alla virgola che separa le
 * colonne). Vedi lo stesso fix gia' applicato in
 * {@code MonteCarloPathAndStoppingExport.java}.
 * </p>
 * <p>
 * Parametro di default: stesso caso benchmark usato in {@code MonteCarloTest}
 * (S0=36, K=40, r=0.06, sigma=0.20, T=1) — e' uno dei casi di Longstaff &amp;
 * Schwartz (2001), Tabella 1, per cui esiste un prezzo di riferimento
 * pubblicato calcolato con un metodo a differenze finite (vedi commento nello
 * script Python per il valore esatto e la fonte).
 * </p>
 *
 * @author Alberto Oliva Medin
 */
public class MonteCarloConvergenceExport {

	public static void main(String[] args) throws IOException {

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
		int numberOfTimes = (int) (maturity / timeStep);
		int timeToExercise = 1;
		double[][] rho = { { 1.0 } };

		// Valori di N (numero di simulazioni) su cui misurare la convergenza di
		// NaiveMC e LSM
		int[] simulationCounts = { 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000 };

		// [AGGIUNTA 2026-10-08] Valori di PASSI (non simulazioni) su cui misurare la
		// convergenza dell'albero binomiale: array separato e piu' fitto, dato che
		// l'albero converge con pochissimi passi rispetto a Monte Carlo (vedi nota
		// [IMPORTANTE] nel Javadoc sopra sul costo O(passi^2) in memoria, che impone
		// comunque un tetto massimo qui a 5000).
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
				System.out.printf("NaiveMC      N=%7d -> prezzo=%.4f%n", n, naivePrice);

				BlackAndScholesModel modelForLsm = new BlackAndScholesModel(initialValue, riskFreeRate, maturity, n,
						sigma, numberOfTimes, rho);
				BermudaOption optionForLsm = new BermudaOption(maturity, put, timeToExercise);
				LongstaffSchwartzPricer lsmPricer = new LongstaffSchwartzPricer(n, numberOfTimes, modelForLsm,
						optionForLsm);
				double lsmPrice = lsmPricer.getFinalPrice();
				writer.printf(Locale.US, "LSM,%d,%.6f%n", n, lsmPrice);
				System.out.printf("LSM          N=%7d -> prezzo=%.4f%n", n, lsmPrice);
			}

			// [AGGIUNTA 2026-10-08] Terzo metodo, deterministico: albero binomiale CRR.
			// Loop separato sul proprio array di passi (binomialSteps), piu' fitto nella
			// zona bassa, dato che l'albero converge con pochi passi (vedi Javadoc). I
			// valori di "numberOfSimulation" scritti sul CSV per questo metodo sono in
			// realta' numero di PASSI dell'albero: condividono comunque l'asse x (scala
			// log) del grafico Python, quindi non serve nessuna modifica li'.
			for (int steps : binomialSteps) {
				try {
					BinomialModel binomialModel = new BinomialModel(initialValue[0], riskFreeRate[0], sigma[0],
							steps, maturity);
					BermudaBinomialOption binomialOption = new BermudaBinomialOption(put, steps, steps,
							binomialModel);
					double binomialPrice = binomialOption.getOptionValues()[0][0];
					writer.printf(Locale.US, "BinomialTree,%d,%.6f%n", steps, binomialPrice);
					System.out.printf("BinomialTree steps=%7d -> prezzo=%.4f%n", steps, binomialPrice);
				} catch (Exception e) {
					// BermudaBinomialOption dichiara "throws Exception" nel costruttore (per il
					// controllo numberOfStepsOption <= numberOfStepsStock, qui sempre vero
					// visto che li passiamo uguali): non dovrebbe mai scattare, ma lo logghiamo
					// invece di interrompere l'intero export se succede.
					System.out.println("BinomialTree steps=" + steps + " -> errore: " + e.getMessage());
				}
			}
		}

		System.out.println("Scritto: " + outFile.getAbsolutePath());
	}
}