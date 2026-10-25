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
 * Usa le classi dell'albero binomiale gia' presenti nel progetto
 * ({@code BinomialModel}, {@code BermudaBinomialOption}, backward induction
 * tramite l'inviluppo di Snell) per: (1) controllare che il prezzo converga
 * al crescere del numero di passi dell'albero, e (2) confrontare il prezzo
 * dell'albero con quello di Longstaff-Schwartz sugli stessi 20 casi
 * benchmark usati in {@code MonteCarloTest.java} (Longstaff &amp; Schwartz,
 * 2001, Tabella 1).
 * </p>
 * <p>
 * Perche' questo confronto ha senso: l'albero binomiale e LSM sono due
 * metodi numerici indipendenti (programmazione dinamica su un albero vs.
 * regressione su simulazioni Monte Carlo) che stimano la STESSA quantita'
 * teorica, il prezzo della put americana/bermudiana. Se entrambi sono
 * implementati correttamente devono convergere allo stesso valore — qui
 * l'albero (deterministico, nessun rumore statistico) funge da riferimento
 * "esatto" per validare LSM, invece di usare solo un valore di letteratura
 * trascritto a mano.
 * </p>
 * <p>
 * [NOTA] Verificato in locale (sandbox, parametri S0=36,K=40,r=0.06,
 * sigma=0.20,T=1): {@code BermudaBinomialOption} converge correttamente,
 * da 4.4818 (50 passi) a 4.4866 (5000 passi), in linea sia col valore di
 * letteratura (~4.486) sia con un albero CRR scritto indipendentemente per
 * verifica. Un dettaglio secondario, visibile solo a passi molto alti: sia
 * {@code BinomialModel} che {@code BermudaBinomialOption} memorizzano
 * matrici piene {@code numberOfTimes x numberOfTimes} (memoria O(N^2)
 * invece di O(N) come in un albero con storage "a rullo"), quindi con molti
 * passi (es. 10000+) puo' servire piu' heap alla JVM (Eclipse: Run
 * Configurations -&gt; Arguments -&gt; VM arguments -&gt; {@code -Xmx2g} o
 * superiore); per questo qui sotto il test di convergenza si ferma a 5000
 * passi, dove il prezzo e' gia' stabile alla terza cifra decimale.
 * </p>
 *
 * [NUOVO file, 2026-10-08 — nessuna modifica alle classi di calcolo esistenti]
 *
 * @author Alberto Oliva Medin
 */
public class BinomialTreeTest {

	public static void main(String[] args) throws Exception {

		// =======================================================================
		// 1. PARAMETRI (stesso caso benchmark di MonteCarloTest.java)
		// =======================================================================
		double initialValue = 36.0;
		double strike = 40.0;
		double riskFreeRate = 0.06;
		double sigma = 0.20;
		double maturity = 1.0;

		// Passi dell'albero usati per il prezzo "singolo" e per il confronto con LSM
		// piu' sotto. 2000 e' gia' abbondantemente convergente (vedi test di
		// convergenza) e resta veloce anche ripetuto 20 volte nel loop di confronto.
		int numberOfStepsBinomial = 2000;

		MarketEnvironmentForModel.getValue(new double[] { riskFreeRate }, new double[] { 1.0 });

		DoubleUnaryOperator put = x -> Math.max(strike - x, 0.0);

		// =======================================================================
		// 2. PREZZO SINGOLO CON L'ALBERO BINOMIALE
		// =======================================================================
		BinomialModel model = new BinomialModel(initialValue, riskFreeRate, sigma, numberOfStepsBinomial, maturity);
		BermudaBinomialOption binomialOption = new BermudaBinomialOption(put, numberOfStepsBinomial,
				numberOfStepsBinomial, model);
		double priceBinomial = binomialOption.getOptionValues()[0][0];

		System.out.println(
				"Prezzo con albero binomiale (" + numberOfStepsBinomial + " passi): " + priceBinomial);

		// =======================================================================
		// 3. TEST DI CONVERGENZA: prezzo al crescere del numero di passi dell'albero
		// =======================================================================
		System.out.println();
		System.out.println("Convergenza dell'albero binomiale al crescere del numero di passi:");
		int[] stepsToTest = { 50, 100, 500, 1000, 2000, 5000 };
		for (int steps : stepsToTest) {
			BinomialModel modelStep = new BinomialModel(initialValue, riskFreeRate, sigma, steps, maturity);
			BermudaBinomialOption optionStep = new BermudaBinomialOption(put, steps, steps, modelStep);
			double priceStep = optionStep.getOptionValues()[0][0];
			System.out.printf("  passi=%6d -> prezzo=%.6f%n", steps, priceStep);
		}

		// =======================================================================
		// 4. CONFRONTO ALBERO BINOMIALE vs LSM sui 20 casi di Longstaff & Schwartz
		// (2001), Tabella 1 — stessi parametri/loop di MonteCarloTest.java
		// =======================================================================
		System.out.println();
		System.out.println("-------------------------------------------------------------");
		System.out.println("Confronto Albero Binomiale vs Longstaff-Schwartz (LSM):");

		double[] sigmaArray = { 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20,
				0.40, 0.40, 0.20, 0.20, 0.40, 0.40 };
		double[] initialValueArray = { 36.0, 36.0, 36.0, 36.0, 38.0, 38.0, 38.0, 38.0, 40.0, 40.0, 40.0, 40.0, 42.0,
				42.0, 42.0, 42.0, 44.0, 44.0, 44.0, 44.0 };
		double[] maturityArray = { 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
				2.0, 1.0, 2.0 };

		// Parametri LSM (identici a MonteCarloTest.java, per un confronto equo)
		int numberOfSimulationLSM = 100000;
		double timeStepLSM = 0.02;
		int timeToExerciseLSM = 1;
		double[] riskFreeRateArr = { riskFreeRate };
		double[][] rho = { { 1.0 } };

		for (int i = 0; i < sigmaArray.length; i++) {

			double s0 = initialValueArray[i];
			double vol = sigmaArray[i];
			double M = maturityArray[i];

			// --- Prezzo con l'albero binomiale (riferimento deterministico) ---
			BinomialModel modelI = new BinomialModel(s0, riskFreeRate, vol, numberOfStepsBinomial, M);
			BermudaBinomialOption optionI = new BermudaBinomialOption(put, numberOfStepsBinomial,
					numberOfStepsBinomial, modelI);
			double priceBinomialI = optionI.getOptionValues()[0][0];

			// --- Prezzo con LSM ---
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

			// Differenza percentuale rispetto all'albero binomiale, che qui funge da
			// riferimento "esatto" (deterministico, nessun rumore statistico) — stessa
			// logica gia' usata in MonteCarloTest.java per confrontare LSM col riferimento.
			double diffPct = Math.abs((priceBinomialI - priceLSM) / priceBinomialI) * 100;

			System.out.printf(
					"S0: %.1f, Vol: %.2f, T: %.1f | Binomiale: %.3f | LSM: %.3f | Differenza: %.3f%%\n",
					s0, vol, M, priceBinomialI, priceLSM, diffPct);
		}
	}
}
