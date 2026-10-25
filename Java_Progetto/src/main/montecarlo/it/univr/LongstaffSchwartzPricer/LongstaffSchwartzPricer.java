package it.univr.LongstaffSchwartzPricer;

import it.univr.model.ModelToUse;
import it.univr.option.IOption;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import it.univr.model.MarketEnvironmentForModel;

/**
 * Implements the Least-Squares Monte Carlo (LSM) algorithm introduced by
 * Longstaff and Schwartz (2001) for pricing American and Bermudan style options
 * through backward induction and regression.
 *
 * [FIX — aggiunto tracking dello stopping time per path (array exerciseTime),
 * usato solo per i grafici (regione di esercizio anticipato), non modifica in
 * alcun modo il calcolo del prezzo: e' una semplice "fotografia" di quale
 * istante temporale, per ciascun path, risulta l'ultimo aggiornamento
 * dell'algoritmo di induzione backward (cioe' il tempo di esercizio ottimale
 * sotto la policy stimata da LSM). Se un path non viene mai esercitato
 * anticipatamente, resta al valore di default numberOfTimes (= esercitato a
 * scadenza), 2026-10-08]
 */
public class LongstaffSchwartzPricer {

	// Model generating the underlying asset paths
	private final ModelToUse model;
	// Option contract definition providing payoffs
	private final IOption option;
	// Total number of simulated trajectories
	private final int numberOfSimulation;
	// Number of discrete time steps along the time horizon
	private final int numberOfTimes;

	// Cached final computed option price
	private double finalPrice;
	// Flag to ensure lazy evaluation and avoid redundant computations
	private boolean isCalculated = false;

	// [FIX 2026-10-08] Tempo di esercizio (indice temporale) per ciascun path,
	// solo per scopi di visualizzazione — vedi Javadoc della classe.
	private int[] exerciseTime;

	/**
	 * Creates a Longstaff-Schwartz pricer for American/Bermudan options.
	 *
	 * @param numberOfSimulation The number of Monte Carlo paths.
	 * @param numberOfTimes      The number of discrete time steps.
	 * @param model              The stochastic model providing asset paths.
	 * @param option             The option contract specification.
	 */
	public LongstaffSchwartzPricer(int numberOfSimulation, int numberOfTimes, ModelToUse model, IOption option) {
		this.numberOfSimulation = numberOfSimulation;
		this.numberOfTimes = numberOfTimes;
		this.model = model;
		this.option = option;
	}

	/**
	 * Executes the regression-based backward induction algorithm to determine
	 * optimal early exercise boundaries and compute the fair option price.
	 */
	public void calculateOptionPrice() {
		// Retrieve the 3D asset paths matrix: [simulation][time][asset]
		double[][][] assetPaths = model.getAssetPaths();
		double dt = model.getDt();

		// Retrieve the market discount parameters
		double riskFreeRateMedium = MarketEnvironmentForModel.getWeightedRate();
		double discountFactor = Math.exp(-riskFreeRateMedium * dt);

		// Track the optimal cash flows along each simulation path
		double[] cashFlows = new double[numberOfSimulation];

		// [FIX 2026-10-08] Inizializza il tracking dello stopping time: di default
		// ogni path e' considerato "esercitato a scadenza" (numberOfTimes) finche'
		// l'induzione backward non trova un esercizio anticipato migliore.
		exerciseTime = new int[numberOfSimulation];
		Arrays.fill(exerciseTime, numberOfTimes);

		// Step 1: Initialize cash flows at maturity with the European payoff
		for (int i = 0; i < numberOfSimulation; i++) {
			cashFlows[i] = option.getPayoff(assetPaths[i][numberOfTimes]);
		}

		// Step 2: Perform backward induction from the penultimate step down to t = 1
		for (int time = numberOfTimes - 1; time > 0; time--) {

			// Collect indices of paths that are In-The-Money (ITM) at the current time step
			List<Integer> itmIndices = new ArrayList<>();
			for (int sim = 0; sim < numberOfSimulation; sim++) {
				double s = assetPaths[sim][time][0];

				if (option.getPayoff(new double[] { s }) > 0.0) {
					itmIndices.add(sim);
				}
			}

			// Discount existing cash flows forward by one time increment
			for (int sim = 0; sim < numberOfSimulation; sim++) {
				cashFlows[sim] *= discountFactor;
			}

			// Execute OLS regression only if sufficient degrees of freedom are available
			// (at least 4 points)
			if (itmIndices.size() > 3) {
				double[] regY = new double[itmIndices.size()];
				double[][] regX = new double[itmIndices.size()][2];

				// Populate regression regressors (S and S^2) and regressands (cash flows) for
				// ITM paths
				for (int j = 0; j < itmIndices.size(); j++) {
					int simIndex = itmIndices.get(j);
					double s = assetPaths[simIndex][time][0];

					regY[j] = cashFlows[simIndex];
					regX[j][0] = s;
					regX[j][1] = s * s;
				}

				// Fit Ordinary Least Squares (OLS) model to estimate the continuation value
				// function
				OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
				regression.newSampleData(regY, regX);
				double[] beta = regression.estimateRegressionParameters();

				// Compare immediate exercise value against estimated continuation value for ITM
				// paths
				for (int simIndex : itmIndices) {
					double s = assetPaths[simIndex][time][0];
					double immediateExercise = option.getPayoff(new double[] { s });

					double continuationValue = beta[0] + beta[1] * s + beta[2] * (s * s);

					// Apply optimal stopping rule: exercise early if immediate payoff exceeds
					// continuation value
					if (immediateExercise > continuationValue) {
						cashFlows[simIndex] = immediateExercise;
						// [FIX 2026-10-08] Registra l'istante di esercizio anticipato per questo path.
						exerciseTime[simIndex] = time;
					}
				}
			}
		}

		// Step 3: Aggregate and discount all optimal path cash flows back to t = 0
		double totalValue = 0.0;
		for (int i = 0; i < numberOfSimulation; i++) {
			totalValue += cashFlows[i] * discountFactor;
		}

		this.finalPrice = totalValue / numberOfSimulation;
		this.isCalculated = true;
	}

	/**
	 * Returns the computed option price using lazy evaluation.
	 *
	 * @return The fair price of the option.
	 */
	public double getFinalPrice() {
		if (!isCalculated) {
			calculateOptionPrice();
		}
		return finalPrice;
	}

	/**
	 * [FIX 2026-10-08] Restituisce, per ciascun path, l'indice temporale al quale
	 * l'algoritmo LSM ha deciso l'esercizio anticipato (oppure numberOfTimes se il
	 * path non viene mai esercitato prima della scadenza). Usato solo per i
	 * grafici della "regione di stopping time" — non influenza il prezzo.
	 *
	 * @return Array di lunghezza numberOfSimulation con l'indice temporale di
	 *         esercizio per ciascun path.
	 */
	public int[] getExerciseTimes() {
		if (!isCalculated) {
			calculateOptionPrice();
		}
		return exerciseTime;
	}
}