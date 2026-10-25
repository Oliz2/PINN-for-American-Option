package it.univr.monteCarloproducts;

import it.univr.model.*;
import it.univr.option.IOption;

/**
 * Class responsible for pricing standard European-style options using direct
 * forward Monte Carlo simulation by averaging discounted terminal payoffs
 * across all simulated paths.
 *
 * [FIX — 2026-10-08] getValue(...) scontava il risultato DUE VOLTE. Il payoff
 * restituito da option.getPayoff(double[][] path) — per come e' implementato
 * in BermudaOption — e' GIA' attualizzato a t = 0 usando il vero istante di
 * esercizio (riga "valueAtTime * Math.exp(-riskFreeRateMedium * timeYears)"
 * dentro BermudaOption.getPayoff). Questo metodo prendeva quel valore GIA'
 * scontato e lo moltiplicava di nuovo per Math.exp(-riskFreeMedium *
 * maturity), scontandolo una seconda volta fino a un tempo (maturity) spesso
 * diverso dal vero istante di esercizio. Bug preesistente, non introdotto
 * dalle modifiche a BermudaOption: era presente anche con la vecchia regola
 * "look-ahead" (si combinava con quel bias, mascherandone parzialmente
 * l'effetto). Verificato numericamente sui risultati riportati: con la
 * regola naive attuale un'opzione in-the-money esercitata "a t=0" restituiva
 * da getPayoff un valore pari a (K - S0) (nessun sconto, perche' timeYears
 * era allora 0 — vedi fix in BermudaOption), ma il prezzo finale stampato era
 * sistematicamente piu' basso in modo esatto di un fattore exp(-r*maturity)
 * (es. per T=1: 4 * exp(-0.06*1) = 3.767; per T=2: 4 * exp(-0.06*2) = 3.548 —
 * entrambi corrispondenti esattamente ai valori osservati). Rimosso il
 * secondo sconto: il valore scontato va preso cosi' com'e' da getPayoff,
 * mediato sulle simulazioni.
 */
public class MonteCarloPricer {

	// Cached fair price value of the option
	private double priceValue;

	/**
	 * Default constructor for the Monte Carlo pricer.
	 */
	public MonteCarloPricer() {
		// Default initialization
	}

	/**
	 * Computes the fair price of a standard option using direct forward simulation
	 * of asset paths. Each path's payoff, as returned by option.getPayoff(...),
	 * is already discounted back to t = 0 at its own (possibly early) exercise
	 * time: this method simply averages those already-discounted payoffs across
	 * simulations — it must NOT discount again.
	 *
	 * @param option The option contract defining the payoff structure.
	 * @param model  The stochastic model providing the simulated asset paths.
	 * @return The expected standard Monte Carlo option price.
	 */
	public double getValue(IOption option, ModelToUse model) {

		// Retrieve the 3D asset paths matrix from the stochastic model
		double[][][] path = model.getAssetPaths();

		int numberOfSimulation = path.length;

		double totalPayoff = 0.0;

		// Evaluate the (already time-0-discounted) payoff for each independent
		// simulation trajectory
		for (int i = 0; i < numberOfSimulation; i++) {
			double finalAssetPrice = option.getPayoff(path[i]);
			totalPayoff += finalAssetPrice;
		}

		// [FIX 2026-10-08] Nessun secondo sconto qui: option.getPayoff(...) ha
		// gia' scontato ciascun payoff a t = 0 usando il proprio istante di
		// esercizio. Si fa solo la media sulle simulazioni (legge dei grandi
		// numeri / stima Monte Carlo del valore atteso scontato).
		this.priceValue = totalPayoff / numberOfSimulation;

		return priceValue;
	}

	/**
	 * Retrieves the cached option price value.
	 *
	 * @return The computed option price.
	 */
	public double getPriceValue() {
		return priceValue;
	}
}