package it.univr.option;

import java.util.function.DoubleUnaryOperator;
import it.univr.model.MarketEnvironmentForModel;

/**
 * Represents a Bermudan (or simplified American-style) option contract,
 * evaluating path-dependent early exercise decisions by comparing immediate
 * intrinsic values against estimated local continuation values along simulated
 * trajectories.
 *
 * [FIX — 2026-10-08] getPayoff(double[][] path) (usato SOLO dal Naive Pricer,
 * MonteCarloPricer — LSM/LongstaffSchwartzPricer non chiama mai questo
 * overload) stimava la continuazione mediando i payoff FUTURI realizzati
 * sullo stesso path: questo e' "look-ahead bias" (usa informazione che al
 * tempo i non sarebbe disponibile nella realta'), e gonfiava il prezzo del
 * ~25% rispetto al valore vero (confermato confrontando con LSM e con un
 * albero binomiale CRR calcolato indipendentemente). Sostituita con la
 * regola naive classica "esercita appena l'opzione e' in-the-money", che usa
 * solo lo stato corrente del path (nessun guardare al futuro): resta una
 * politica subottimale (da cui il nome "naive"), ma e' una politica di
 * arresto non-anticipativa valida, quindi per costruzione fornisce un lower
 * bound legittimo del prezzo americano.
 *
 * [FIX 2 — 2026-10-08] Il ciclo sotto partiva da i = 0. Ma i = 0 e' l'istante
 * di valutazione (t = 0): il prezzo path[0][0] e' SEMPRE S0, lo stesso valore
 * deterministico identico su TUTTI i path (non e' ancora stato simulato
 * nulla). Non e' quindi una vera opportunita' di esercizio stocastica: e' solo
 * il dato di partenza. Controllarlo come se fosse un istante di decisione
 * faceva "esercitare a t=0" ogni volta che S0 era in-the-money (cioe' ogni
 * volta che S0 < K per una put), collassando il prezzo al valore puramente
 * deterministico (K - S0) scontato — identico per costruzione a QUALSIASI
 * livello di volatilita', il che e' esattamente il sintomo osservato (prezzi
 * NaiveMC identici al variare di sigma). Il ciclo ora parte dalla prima vera
 * data di esercizio (i = timetoExcersize), non da t = 0.
 */
public class BermudaOption implements IOption {

	// Option maturity in years
	private final double maturity;
	// Functional interface defining the option payoff structure (e.g., put or call)
	private final DoubleUnaryOperator payoff;
	// Frequency or interval step count for early exercise opportunities
	private final int timetoExcersize;

	/**
	 * Constructs a Bermudan option with specified maturity, payoff, and exercise
	 * frequency.
	 *
	 * @param maturity        Option maturity in years.
	 * @param payoff          Functional operator representing the option payoff.
	 * @param timetoExcersize Interval steps for discrete early exercise points.
	 */
	public BermudaOption(double maturity, DoubleUnaryOperator payoff, int timetoExcersize) {
		this.maturity = maturity;
		this.payoff = payoff;
		this.timetoExcersize = timetoExcersize;
	}

	/**
	 * Evaluates the option payoff over a simulated multi-step path matrix
	 * (Bermudan style), usando la regola naive non-anticipativa "esercita
	 * appena in-the-money" (vedi note [FIX] in cima alla classe).
	 *
	 * @param path The 2D matrix representing a single simulation path:
	 *             [timeSteps][assets].
	 * @return The discounted cash flow resulting from early exercise or
	 *         terminal maturity.
	 */
	@Override
	public double getPayoff(double[][] path) {

		int numberOfSteps = path.length - 1;
		// Retrieves the weighted risk-free rate from the global market environment
		double riskFreeRateMedium = MarketEnvironmentForModel.getWeightedRate();

		// [FIX 2026-10-08] Parte da timetoExcersize, non da 0: i = 0 e' t = 0
		// (S0 deterministico, uguale su tutti i path), non una vera data di
		// esercizio. Vedi nota [FIX 2] in cima alla classe.
		for (int i = timetoExcersize; i < path.length; i += timetoExcersize) {
			double timeYears = ((double) i / numberOfSteps) * maturity;
			// Immediate intrinsic exercise value at the current time step for the primary
			// asset (path[i][0]) — nessuna informazione sul futuro viene usata qui.
			double valueAtTime = payoff.applyAsDouble(path[i][0]);

			// Regola naive non-anticipativa: esercita appena il payoff
			// immediato e' positivo (in-the-money), invece di confrontarlo con una
			// stima di continuazione calcolata guardando il futuro realizzato.
			if (valueAtTime > 0.0) {
				// Return immediate exercise payoff discounted from the exercise time back to t
				// = 0
				return valueAtTime * Math.exp(-riskFreeRateMedium * timeYears);
			}
		}

		// Fallback: if never in-the-money at any exercise date, return terminal
		// payoff at maturity discounted to t = 0
		return payoff.applyAsDouble(path[numberOfSteps][0]) * Math.exp(-riskFreeRateMedium * maturity);
	}

	/**
	 * Returns the option maturity in years.
	 *
	 * @return The maturity time horizon.
	 */
	@Override
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Evaluates the payoff for a single flat path array (European-style fallback).
	 *
	 * @param path Array of asset prices along a simulation path.
	 * @return The option payoff evaluated at the final path element.
	 */
	@Override
	public double getPayoff(double[] path) {
		// Returns the payoff evaluated strictly at the terminal element of the path
		// array
		return payoff.applyAsDouble(path[path.length - 1]);
	}
}