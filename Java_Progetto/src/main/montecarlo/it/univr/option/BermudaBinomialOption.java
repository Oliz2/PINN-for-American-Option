package it.univr.option;

import java.util.function.DoubleUnaryOperator;
import it.univr.BinomialModel.*;
import it.univr.utils.*;

public class BermudaBinomialOption implements BinomialOptions {
	
	// time discretization
	private int numberOfStepsOption;
	
	// option parameters
	private double[][] snellEnvelope;
	private double[][] excerciseValue;
	private DoubleUnaryOperator transformFunction;
	
	// model used to simulate the underlying asset
	BinomialModel model;
	
	
	
	public BermudaBinomialOption (DoubleUnaryOperator transformFunction, int numberOfStepsOption,
			int numberOfStepsStock, BinomialModel model) throws Exception {
		if (numberOfStepsOption > numberOfStepsStock) {
			throw new IllegalArgumentException ("The time discretization is less than the discretization of the option");
		}
		this.transformFunction=transformFunction;
		this.numberOfStepsOption=numberOfStepsOption;
		this.model=model;
		}
	
	
	// This method computes the American option values using the Snell envelope and backward induction
	private void computeValues () {
		
		/*
		 * Initialization of matrices:
		 * 
		 * excerciseValue stores the immediate exercise value of the option
		 * 
		 * snellEnvelope stores the option value at each time step
		 */
		excerciseValue = new double [numberOfStepsOption][numberOfStepsOption];
		snellEnvelope = new double [numberOfStepsOption][numberOfStepsOption];
		
		// computation of the payoff
		snellEnvelope[numberOfStepsOption-1] = model.getTransformedValuesAtGivenTimeIndex(numberOfStepsOption-1, transformFunction);
		excerciseValue[numberOfStepsOption-1] = model.getTransformedValuesAtGivenTimeIndex(numberOfStepsOption-1, transformFunction);
		
		/*
		 * Backward loop to compute option values from maturity to time zero.
		 * Index t represents the current time step 
		 */
		for (int t = numberOfStepsOption-2; t >= 0; t--) {
			
			// compute the immediate exercise value of the option
			excerciseValue[t] = model.getTransformedValuesAtGivenTimeIndex(t, transformFunction);
			
			// computation of the value that would be obtained by holding the option
			double[] continuationValues = model.getConditionalExpectation(snellEnvelope[t+1], t);
			
			// apply the Snell envelope formula to determine the option value at this step
			snellEnvelope[t] = UsefulMethodsForArrays.getMaxValuesBetweenTwoArrays(excerciseValue[t], continuationValues);
		}
	}

	
	@Override
	public double[][] getOptionValues () {
		if (snellEnvelope == null) {
	        this.computeValues();
	    }
		return snellEnvelope;
	}

}