package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

/**
 * Created by Виктория on 12.12.2017.
 * L1
 */
public class LaplacianLoss implements LossFunction, LinearMinimizible {
    @Override
    public Vector computeGradient(Vector labels, Vector predictions) {
        Vector gradient = labels.minus(predictions);
        for (int i = 0; i < labels.size(); i++) {
            double value = gradient.get(i);
            gradient.set(i, value * Math.signum(value));
        }
        return gradient;
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        Vector residual = labels.minus(predictions);
        double sum = 0.0;
        for (int i = 0; i < labels.size(); i++) {
            sum += Math.abs(residual.get(i));
        }
        return sum;
    }

    @Override
    public double minimize(Vector point, Vector increment) {
        double minimizeCoefficient = 0;
        for (int i = 0; i < point.size(); i++) {
            minimizeCoefficient -= point.get(i) * (1 / increment.get(i));
        }
        return minimizeCoefficient;
    }
}
