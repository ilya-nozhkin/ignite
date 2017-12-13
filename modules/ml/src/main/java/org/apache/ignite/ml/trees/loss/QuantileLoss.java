package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

public class QuantileLoss implements LossFunction, LinearMinimizible {
    private double quantile = 0.25;

    public QuantileLoss(double quantile) {
        if (quantile >= 0 && quantile <= 1) {
            this.quantile = quantile;
        }
    }

    @Override
    public Vector invGradient(Vector labels, Vector predictions) {
        Vector gradient = labels.minus(predictions);
        for (int i = 0; i < labels.size(); i++) {
            double signum = Math.signum(gradient.get(i));
            double gradientI = (signum == -1.0) ? quantile : (quantile - 1);
            gradient.set(i, gradientI);
        }
        return gradient;
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        Vector residual = labels.minus(predictions);
        double sum = 0;
        for (int i = 0; i < labels.size(); i++) {
            double residualI = residual.get(i);
            sum += (residualI < 0) ? (1 - quantile) * Math.abs(residualI) : quantile * residualI;
        }
        return sum;
    }

    @Override
    public double minimize(Vector labels, Vector predictions, Vector direction) {
        return 0;
    }
}
