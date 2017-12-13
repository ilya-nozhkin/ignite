package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

/**
 * Created by Виктория on 12.12.2017.
 * Lq
 */
public class QuantileLoss implements LossFunction, LinearMinimizible {
    private double quantile = 0.25;

    public QuantileLoss(double quantile) {
        if (quantile >= 0 && quantile <= 1) {
            this.quantile = quantile;
        }
    }

    @Override
    public Vector computeGradient(Vector labels, Vector predictions) {
        Vector gradient = new DenseLocalOnHeapVector();
        for (int i = 0; i < labels.size(); i++) {
            double signum = Math.signum(labels.get(i) - predictions.get(i));
            double gradientI = (signum == -1.0) ? quantile : (quantile - 1);
            gradient.set(i, gradientI);
        }
        return gradient;
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        double sum = 0;
        for (int i = 0; i < labels.size(); i++) {
            double residual = labels.get(i) - predictions.get(i);
            residual = (residual < 0) ? (1 - quantile) * Math.abs(residual) : quantile * residual;
            sum += residual;
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
