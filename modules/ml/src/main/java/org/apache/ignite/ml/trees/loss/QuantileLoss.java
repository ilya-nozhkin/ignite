package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;

/**
 * Created by Виктория on 12.12.2017.
 * Lq
 */
public class QuantileLoss implements LossFunction {
    private double quantile = 0.25;

    public QuantileLoss(double quantile) {
        if (quantile >= 0 && quantile <= 1) {
            this.quantile = quantile;
        }
    }

    @Override
    public Vector computeGradient(Vector labels, Vector predictions) {
        return null;
    }

    @Override
    public void linearMinimization() {
        //todo
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        double sum = 0;
        for (int i = 0; i < labels.size(); i++) {
            double residual = labels.get(i) - predictions.get(i);
            residual = (residual < 0) ? (1 - quantile) * residual : quantile * residual;
            sum += residual;
        }
        return sum;
    }
}
