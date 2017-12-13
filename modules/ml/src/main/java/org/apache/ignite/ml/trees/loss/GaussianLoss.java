package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;

public class GaussianLoss implements LossFunction, LinearlyMinimizable {
    @Override
    public Vector invGradient(Vector labels, Vector predictions) {
        return labels.minus(predictions);
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        Vector residual = labels.minus(predictions);
        return residual.dot(residual);
    }

    @Override
    public double minimize(Vector labels, Vector predictions, Vector direction) {
        Vector diff = labels.minus(predictions);
        return diff.dot(direction) / direction.dot(direction);
    }
}
