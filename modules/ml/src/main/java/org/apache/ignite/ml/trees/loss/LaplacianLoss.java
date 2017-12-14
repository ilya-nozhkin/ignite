package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;

public class LaplacianLoss implements LossFunction, LinearlyMinimizable {
    @Override
    public Vector invGradient(Vector labels, Vector predictions) {
        Vector gradient = labels.minus(predictions);
        for (int i = 0; i < labels.size(); i++) {
            double value = gradient.get(i);
            gradient.set(i, Math.signum(value));
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
    public double minimize(Vector labels, Vector predictions, Vector direction) {
        int dim = predictions.size();
        Vector diff = labels.minus(predictions);

        double mint = 0;
        double minDist = apply(labels, predictions);

        for (int i = 0; i < dim; i++) {
            double t = diff.getX(i) / direction.getX(i);
            if (!Double.isNaN(t)) {
                double dist = apply(labels, direction.times(t).plus(predictions));
                if (dist < minDist) {
                    minDist = dist;
                    mint = t;
                }
            }
        }

        return mint;
    }
}
