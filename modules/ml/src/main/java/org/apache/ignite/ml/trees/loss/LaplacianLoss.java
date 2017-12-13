package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

public class LaplacianLoss implements LossFunction, LinearMinimizible {
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
        double min = apply(labels, predictions);
        double t = 0.01;

        while (true) {
            double cur = apply(labels, predictions.plus(direction.times(t)));
            if (cur > min) {
                break;
            }

            t += 0.01;
        }

        return t / 2;
    }
}
