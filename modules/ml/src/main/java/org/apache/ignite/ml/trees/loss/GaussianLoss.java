package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

/**
 * Created by Виктория on 12.12.2017.
 * L2
 */
public class GaussianLoss implements LossFunction, LinearMinimizible {
    @Override
    public Vector computeGradient(Vector labels, Vector predictions) {
        Vector gradient = new DenseLocalOnHeapVector();
        for (int i = 0; i < labels.size(); i++) {
            double gradientValue = labels.get(i) - predictions.get(i);
            gradient.set(i, gradientValue);
        }
        return gradient;
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        double sum = 0.0;
        for (int i = 0; i < labels.size(); i++) {
            double residual = labels.get(i) - predictions.get(i);
            sum += residual * residual;
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
