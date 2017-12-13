package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

/**
 * Created by Виктория on 12.12.2017.
 * L1
 */
public class LaplacianLoss implements LossFunction{
    @Override
    public Vector computeGradient(Vector labels, Vector predictions) {
        Vector gradient = new DenseLocalOnHeapVector();
        for (int i = 0; i < labels.size(); i++) {
            double signum = Math.signum(labels.get(i) - predictions.get(i));
            gradient.set(i, signum);
        }
        return gradient;
    }

    @Override
    public void linearMinimization() {
        //todo
    }

    @Override
    public Double apply(Vector labels, Vector predictions) {
        double sum = 0.0;
        for (int i = 0; i < labels.size(); i++) {
            double residual = labels.get(i) - predictions.get(i);
            sum += Math.abs(residual);
        }
        return sum;
    }
}
