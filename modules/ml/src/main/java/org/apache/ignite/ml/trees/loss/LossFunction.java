package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;

/**
 * Created by Виктория on 12.12.2017.
 */
public interface LossFunction extends IgniteBiFunction<Vector, Vector, Double> {
    Vector computeGradient(Vector labels, Vector predictions);
    void linearMinimization();
}
