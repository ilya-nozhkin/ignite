package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;

public interface LossFunction extends IgniteBiFunction<Vector, Vector, Double> {
    Vector invGradient(Vector labels, Vector predictions);
}
