package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;

public interface LinearlyMinimizable {
    double minimize(Vector labels, Vector predictions, Vector direction);
}
