package org.apache.ignite.ml.descents;

import org.apache.ignite.ml.math.Vector;

public interface Descent {
    double perform(Vector beginPoint, double delta, int maxIterations);
}
