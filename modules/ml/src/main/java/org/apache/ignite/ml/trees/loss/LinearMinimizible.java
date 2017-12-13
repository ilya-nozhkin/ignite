package org.apache.ignite.ml.trees.loss;

import org.apache.ignite.ml.math.Vector;

/**
 * Created by Виктория on 13.12.2017.
 */
public interface LinearMinimizible {
    /**
     *
     * @param point is x_0
     * @param increment is e
     * @return t, which implements minimum of F(x_0 + t * e)
     */
    double minimize(Vector point, Vector increment);
}
