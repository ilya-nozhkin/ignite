package org.apache.ignite.ml.descents;

import org.apache.ignite.ml.math.Vector;

import java.util.function.Function;

/**
 * Created by Виктория on 03.12.2017.
 */
public class DistributedGradientDescent extends BaseGradientDescent {
    public DistributedGradientDescent(Function<Vector, Double> function) {
        super(function);
    }

    public DistributedGradientDescent(Function<Vector, Double> function, Function<Vector, Double> derivative) {
        super(function, derivative);
    }

    @Override
    protected Vector gradient(Vector point) {
        return null;
    }

    @Override
    public double perform(Vector beginPoint, double delta, int maxIterations) {
        Vector minPoint = super.descent(beginPoint, delta, maxIterations);
        return function.apply(minPoint);
    }
}
