package org.apache.ignite.ml.math.derivatives;

import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.functions.IgniteToDoubleFunction;

public class VectorToDoubleDerivative implements IgniteToDoubleFunction<Vector> {
    private IgniteToDoubleFunction<Vector> function;
    private Vector direction;
    private double norm;

    public VectorToDoubleDerivative(IgniteToDoubleFunction<Vector> function, Vector direction) {
        this.function = function;
        this.direction = direction;
        this.norm = Math.sqrt(direction.dot(direction));
    }

    @Override
    public double applyAsDouble(Vector point) {
        double value = function.applyAsDouble(point);
     /*   Vector nextPoint = point.plus(direction);
        double nextValue = function.applyAsDouble(nextPoint);
        return (nextValue - value) / norm;*/
        return applyAsDouble(point, value);
    }

    public double applyAsDouble(Vector point, double value) {
        Vector nextPoint = point.plus(direction);
        double nextValue = function.applyAsDouble(nextPoint);
        return (nextValue - value) / norm;
    }
}
