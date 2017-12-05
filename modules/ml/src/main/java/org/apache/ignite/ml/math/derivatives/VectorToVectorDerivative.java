package org.apache.ignite.ml.math.derivatives;

import org.apache.ignite.ml.math.Vector;
import java.util.function.Function;

public class VectorToVectorDerivative implements Function<Vector, Vector> {
    private Function<Vector, Vector> function;
    private Vector direction;
    private double norm;

    public VectorToVectorDerivative(Function<Vector, Vector> function, Vector direction) {
        this.function = function;
        this.direction = direction;
        this.norm = Math.sqrt(direction.dot(direction));
    }

    @Override
    public Vector apply(Vector point) {
        Vector value = function.apply(point);
      /*  Vector nextValue = function.apply(point.plus(direction));
        Vector difference = nextValue.minus(value);
        return difference.divide(norm);*/
        return apply(point, value);
    }

    public Vector apply(Vector point, Vector value) {
        Vector nextValue = function.apply(point.plus(direction));
        Vector difference = nextValue.minus(value);
        return difference.divide(norm);
    }
}
