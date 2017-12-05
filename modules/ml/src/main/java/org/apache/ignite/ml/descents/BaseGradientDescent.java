package org.apache.ignite.ml.descents;

import org.apache.ignite.ml.math.Vector;

import java.util.function.Function;

/**
 * Created by Виктория on 03.12.2017.
 */
abstract public class BaseGradientDescent implements Descent{
    Function<Vector, Double> function;
    Function<Vector, Double> derivative;

    public BaseGradientDescent(Function<Vector, Double> function) {
        this.function = function;
    }

    public BaseGradientDescent(Function<Vector, Double> function, Function<Vector, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    protected Vector descent(Vector point, double delta, int maxIterations) {
        Vector lastPoint;
        int iteration = 0;
        Vector gradient = gradient(point).divide(gradientStep(iteration));
        while (iteration < maxIterations && Math.sqrt(gradient.dot(gradient)) < delta) {
            lastPoint = point;
            point = lastPoint.minus(gradient.divide(iteration));
            gradient = gradient(point);
            iteration++;
        }
        return point;
    }

    private double gradientStep(int iteration) {
        return 1 / iteration;
    }

    abstract protected Vector gradient(Vector point);
}
