package org.apache.ignite.ml.trees.trainers.boosting;

import org.apache.ignite.Ignite;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.trees.trainers.columnbased.ColumnDecisionTreeTrainerInput;

import java.util.Map;
import java.util.stream.Stream;

public class ColumnBoostingDecisionTreesTrainerInput
        implements BoostingDecisionTreesTrainerInput<ColumnDecisionTreeTrainerInput> {

    double[] substitutedLabels = null;
    ColumnDecisionTreeTrainerInput source;
    ColumnDecisionTreesTrainerInputProxy proxy;

    public ColumnBoostingDecisionTreesTrainerInput(ColumnDecisionTreeTrainerInput input) {
        source = input;
        proxy = new ColumnDecisionTreesTrainerInputProxy();
    }

    @Override
    public void substituteLabels(double[] labels) {
        substitutedLabels = labels;
    }

    @Override
    public ColumnDecisionTreeTrainerInput getInput() {
        return proxy;
    }

    private class ColumnDecisionTreesTrainerInputProxy implements ColumnDecisionTreeTrainerInput {

        // Changed.

        @Override
        public double[] labels(Ignite ignite) {
            if (substitutedLabels != null)
                return substitutedLabels;
            else
                return source.labels(ignite);
        }

        // Not changed.

        @Override
        public Stream<IgniteBiTuple<Integer, Double>> values(int idx) {
            return source.values(idx);
        }

        @Override
        public Object affinityKey(int idx, Ignite ignite) {
            return source.affinityKey(idx, ignite);
        }

        @Override
        public Stream<IgniteBiTuple<Integer, Vector>> samples() {
            return source.samples();
        }

        @Override
        public Map<Integer, Integer> catFeaturesInfo() {
            return source.catFeaturesInfo();
        }

        @Override
        public int featuresCount() {
            return source.featuresCount();
        }

        @Override
        public int samplesCount() {
            return source.samplesCount();
        }
    }
}
