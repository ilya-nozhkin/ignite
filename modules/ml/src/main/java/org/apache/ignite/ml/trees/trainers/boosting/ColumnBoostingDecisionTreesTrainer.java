package org.apache.ignite.ml.trees.trainers.boosting;

import org.apache.ignite.Ignite;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.Trainer;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.ml.trees.ContinuousRegionInfo;
import org.apache.ignite.ml.trees.ContinuousSplitCalculator;
import org.apache.ignite.ml.trees.models.BoostedTreesModel;
import org.apache.ignite.ml.trees.models.DecisionTreeModel;
import org.apache.ignite.ml.trees.trainers.columnbased.ColumnDecisionTreeTrainer;
import org.apache.ignite.ml.trees.trainers.columnbased.ColumnDecisionTreeTrainerInput;
import org.apache.ignite.ml.trees.trainers.columnbased.MatrixColumnDecisionTreeTrainerInput;

import java.util.ArrayList;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class ColumnBoostingDecisionTreesTrainer<D extends ContinuousRegionInfo>
        implements Trainer<BoostedTreesModel, ColumnBoostingDecisionTreesTrainerInput> {
    private final double value = 0;
    private ColumnDecisionTreeTrainer trainer;
    private Ignite ignite;
    private int maxIterations;

    private double[] prediction;
    private double[] labels;

    public ColumnBoostingDecisionTreesTrainer(int maxDepth,
                IgniteFunction<ColumnDecisionTreeTrainerInput, ? extends ContinuousSplitCalculator<D>> continuousCalculatorProvider,
                IgniteFunction<ColumnDecisionTreeTrainerInput, IgniteFunction<DoubleStream, Double>> categoricalCalculatorProvider,
                IgniteFunction<DoubleStream, Double> regCalc, Ignite ignite) {
        this.trainer = new ColumnDecisionTreeTrainer(maxDepth, continuousCalculatorProvider,
                categoricalCalculatorProvider, regCalc, ignite);
        this.ignite = ignite;
    }

    private void computeGradient(double residuals[]) {
        for (int i = 0; i < residuals.length; i++) {
            residuals[i] = labels[i] - prediction[i]; // * 2
        }
    }

    private double[] computePrediction(DecisionTreeModel model) {
        return null;
    }

    private double computeWeight() {
        return 1;
    }

    @Override
    public BoostedTreesModel train(ColumnBoostingDecisionTreesTrainerInput boostingInput) {
            ColumnDecisionTreeTrainerInput input =
                    new MatrixColumnDecisionTreeTrainerInput(boostingInput.getMatrix(), boostingInput.getFeaturesInfo());
            this.labels = input.labels(ignite);

            ColumnDecisionTreeTrainerInputProxy proxy = new ColumnDecisionTreeTrainerInputProxy(input, ignite);

            double residuals[] = new double[boostingInput.getMatrix().rowSize()];

            ArrayList<Double> weights = new ArrayList<>();
            ArrayList<DecisionTreeModel> trees = new ArrayList<>();

            for (int i = 0; i < maxIterations; i++) {
                computeGradient(residuals);
                proxy.setSubsitutedLabels(residuals);

                DecisionTreeModel tree = trainer.train(proxy);
                double weight = computeWeight();

                weights.add(weight);
                trees.add(tree);
            }

        return new BoostedTreesModel(trees, weights);
    }




    private class ColumnDecisionTreeTrainerInputProxy implements ColumnDecisionTreeTrainerInput {
        private ColumnDecisionTreeTrainerInput input;
        private double[] substitutedLabels;

        ColumnDecisionTreeTrainerInputProxy(ColumnDecisionTreeTrainerInput input, Ignite ignite) {
            this.input = input;
            this.substitutedLabels = input.labels(ignite);
        }

        public void setSubsitutedLabels(double subsitutedLabels[]) {
            this.substitutedLabels = subsitutedLabels;
        }

        @Override
        public Stream<IgniteBiTuple<Integer, Double>> values(int idx) {
            return input.values(idx);
        }

        @Override
        public double[] labels(Ignite ignite) {
            return substitutedLabels;
        }

        @Override
        public Map<Integer, Integer> catFeaturesInfo() {
            return input.catFeaturesInfo();
        }

        @Override
        public int featuresCount() {
            return input.featuresCount();
        }

        @Override
        public Object affinityKey(int idx, Ignite ignite) {
            return affinityKey(idx, ignite);
        }
    }
}
