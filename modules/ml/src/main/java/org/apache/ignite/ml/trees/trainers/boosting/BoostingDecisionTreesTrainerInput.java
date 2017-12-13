package org.apache.ignite.ml.trees.trainers.boosting;

public interface BoostingDecisionTreesTrainerInput<I> {
    void substituteLabels(double[] labels);
    I getInput();
}
