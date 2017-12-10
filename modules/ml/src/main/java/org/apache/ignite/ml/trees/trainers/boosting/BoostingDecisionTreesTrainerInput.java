package org.apache.ignite.ml.trees.trainers.boosting;

import org.apache.ignite.ml.trees.trainers.DecisionTreeTrainerInput;

public interface BoostingDecisionTreesTrainerInput<I> {
    void substituteLabels(double[] labels);
    I getInput();
}
