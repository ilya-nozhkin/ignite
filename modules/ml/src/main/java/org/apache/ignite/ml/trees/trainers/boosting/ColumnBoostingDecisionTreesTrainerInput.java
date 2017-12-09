package org.apache.ignite.ml.trees.trainers.boosting;

import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.math.impls.matrix.SparseDistributedMatrix;

import java.util.Map;

public class ColumnBoostingDecisionTreesTrainerInput {
    private final SparseDistributedMatrix matrix;
    private final Map<Integer, Integer> featuresInfo;

    public ColumnBoostingDecisionTreesTrainerInput(SparseDistributedMatrix matrix, Map<Integer, Integer> featuresInfo) {
        this.matrix = matrix;
        this.featuresInfo = featuresInfo;
    }

    public SparseDistributedMatrix getMatrix() {
        return this.matrix;
    }

    public Map<Integer, Integer> getFeaturesInfo() {
        return this.featuresInfo;
    }
}
