package org.apache.ignite.ml.trees;

import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.math.StorageConstants;
import org.apache.ignite.ml.math.impls.matrix.SparseDistributedMatrix;
import org.apache.ignite.ml.trees.trainers.boosting.ColumnBoostingDecisionTreesTrainer;
import org.apache.ignite.ml.trees.trainers.columnbased.MatrixColumnDecisionTreeTrainerInput;
import org.apache.ignite.ml.trees.trainers.columnbased.contsplitcalcs.ContinuousSplitCalculators;
import org.apache.ignite.ml.trees.trainers.columnbased.regcalcs.RegionCalculators;

import java.util.HashMap;

/**
 * Created by kroonk on 09.12.17.
 */
public class ColumnBoostingDecisionTreeTrainerTest extends BaseDecisionTreeTest {
    public void test1() {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        int samplesCnt = 499;
        SparseDistributedMatrix samples = new SparseDistributedMatrix(samplesCnt, 2, StorageConstants.COLUMN_STORAGE_MODE, StorageConstants.RANDOM_ACCESS_MODE);

        double[] data = new double[2];
        for (int i = 0; i < samplesCnt; i++) {
            data[0] = 20.0 * Math.PI / samplesCnt * i;
            data[1] = Math.cos(data[0]) > 0 ? 1 : 0;
            samples.setRow(i, data);
        }

        ColumnBoostingDecisionTreesTrainer trainer = new ColumnBoostingDecisionTreesTrainer(4,
                ContinuousSplitCalculators.VARIANCE, RegionCalculators.GINI, RegionCalculators.MEAN, ignite);

        trainer.train(new MatrixColumnDecisionTreeTrainerInput(samples, new HashMap<>()));
    }
}
