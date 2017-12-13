package org.apache.ignite.ml.trees;

import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.math.StorageConstants;
import org.apache.ignite.ml.math.Tracer;
import org.apache.ignite.ml.math.impls.matrix.SparseDistributedMatrix;
import org.apache.ignite.ml.trees.loss.GaussianLoss;
import org.apache.ignite.ml.trees.loss.LaplacianLoss;
import org.apache.ignite.ml.trees.trainers.boosting.ColumnBoostingDecisionTreesTrainerInput;
import org.apache.ignite.ml.trees.trainers.boosting.GradientBoostingDecisionTreesTrainer;
import org.apache.ignite.ml.trees.trainers.columnbased.ColumnDecisionTreeTrainer;
import org.apache.ignite.ml.trees.trainers.columnbased.MatrixColumnDecisionTreeTrainerInput;
import org.apache.ignite.ml.trees.trainers.columnbased.contsplitcalcs.ContinuousSplitCalculators;
import org.apache.ignite.ml.trees.trainers.columnbased.regcalcs.RegionCalculators;

import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

public class BoostingDecisionTreeTrainerTest extends BaseDecisionTreeTest {
    public void test1() {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        int samplesCnt = 1000;
        SparseDistributedMatrix samples = new SparseDistributedMatrix(samplesCnt, 2, StorageConstants.COLUMN_STORAGE_MODE, StorageConstants.RANDOM_ACCESS_MODE);

        Random rnd = new Random();

        double[] data = new double[2];
        for (int i = 0; i < samplesCnt; i++) {
            data[0] = 5 * rnd.nextDouble() * Math.PI * 2.0;
            data[1] = Math.cos(data[0]) > 0 ? 1 : 0;
            samples.setRow(i, data);
        }

        try {
            Tracer.saveAsCsv(samples, "%.6f", "/home/kroonk/gbdt/matrix.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }

        GaussianLoss loss = new GaussianLoss();
        LaplacianLoss lap = new LaplacianLoss();
        ColumnDecisionTreeTrainer baseTrainer = new ColumnDecisionTreeTrainer(4,
                ContinuousSplitCalculators.VARIANCE, RegionCalculators.VARIANCE, RegionCalculators.MEAN, ignite);
        GradientBoostingDecisionTreesTrainer trainer = new GradientBoostingDecisionTreesTrainer(baseTrainer, loss, ignite);

        trainer.train(new ColumnBoostingDecisionTreesTrainerInput(new MatrixColumnDecisionTreeTrainerInput(samples, new HashMap<>())));
    }
}
