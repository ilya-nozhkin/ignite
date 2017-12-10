package org.apache.ignite.ml.trees.trainers.boosting;

import org.apache.ignite.Ignite;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.Trainer;
import org.apache.ignite.ml.math.StorageConstants;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.VectorUtils;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.apache.ignite.ml.math.distributed.keys.impl.SparseMatrixKey;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.impls.matrix.SparseDistributedMatrix;
import org.apache.ignite.ml.math.impls.storage.matrix.SparseDistributedMatrixStorage;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.apache.ignite.ml.trees.models.BoostedTreesModel;
import org.apache.ignite.ml.trees.models.DecisionTreeModel;
import org.apache.ignite.ml.trees.trainers.DecisionTreeTrainerInput;
import org.apache.ignite.ml.trees.trainers.columnbased.ColumnDecisionTreeTrainerInput;

import javax.cache.Cache;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

public class GradientBoostingDecisionTreesTrainer<V extends DecisionTreeTrainerInput,
                                                  I extends BoostingDecisionTreesTrainerInput<V>>
                    implements Trainer<BoostedTreesModel, I> {
    private Trainer<DecisionTreeModel, V> trainer;
    private Ignite ignite;
    private int maxIterations = 10;

    private double[] fullPrediction;
    private double[] labels;

    public GradientBoostingDecisionTreesTrainer(Trainer<DecisionTreeModel, V> treeTrainer, Ignite ignite) {
        this.trainer = treeTrainer;
        this.ignite = ignite;
    }

    private void computeGradient(double residuals[]) {
        double sum = 0.0;
        for (int i = 0; i < residuals.length; i++) {
            residuals[i] = labels[i] - fullPrediction[i]; // * 2
            sum += residuals[i] * residuals[i];
        }
        System.out.println(sum);
    }

    private double[] computePrediction(DecisionTreeModel model, SparseDistributedMatrix samples) {
        String cacheName = ((SparseDistributedMatrixStorage) samples.getStorage()).cacheName();
        UUID uuid = samples.getUUID();

        int featuresCnt = samples.columnSize();
        ConcurrentHashMap<Integer, Double> predictionMap = CacheUtils.distributedFold(cacheName,
                (IgniteBiFunction<Cache.Entry<SparseMatrixKey, Map<Integer, Double>>,
                                  ConcurrentHashMap<Integer, Double>,
                                  ConcurrentHashMap<Integer, Double>>)
                        (vectorWithIndex, localPrediction) -> {
                    int idx = vectorWithIndex.getKey().index();
                    Vector features;
                    if (vectorWithIndex.getValue().size() > 0)
                        features = VectorUtils.fromMap(vectorWithIndex.getValue(), false);
                    else
                        features = new DenseLocalOnHeapVector(featuresCnt);

                    Double currentPrediction = model.predict(features);
                    localPrediction.put(idx, currentPrediction);
                    return localPrediction;
                },
                key -> key.dataStructureId().equals(uuid),
                (prediction1, prediction2) -> {
                    prediction1.putAll(prediction2);
                    return prediction1;
                },
                () -> new ConcurrentHashMap<>());

        int samplesCnt = samples.rowSize();
        double[] prediction = new double[samplesCnt];
        predictionMap.forEach((key, value) -> prediction[key] = value);

        return prediction;
    }

    private double computeWeight() {
        return 1;
    }

    private void updatePrediction(double[] newPrediction, double weight) {
        for (int i = 0; i < newPrediction.length; i++) {
            fullPrediction[i] += newPrediction[i] * weight;
        }
    }

    private SparseDistributedMatrix createSamplesMatrix(V input) {
        SparseDistributedMatrix samples = new SparseDistributedMatrix(input.samplesCount(), input.featuresCount(),
                StorageConstants.ROW_STORAGE_MODE, StorageConstants.RANDOM_ACCESS_MODE);

        double[] vectorData = new double[input.featuresCount()];
        input.samples().forEach(sample -> {
            int idx = sample.get1();
            Vector features = sample.get2();

            for (int i = 0; i < features.size(); i++) {
                vectorData[i] = features.getX(i);
            }

            samples.setRow(idx, vectorData);
        });

        return samples;
    }

    @Override
    public BoostedTreesModel train(I input) {
            fullPrediction = new double[input.getInput().samplesCount()];

            SparseDistributedMatrix samples = createSamplesMatrix(input.getInput());

            this.labels = Arrays.copyOf(input.getInput().labels(ignite), input.getInput().samplesCount());

            double residuals[] = new double[this.labels.length];

            ArrayList<Double> weights = new ArrayList<>();
            ArrayList<DecisionTreeModel> trees = new ArrayList<>();

            for (int i = 0; i < maxIterations; i++) {
                computeGradient(residuals);
                input.substituteLabels(residuals);

                DecisionTreeModel tree = trainer.train(input.getInput());
                double[] newPrediction = computePrediction(tree, samples);

                double weight = computeWeight();

                weights.add(weight);
                trees.add(tree);

                updatePrediction(newPrediction, weight);
            }

        return new BoostedTreesModel(trees, weights);
    }
}
