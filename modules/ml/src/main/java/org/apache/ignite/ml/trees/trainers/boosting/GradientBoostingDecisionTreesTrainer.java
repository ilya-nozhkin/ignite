package org.apache.ignite.ml.trees.trainers.boosting;

import org.apache.ignite.Ignite;
import org.apache.ignite.ml.Trainer;
import org.apache.ignite.ml.math.StorageConstants;
import org.apache.ignite.ml.math.Tracer;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.VectorUtils;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.apache.ignite.ml.math.distributed.keys.impl.SparseMatrixKey;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.impls.matrix.SparseDistributedMatrix;
import org.apache.ignite.ml.math.impls.storage.matrix.SparseDistributedMatrixStorage;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.apache.ignite.ml.trees.loss.LinearlyMinimizable;
import org.apache.ignite.ml.trees.loss.LossFunction;
import org.apache.ignite.ml.trees.models.BoostedTreesModel;
import org.apache.ignite.ml.trees.models.DecisionTreeModel;
import org.apache.ignite.ml.trees.trainers.DecisionTreeTrainerInput;

import javax.cache.Cache;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class GradientBoostingDecisionTreesTrainer<V extends DecisionTreeTrainerInput,
                                                  I extends BoostingDecisionTreesTrainerInput<V>,
                                                  F extends LossFunction & LinearlyMinimizable>
                    implements Trainer<BoostedTreesModel, I> {
    private Trainer<DecisionTreeModel, V> trainer;
    private Ignite ignite;
    private F loss;
    private int maxIterations = 100;

    public GradientBoostingDecisionTreesTrainer(Trainer<DecisionTreeModel, V> treeTrainer, F loss, Ignite ignite) {
        this.trainer = treeTrainer;
        this.ignite = ignite;
        this.loss = loss;
    }

    private void computePrediction(Vector newPrediction, DecisionTreeModel model, SparseDistributedMatrix samples) {
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

        predictionMap.forEach((key, value) -> newPrediction.setX(key, value));
    }

    private double computeWeight(Vector labels, Vector fullPrediction, Vector newPrediction) {
        return loss.minimize(labels, fullPrediction, newPrediction);
    }

    private void updatePrediction(Vector fullPrediction, Vector newPrediction, double weight) {
        for (int i = 0; i < newPrediction.size(); i++) {
            fullPrediction.setX(i, fullPrediction.getX(i) + newPrediction.getX(i) * weight);
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
        int numSamples = input.getInput().samplesCount();

        Vector fullPrediction = new DenseLocalOnHeapVector(numSamples);
        Vector newPrediction = new DenseLocalOnHeapVector(numSamples);
        Vector labels = new DenseLocalOnHeapVector(numSamples);

        SparseDistributedMatrix samples = createSamplesMatrix(input.getInput());

        labels.assign(input.getInput().labels(ignite));

        ArrayList<Double> weights = new ArrayList<>();
        ArrayList<DecisionTreeModel> trees = new ArrayList<>();

        for (int i = 0; i < maxIterations; i++) {
            Vector residuals = loss.invGradient(labels, fullPrediction);
            input.substituteLabels(residuals.getStorage().data());

            DecisionTreeModel tree = trainer.train(input.getInput());
            computePrediction(newPrediction, tree, samples);

            double weight = computeWeight(labels, fullPrediction, newPrediction);

            weights.add(weight);
            trees.add(tree);

            updatePrediction(fullPrediction, newPrediction, weight);
            System.out.print(loss.apply(labels, fullPrediction));
            System.out.print(" ");
            System.out.println(weight);

            try {
                Tracer.saveAsCsv(fullPrediction, "%.6f", "/home/kroonk/gbdt/prediction_" + Integer.toString(i) + ".csv");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return new BoostedTreesModel(trees, weights);
    }
}
