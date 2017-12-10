package org.apache.ignite.ml.trees.trainers;

import org.apache.ignite.Ignite;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.math.Vector;

import java.util.Map;
import java.util.stream.Stream;

public interface DecisionTreeTrainerInput {
    /**
     * Stream of feature vectors.
     *
     * @return
     */
    Stream<IgniteBiTuple<Integer, Vector>> features();

    /**
     * Labels.
     *
     * @param ignite Ignite instance.
     */
    double[] labels(Ignite ignite);

    /** Information about which features are categorical in the form of feature index -> number of categories. */
    Map<Integer, Integer> catFeaturesInfo();

    /** Number of features. */
    int featuresCount();

    /** Number of samples. */
    int samplesCount();
}
