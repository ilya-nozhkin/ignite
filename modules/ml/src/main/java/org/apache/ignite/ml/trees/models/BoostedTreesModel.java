package org.apache.ignite.ml.trees.models;

import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.math.Vector;

import java.util.ArrayList;

public class BoostedTreesModel implements Model<Vector, Double> {
    private final ArrayList<DecisionTreeModel> trees;
    private final ArrayList<Double> weights;

    public BoostedTreesModel(ArrayList<DecisionTreeModel> roots, ArrayList<Double> weights) {
        this.trees = roots;
        this.weights = weights;
    }

    @Override public Double predict(Vector val) {
        Double prediction = 0.0;
        for (int i = 0; i < trees.size(); i++) {
            prediction += trees.get(i).predict(val) * weights.get(i);
        }
        return prediction;
    }
}
