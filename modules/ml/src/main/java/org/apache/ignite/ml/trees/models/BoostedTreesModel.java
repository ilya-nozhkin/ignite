package org.apache.ignite.ml.trees.models;

import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.trees.nodes.DecisionTreeNode;

import java.util.ArrayList;

public class BoostedTreesModel implements Model<Vector, Double> {
    private final ArrayList<DecisionTreeNode> roots;
    private final Vector weights;

    public BoostedTreesModel(ArrayList<DecisionTreeNode> roots, Vector weights) {
        this.roots = roots;
        this.weights = weights;
    }

    @Override public Double predict(Vector val) {
        Double prediction = 0.0;
        for (int i = 0; i < roots.size(); i++) {
            prediction += roots.get(i).process(val) * weights.get(i);
        }
        return prediction;
    }
}
