package org.apache.ignite.ml.clustering;

import org.apache.ignite.lang.IgniteUuid;
import org.apache.ignite.ml.math.DistanceMeasure;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.VectorUtils;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.apache.ignite.ml.math.distributed.keys.impl.SparseMatrixKey;
import org.apache.ignite.ml.math.functions.Functions;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.impls.matrix.DenseLocalOnHeapMatrix;
import org.apache.ignite.ml.math.impls.matrix.SparseDistributedMatrix;
import org.apache.ignite.ml.math.impls.storage.matrix.SparseDistributedMatrixStorage;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.apache.ignite.ml.math.util.MatrixUtil;

import javax.cache.Cache;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/** Implements distributed version of Fuzzy C-Means clusterization on equal-weighted points */
public class FuzzyCMeansDistributedClusterer extends BaseFuzzyCMeansClusterer<SparseDistributedMatrix> {
    /** Random numbers generator which is used in centers selection */
    private Random random;

    /** The value that is used to initialize random numbers generator */
    private long seed;

    /** The number of initialization steps each of which adds some number of candidates for being a center */
    private int initializationSteps;

    /** The maximum number of iterations of K-Means algorithm which selects the required number of centers */
    private int kMeansMaxIterations;

    /**
     * Constructor that stores all required parameters
     *
     * @param measure distance measure
     * @param exponentialWeight specific constant which is used in calculating of membership matrix
     * @param maxCentersDelta max distance between old and new centers which indicates that algorithm should stop
     * @param seed seed for random numbers generator
     * @param initializationSteps number of steps in primary centers selection (the more steps, the more candidates)
     * @param kMeansMaxIterations maximum number of K-Means iteration in primary centers selection
     */
    public FuzzyCMeansDistributedClusterer(DistanceMeasure measure, double exponentialWeight, double maxCentersDelta,
                                           Long seed, int initializationSteps, int kMeansMaxIterations) {
        super(measure, exponentialWeight, maxCentersDelta);

        this.seed = seed != null ? seed : new Random().nextLong();
        this.initializationSteps = initializationSteps;
        this.kMeansMaxIterations = kMeansMaxIterations;
        random = new Random(this.seed);
    }

    /**
     * Calculate new minimum distances from each point to nearest center
     *
     * @param cacheName cache name of point matrix
     * @param uuid uuid of point matrix
     * @param newCenters list of centers that was added on previous step
     * @return hash map of distances
     */
    private ConcurrentHashMap<Integer, Double> getNewCosts(String cacheName, IgniteUuid uuid,
                                                           List<Vector> newCenters) {
        return CacheUtils.distributedFold(cacheName,
                (IgniteBiFunction<Cache.Entry<SparseMatrixKey, ConcurrentHashMap<Integer, Double>>,
                ConcurrentHashMap<Integer, Double>,
                ConcurrentHashMap<Integer, Double>>)(vectorWithIndex, map) -> {
                    Vector vector = VectorUtils.fromMap(vectorWithIndex.getValue(), false);

                    for (Vector center : newCenters) {
                        map.merge(vectorWithIndex.getKey().index(),
                                  distance(vector, center),
                                  Functions.MIN);
                    }

                    return map;
                },
                key -> key.matrixId().equals(uuid),
                (map1, map2) -> {
                    map1.putAll(map2);
                    return map1;
                },
                new ConcurrentHashMap<>());
    }

    /**
     * choose some number of center candidates from source points according to their costs
     *
     * @param cacheName cache name of point matrix
     * @param uuid uuid of point matrix
     * @param costs hash map with costs (distances to nearest center)
     * @param costsSum sum of costs
     * @param k the estimated number of centers
     * @return the list of new candidates
     */
    private List<Vector> getNewCenters(String cacheName, IgniteUuid uuid,
                                       ConcurrentHashMap<Integer, Double> costs, double costsSum, int k) {
        return CacheUtils.distributedFold(cacheName,
                (IgniteBiFunction<Cache.Entry<SparseMatrixKey, ConcurrentHashMap<Integer, Double>>,
                                  List<Vector>,
                                  List<Vector>>)(vectorWithIndex, centers) -> {
                    Integer index = vectorWithIndex.getKey().index();
                    Vector vector = VectorUtils.fromMap(vectorWithIndex.getValue(), false);

                    double probability = (costs.get(index) * 2.0 * k) / costsSum;

                    if (new Random(seed * (index + 1)).nextDouble() < probability) {
                        centers.add(vector);
                    }

                    return centers;
                },
                key -> key.matrixId().equals(uuid),
                (list1, list2) -> {
                    list1.addAll(list2);
                    return list1;
                },
                new ArrayList<>());
    }

    /**
     * Weight each center with number of points for which it is the nearest
     *
     * @param cacheName cache name of the point matrix
     * @param uuid uuid of the point matrix
     * @param centers list of centers
     * @return hash map of weights
     */
    public ConcurrentHashMap<Integer, Integer> weightCenters(String cacheName, IgniteUuid uuid, List<Vector> centers) {
        if (centers.size() == 0) {
            return new ConcurrentHashMap<>();
        }

        return CacheUtils.distributedFold(cacheName,
                (IgniteBiFunction<Cache.Entry<SparseMatrixKey, ConcurrentHashMap<Integer, Double>>,
                                  ConcurrentHashMap<Integer, Integer>,
                                  ConcurrentHashMap<Integer, Integer>>)(vectorWithIndex, counts) -> {
                    Vector vector = VectorUtils.fromMap(vectorWithIndex.getValue(), false);

                    int nearest = 0;
                    double minDistance = distance(centers.get(nearest), vector);

                    for (int i = 0; i < centers.size(); i++) {
                        double currentDistance = distance(centers.get(i), vector);
                        if (currentDistance < minDistance) {
                            minDistance = currentDistance;
                            nearest = i;
                        }
                    }

                    counts.compute(nearest, (index, value) -> value == null ? 1 : value + 1);

                    return counts;
                },
                key -> key.matrixId().equals(uuid),
                (map1, map2) -> {
                    map1.putAll(map2);
                    return map1;
                },
                new ConcurrentHashMap<>());
    }

    /**
     * Weight candidates and use K-Means to choose required number of them
     *
     * @param cacheName cache name of the point matrix
     * @param uuid uuid of the point matrix
     * @param centers list of candidates
     * @param k the estimated number of centers
     * @return k centers
     */
    public Vector[] chooseKCenters(String cacheName, IgniteUuid uuid, List<Vector> centers, int k) {
        centers = centers.stream().distinct().collect(Collectors.toList());

        ConcurrentHashMap<Integer, Integer> weightsMap = weightCenters(cacheName, uuid, centers);

        List<Double> weights = new ArrayList<>(centers.size());

        for (int i = 0; i < centers.size(); i++) {
            weights.add(i, Double.valueOf(weightsMap.getOrDefault(i, 0)));
        }

        DenseLocalOnHeapMatrix centersMatrix = MatrixUtil.fromList(centers, true);

        KMeansLocalClusterer clusterer = new KMeansLocalClusterer(measure, kMeansMaxIterations, seed);
        return clusterer.cluster(centersMatrix, k).centers();
    }

    /**
     * Choose k primary centers from source points
     *
     * @param points matrix with source points
     * @param k number of centers
     * @return array of primary centers
     */
    public Vector[] initializeCenters(SparseDistributedMatrix points, int k) {
        int pointsNumber = points.rowSize();

        Vector firstCenter = points.viewRow(random.nextInt(pointsNumber));

        List<Vector> centers = new ArrayList<>();
        List<Vector> newCenters = new ArrayList<>();

        centers.add(firstCenter);
        newCenters.add(firstCenter);

        ConcurrentHashMap<Integer, Double> costs = new ConcurrentHashMap<>();

        int step = 0;
        IgniteUuid uuid = points.getUUID();
        String cacheName = ((SparseDistributedMatrixStorage) points.getStorage()).cacheName();

        while(step < initializationSteps) {
            ConcurrentHashMap<Integer, Double> newCosts = getNewCosts(cacheName, uuid, newCenters);

            for (Integer key : newCosts.keySet()) {
                costs.merge(key, newCosts.get(key), Math::min);
            }

            double costsSum = costs.values().stream().mapToDouble(Double::valueOf).sum();

            newCenters = getNewCenters(cacheName, uuid, costs, costsSum, k);
            centers.addAll(newCenters);

            step++;
        }

        return chooseKCenters(cacheName, uuid, centers, k);
    }

    /**
     * calculate matrix of membership coefficients for each point and each center
     *
     * @param points matrix with source points
     * @param centers array of current centers
     * @return membership matrix and sums of membership coefficient for each center
     */
    public MembershipsAndSums calculateMembership(SparseDistributedMatrix points, Vector[] centers) {
        String cacheName = ((SparseDistributedMatrixStorage) points.getStorage()).cacheName();
        IgniteUuid uuid = points.getUUID();
        double fuzzyMembershipCoefficient = 2 / (exponentialWeight - 1);

        return CacheUtils.distributedFold(cacheName,
                (IgniteBiFunction<Cache.Entry<SparseMatrixKey, ConcurrentHashMap<Integer, Double>>,
                        MembershipsAndSums,
                        MembershipsAndSums>)(vectorWithIndex, membershipsAndSums) -> {
                    Integer index = vectorWithIndex.getKey().index();
                    Vector point = VectorUtils.fromMap(vectorWithIndex.getValue(), false);
                    Vector distances = new DenseLocalOnHeapVector(centers.length);
                    Vector pointMemberships = new DenseLocalOnHeapVector(centers.length);

                    for (int i = 0; i < centers.length; i++) {
                        distances.setX(i, distance(centers[i], point));
                    }

                    for (int i = 0; i < centers.length; i++) {
                        double invertedFuzzyWeight = 0.0;
                        for (int j = 0; j < centers.length; j++) {
                            double value = Math.pow(distances.getX(i) / distances.getX(j), fuzzyMembershipCoefficient);
                            if (Double.isNaN(value)) {
                                value = 1.0;
                            }
                            invertedFuzzyWeight += value;
                        }
                        double membership = Math.pow(1.0 / invertedFuzzyWeight, exponentialWeight);
                        pointMemberships.setX(i, membership);
                    }

                    membershipsAndSums.memberships.put(index, pointMemberships);
                    membershipsAndSums.membershipSums = membershipsAndSums.membershipSums.plus(pointMemberships);

                    return membershipsAndSums;
                },
                key -> key.matrixId().equals(uuid),
                (mem1, mem2) -> {
                    mem1.merge(mem2);
                    return mem1;
                },
                new MembershipsAndSums(centers.length));
    }

    /**
     * Calculate new centers according to membership matrix
     *
     * @param points matrix with source points
     * @param membershipsAndSums membership matrix and sums of membership coefficient for each center
     * @param k number of centers
     * @return array of new centers
     */
    public Vector[] calculateNewCenters(SparseDistributedMatrix points, MembershipsAndSums membershipsAndSums, int k) {
        String cacheName = ((SparseDistributedMatrixStorage) points.getStorage()).cacheName();
        IgniteUuid uuid = points.getUUID();

        DenseLocalOnHeapVector[] centerSumsArray = new DenseLocalOnHeapVector[k];
        for (int i = 0; i < k; i++) {
            centerSumsArray[i] = new DenseLocalOnHeapVector(points.columnSize());
        }

        Vector[] centers = CacheUtils.distributedFold(cacheName,
                (IgniteBiFunction<Cache.Entry<SparseMatrixKey, ConcurrentHashMap<Integer, Double>>,
                                  Vector[],
                                  Vector[]>)(vectorWithIndex, centerSums) -> {
                    Integer index = vectorWithIndex.getKey().index();
                    Vector point = MatrixUtil.localCopyOf(VectorUtils.fromMap(vectorWithIndex.getValue(), false));
                    Vector pointMemberships = membershipsAndSums.memberships.get(index);

                    for (int i = 0; i < k; i++) {
                        Vector weightedPoint = point.times(pointMemberships.getX(i));
                        centerSums[i] = centerSums[i].plus(weightedPoint);
                    }

                    return centerSums;
                },
                key -> key.matrixId().equals(uuid),
                (sums1, sums2) -> {
                    for (int i = 0; i < k; i++) {
                        sums1[i] = sums1[i].plus(sums2[i]);
                    }
                    return sums1;
                },
                centerSumsArray);

        for (int i = 0; i < k; i++) {
            centers[i] = centers[i].divide(membershipsAndSums.membershipSums.getX(i));
        }

        return centers;
    }

    /**
     * Check if centers have moved insignificantly
     *
     * @param centers old centers
     * @param newCenters new centers
     * @return the result of comparison
     */
    private boolean isFinished(Vector[] centers, Vector[] newCenters) {
        int numCenters = centers.length;

        for (int i = 0; i < numCenters; i++) {
            if (distance(centers[i], newCenters[i]) > maxCentersDelta) {
                return false;
            }
        }

        return true;
    }

    /** {@inheritDoc} */
    @Override
    public FuzzyCMeansModel cluster(SparseDistributedMatrix points, int k) {
        Vector[] centers = initializeCenters(points, k);

        boolean finished = false;
        while (!finished) {
            MembershipsAndSums newMembershipsAndSums = calculateMembership(points, centers);
            Vector[] newCenters = calculateNewCenters(points, newMembershipsAndSums, k);

            finished = isFinished(centers, newCenters);

            centers = newCenters;
        }

        return new FuzzyCMeansModel(centers, measure);
    }

    /** Service class used to optimize counting of membership sums */
    private class MembershipsAndSums {
        /** Membership matrix */
        public ConcurrentHashMap<Integer, Vector> memberships = new ConcurrentHashMap<>();

        /** Membership sums */
        public Vector membershipSums;

        /**
         * Default constructor
         *
         * @param k number of centers
         */
        public MembershipsAndSums(int k) {
            membershipSums = new DenseLocalOnHeapVector(k);
        }

        /**
         * Merge results of calculation for different parts of points
         * @param another another part of memberships and sums
         */
        public void merge(MembershipsAndSums another) {
            memberships.putAll(another.memberships);
            membershipSums = membershipSums.plus(another.membershipSums);
        }
    }
}
