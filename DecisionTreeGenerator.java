import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DecisionTreeGenerator {

    private final Random rng;

    // How often to place a leaf early during GROW (0.0–1.0)
    private static final double LEAF_PROB = 0.35;

    // Threshold ranges per feature — keeps conditions sensible
    // [min, max] for random threshold generation
    private static final double[][] THRESHOLD_RANGES = {
            {0, 5},    // age          (encoded 0–5)
            {0, 2},    // menopause    (0, 1, 2)
            {0, 14},   // tumorSize    (binned range)
            {0, 14},   // invNodes     (binned range)
            {0, 2},    // nodeCaps     (0, 1, 2)
            {1, 3},    // degMalig     (1, 2, 3)
            {0, 1},    // breast       (0, 1)
            {0, 5},    // breastQuad   (0–5)
            {0, 2},    // irradiat     (0, 1, 2)
    };

    public DecisionTreeGenerator(long seed) {
        this.rng = new Random(seed);
    }

    // ── public entry point ─────────────────────────────────────
    /**
     * Build the initial population using ramped half-and-half.
     *
     * @param populationSize  total trees (e.g. 200)
     * @param minDepth        smallest depth (usually 2)
     * @param maxDepth        largest depth  (your 'dd', e.g. 5)
     * @return                list of root DecisionTreeNodes
     */
    public List<DecisionTreeNode> rampedHalfAndHalf(int populationSize,int minDepth, int maxDepth) {
        List<DecisionTreeNode> population = new ArrayList<>();

        int numDepths    = maxDepth - minDepth + 1;
        int perDepth     = populationSize / numDepths;
        int halfPerDepth = perDepth / 2;

        for (int depth = minDepth; depth <= maxDepth; depth++) {
            for (int i = 0; i < halfPerDepth; i++) population.add(buildFull(depth));
            for (int i = 0; i < halfPerDepth; i++) population.add(buildGrow(depth));
        }

        // Fill any remainder
        while (population.size() < populationSize)
            population.add(buildGrow(maxDepth));

        return population;
    }

    // ── FULL method ────────────────────────────────────────────
    /**
     * Every node at depth < maxDepth is a decision node.
     * Every node at maxDepth is a leaf (class 0 or 1).
     */
    DecisionTreeNode buildFull(int maxDepth) {
        return buildFullHelper(0, maxDepth);
    }

    private DecisionTreeNode buildFullHelper(int currentDepth, int maxDepth) {
        if (currentDepth == maxDepth) {
            return randomLeaf();            // must stop here → class label
        }
        DecisionTreeNode node = randomDecisionTreeNode();
        node.yes  = buildFullHelper(currentDepth + 1, maxDepth);
        node.no = buildFullHelper(currentDepth + 1, maxDepth);
        return node;
    }

    // ── GROW method ────────────────────────────────────────────
    /**
     * Before maxDepth, randomly decide: decision node OR leaf.
     * Produces asymmetric, varied-depth trees.
     */
    DecisionTreeNode buildGrow(int maxDepth) {
        return buildGrowHelper(0, maxDepth);
    }

    private DecisionTreeNode buildGrowHelper(int currentDepth, int maxDepth) {
        if (currentDepth == maxDepth) {
            return randomLeaf();            // forced leaf at max depth
        }
        // Below max depth: randomly stop early
        if (currentDepth > 0 && rng.nextDouble() < LEAF_PROB) {
            return randomLeaf();
        }
        DecisionTreeNode node = randomDecisionTreeNode();
        node.yes  = buildGrowHelper(currentDepth + 1, maxDepth);
        node.no = buildGrowHelper(currentDepth + 1, maxDepth);
        return node;
    }

    // ── helpers ───────────────────────────────────────────────

    /**
     * Creates a random decision node with:
     *   - a random feature column
     *   - a random operator (<, >, <=, >=)
     *   - a random threshold within that feature's sensible range
     */
    private DecisionTreeNode randomDecisionTreeNode() {
        int    featureIdx = rng.nextInt(DecisionTreeNode.FEATURE_NAMES.length);
        String operator   = DecisionTreeNode.OPERATORS[rng.nextInt(DecisionTreeNode.OPERATORS.length)];
        double min        = THRESHOLD_RANGES[featureIdx][0];
        double max        = THRESHOLD_RANGES[featureIdx][1];
        // Round to 1 decimal place for readability
        double threshold  = (double) Math.round(min + rng.nextDouble() * (max - min));

        return new DecisionTreeNode(featureIdx, operator, threshold);
    }

    /**
     * Creates a leaf node: randomly class 0 or class 1.
     * (Evolution will later favour the correct label via fitness.)
     */
    private DecisionTreeNode randomLeaf() {
        return new DecisionTreeNode(rng.nextBoolean() ? 1 : 0);
    }
}
