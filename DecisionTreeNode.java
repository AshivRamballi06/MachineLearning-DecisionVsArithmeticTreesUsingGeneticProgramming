public class DecisionTreeNode {

    public static final String[] FEATURE_NAMES = {
            "age", "menopause", "tumorSize", "invNodes",
            "nodeCaps", "degMalig", "breast", "breastQuad", "irradiat"
    };

    // ── comparison operators available to evolved conditions ──
    public static final String[] OPERATORS = {"<", ">", "<=", ">="};

    boolean isLeaf;         // true → this node IS a class label

    int classLabel;         // 0 = no recurrence, 1 = recurrence

    double fitness=-Double.MAX_VALUE;

    int    featureIndex;    // which column to test (0–8)
    String operator;        // "<", ">", "<=", ">="
    double threshold;       // value to compare against

    public DecisionTreeNode yes;
    public DecisionTreeNode no;

    // ── leaf constructor ──────────────────────────────────────
    DecisionTreeNode(int classLabel) {
        this.isLeaf     = true;
        this.classLabel = classLabel;
    }

    // ── decision constructor ──────────────────────────────────
    DecisionTreeNode(int featureIndex, String operator, double threshold) {
        this.isLeaf        = false;
        this.featureIndex  = featureIndex;
        this.operator      = operator;
        this.threshold     = threshold;
    }

    boolean conditionHolds(double[] features) {
        double val = features[featureIndex];
        switch (operator) {
            case "<":  return val <  threshold;
            case ">":  return val >  threshold;
            case "<=": return val <= threshold;
            case ">=": return val >= threshold;
            default:   return false;
        }
    }

    // ── classify one patient ──────────────────────────────────
    /**
     * Walk the tree by following yes/no branches until a leaf.
     * @param features  double[] from Patient.getFeatures()
     * @return          0 or 1
     */
    int classify(double[] features) {
        if (isLeaf) return classLabel;

        if (conditionHolds(features)) {
            return yes.classify(features);   // condition yes  → left
        } else {
            return no.classify(features);  // condition no → right
        }
    }

    // ── tree depth ────────────────────────────────────────────
    int depth() {
        if (isLeaf) return 0;
        return 1 + Math.max(yes.depth(), no.depth());
    }

    // ── count nodes (useful for bloat control later) ──────────
    int size() {
        if (isLeaf) return 1;
        return 1 + yes.size() + no.size();
    }

    // ── pretty print ──────────────────────────────────────────
    @Override
    public String toString() {
        return toStringHelper(0);
    }

    private String toStringHelper(int indent) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < indent; i++) sb.append("  ");
        String pad = sb.toString();
        if (isLeaf) {
            String label = (classLabel == 1) ? "RECURRENCE" : "NO-RECURRENCE";
            return pad + "[CLASS: " + label + "]";
        }
        String condition = String.format("if %s %s %.2f",
                FEATURE_NAMES[featureIndex], operator, threshold);
        return pad + "(" + condition + ")\n"
                + pad + " YES  → \n" + yes.toStringHelper(indent + 2) + "\n"
                + pad + " NO → \n" + no.toStringHelper(indent + 2);
    }
}
