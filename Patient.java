public class Patient {
    private int classLabel;   // 0 = no recurrence, 1 = recurrence  (TARGET)
    private int age;
    private int menopause;
    private int tumorSize;
    private int invNodes;
    private int nodeCaps;
    private int degMalig;
    private int breast;
    private int breastQuad;
    private int irradiat;
    public int prediction;

    public Patient(int classLabel, int age, int menopause, int tumorSize,int invNodes, int nodeCaps, int degMalig, int breast, int breastQuad, int irradiat) {

        this.classLabel  = classLabel;
        this.age         = age;
        this.menopause   = menopause;
        this.tumorSize   = tumorSize;
        this.invNodes    = invNodes;
        this.nodeCaps    = nodeCaps;
        this.degMalig    = degMalig;
        this.breast      = breast;
        this.breastQuad  = breastQuad;
        this.irradiat    = irradiat;
    }

    public int getClassLabel()  { return classLabel; }
    public int getAge()         { return age; }
    public int getMenopause()   { return menopause; }
    public int getTumorSize()   { return tumorSize; }
    public int getInvNodes()    { return invNodes; }
    public int getNodeCaps()    { return nodeCaps; }
    public int getDegMalig()    { return degMalig; }
    public int getBreast()      { return breast; }
    public int getBreastQuad()  { return breastQuad; }
    public int getIrradiat()    { return irradiat; }

    /**
     * Returns all feature values (everything EXCEPT the class label)
     * as a double array. Useful for passing directly into GP evaluation.
     *
     * Index mapping:
     *   0 = age          5 = degMalig
     *   1 = menopause    6 = breast
     *   2 = tumorSize    7 = breastQuad
     *   3 = invNodes     8 = irradiat
     *   4 = nodeCaps
     */
    public double[] getFeatures() {
        return new double[]{ age, menopause, tumorSize, invNodes,nodeCaps, degMalig, breast, breastQuad, irradiat};
    }

    @Override
    public String toString() {
        return String.format("Patient[class=%d, age=%d, menopause=%d, tumorSize=%d, " +
                        "invNodes=%d, nodeCaps=%d, degMalig=%d, " +
                        "breast=%d, breastQuad=%d, irradiat=%d]",
                classLabel, age, menopause, tumorSize,
                invNodes, nodeCaps, degMalig,
                breast, breastQuad, irradiat
        );
    }
}
