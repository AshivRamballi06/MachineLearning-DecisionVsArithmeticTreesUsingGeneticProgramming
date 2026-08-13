import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Assignment3 {

    public static List<Patient> loadCSV(String filePath) throws IOException {
        List<Patient> patients = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isHeader = true;
            while ((line = br.readLine()) != null) {
                if (isHeader) { isHeader = false; continue; }
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length != 10) {
                    System.err.println("Skipping malformed row: " + line);
                    continue;
                }
                try {
                    patients.add(new Patient(
                            Integer.parseInt(parts[0].trim()),
                            Integer.parseInt(parts[1].trim()),
                            Integer.parseInt(parts[2].trim()),
                            Integer.parseInt(parts[3].trim()),
                            Integer.parseInt(parts[4].trim()),
                            Integer.parseInt(parts[5].trim()),
                            Integer.parseInt(parts[6].trim()),
                            Integer.parseInt(parts[7].trim()),
                            Integer.parseInt(parts[8].trim()),
                            Integer.parseInt(parts[9].trim())
                    ));
                } catch (NumberFormatException e) {
                    System.err.println("Skipping row with non-numeric value: " + line);
                }
            }
        }
        return patients;
    }

    public static DecisionTreeNode GeneticProgramming(
            List<Patient> patients,
            long seed,
            int generations,
            int populationSize,
            int initialTreeDepth,
            int tournamentSize,
            int maxDepth,
            double crossoverRate,
            double mutationRate
    ) {
        DecisionTreeGenerator gen = new DecisionTreeGenerator(seed);

        // Generate initial population using the dynamic initial tree depth
        List<DecisionTreeNode> population = gen.rampedHalfAndHalf(populationSize, 2, initialTreeDepth);
        DecisionTreeNode bestTree    = null;
        double bestFitness = -Double.MAX_VALUE;
        Random rng = new Random(seed);

        System.out.println("============================================================");
        System.out.println("          Decision Tree GP — Training Seed=" + seed);
        System.out.println("============================================================");

        for (int curgen = 1; curgen <= generations; curgen++) {

            //  1. Evaluate every individual
            for (DecisionTreeNode tree : population) {
                if (tree == null) continue;
                Fitness(tree, patients);
            }

            //  2. Find best this generation
            DecisionTreeNode genBest     = null;
            double           genBestF1   = -Double.MAX_VALUE;
            double           fitnessSum  = 0;
            int              counted     = 0;

            // Second and third best for elitism
            DecisionTreeNode second = null, third = null;
            double secondFit = -Double.MAX_VALUE, thirdFit = -Double.MAX_VALUE;

            for (DecisionTreeNode tree : population) {
                if (tree == null) continue;
                fitnessSum += tree.fitness;
                counted++;
                if (tree.fitness > genBestF1) {
                    thirdFit = secondFit;
                    third = second;
                    secondFit = genBestF1;
                    second = genBest;
                    genBestF1 = tree.fitness;
                    genBest   = tree;
                }
            }

            // Update global best
            if (genBestF1 > bestFitness) {
                bestFitness = genBestF1;
                bestTree    = cloneTree(genBest);
            }

            if(genBest!=null){
                printGeneration(genBest,patients,String.valueOf(curgen));
            }

            //  5. Build new population
            List<DecisionTreeNode> newPopulation = new ArrayList<>();

            // Elitism: carry the global best forward unchanged
            newPopulation.add(cloneTree(bestTree));
            if (second != null) newPopulation.add(cloneTree(second));
            if (third  != null) newPopulation.add(cloneTree(third));

            while (newPopulation.size() < populationSize) {

                double r = rng.nextDouble();

                // Use dynamic Crossover Rate
                if (r < crossoverRate) {
                    //  Subtree crossover
                    DecisionTreeNode p1 = tournamentSelection(population, tournamentSize, rng);
                    DecisionTreeNode p2 = tournamentSelection(population, tournamentSize, rng);
                    DecisionTreeNode child1 = crossover(p1, p2, rng, maxDepth);
                    DecisionTreeNode child2 = crossover(p2, p1, rng, maxDepth);
                    newPopulation.add(child1);
                    if (newPopulation.size() < populationSize) newPopulation.add(child2);

                } else {
                    //  Mutation
                    DecisionTreeNode parent = tournamentSelection(population, tournamentSize, rng);
                    DecisionTreeNode mutant  = pointMutate(parent, rng);
                    if (mutant.depth() > maxDepth) mutant = cloneTree(parent); // apply max depth limit
                    newPopulation.add(mutant);
                }
            }

            population = newPopulation;
        }

        // Final report
        System.out.println("============================================================");

        return bestTree;
    }

    //  FITNESS  (F1 score with parsimony pressure)
    public static double Fitness(DecisionTreeNode tree, List<Patient> patients) {
        int tp = 0, fp = 0, fn = 0, tn = 0;

        for (Patient p : patients) {
            int actual    = p.getClassLabel();
            int predicted = tree.classify(p.getFeatures());
            if      (predicted == 1 && actual == 1) tp++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 1) fn++;
            else                                    tn++;
        }

        // Standard precision and recall — no distortion on either error type
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall    = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;

        // Standard F1
        double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

        // Balanced accuracy: average recall across BOTH classes
        // Stops the tree ignoring the minority class (recurrence = 1)
        // and stops it ignoring the majority class (no-recurrence = 0)`
        // recallClass0 = how well we identify true negatives
        double recallClass0 = (tn + fp) > 0 ? (double) tn / (tn + fp) : 0.0;
        double balancedAcc  = (recall + recallClass0) / 2.0;

        // 60% F1 + 40% balanced accuracy
        // F1 drives TP up, balancedAcc keeps both classes in check
        double fnPenalty = fn * 0.008;   // missing real recurrence = more costly
        double fpPenalty = fp * 0.002;   // false alarm = less costly

        double combined = 0.60 * f1  +  0.40 * balancedAcc  -  fnPenalty  -  fpPenalty;

        double penalty = 0.001;
        tree.fitness = combined - (tree.size() * penalty);
        return tree.fitness;
    }

    //  ACCURACY  (separate from fitness — for clean reporting)
    public static double accuracy(DecisionTreeNode tree, List<Patient> patients) {
        int correct = 0;
        for (Patient p : patients) {
            if (tree.classify(p.getFeatures()) == p.getClassLabel()) correct++;
        }
        return (double) correct / patients.size();
    }

    //  METRICS REPORT  (confusion matrix + accuracy + F1)
    public static void printMetrics(DecisionTreeNode tree,List<Patient> patients,String label) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (Patient p : patients) {
            int actual    = p.getClassLabel();
            int predicted = tree.classify(p.getFeatures());
            if      (predicted == 1 && actual == 1) tp++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 0) tn++;
            else if (predicted == 0 && actual == 1) fn++;
        }
        double acc       = (double)(tp + tn) / patients.size();
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall    = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1        = (precision + recall) > 0
                ? 2.0 * precision * recall / (precision + recall) : 0.0;

        System.out.println("------------------------------------------------------------");
        System.out.println(label + " Results:");
        System.out.printf("  Accuracy  : %.4f  (%.2f%%)%n", acc, acc * 100);
        System.out.printf("  Precision : %.4f%n", precision);
        System.out.printf("  Recall    : %.4f%n", recall);
        System.out.printf("  F-Measure : %.4f%n", f1);
        System.out.printf("  TP: %d  |  FP: %d  |  TN: %d  |  FN: %d%n", tp, fp, tn, fn);
        System.out.println("------------------------------------------------------------");
    }

    //  SINGLE-LINE METRICS REPORT
    public static void printGeneration(DecisionTreeNode tree, List<Patient> patients, String label) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (Patient p : patients) {
            int actual    = p.getClassLabel();
            int predicted = tree.classify(p.getFeatures());
            if      (predicted == 1 && actual == 1) tp++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 0) tn++;
            else if (predicted == 0 && actual == 1) fn++;
        }
        double acc       = (double)(tp + tn) / patients.size();
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall    = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1        = (precision + recall) > 0
                ? 2.0 * precision * recall / (precision + recall) : 0.0;

        // Print everything on one clean line
        System.out.printf("[%s] Acc: %.2f%% | F1: %.4f | Prec: %.4f | Rec: %.4f | TP:%d FP:%d TN:%d FN:%d%n",
                label, acc * 100, f1, precision, recall, tp, fp, tn, fn);
    }

    //  SELECTION
    public static DecisionTreeNode tournamentSelection(List<DecisionTreeNode> population,int tournamentSize, Random rng) {
        DecisionTreeNode best = null;
        for (int i = 0; i < tournamentSize; i++) {
            DecisionTreeNode competitor = population.get(rng.nextInt(population.size()));
            if (best == null || competitor.fitness > best.fitness) {
                best = competitor;
            }
        }
        return best;
    }

    //  CROSSOVER
    public static DecisionTreeNode crossover(DecisionTreeNode parent1,DecisionTreeNode parent2,Random rng, int maxDepth) {
        DecisionTreeNode offspring   = cloneTree(parent1);
        List<DecisionTreeNode> oNodes = getAllNodes(offspring);
        List<DecisionTreeNode> pNodes = getAllNodes(parent2);

        if (oNodes.isEmpty() || pNodes.isEmpty()) return offspring;

        // Avoid replacing the root (index 0) unless it's the only node
        int replaceIdx  = oNodes.size() > 1 ? 1 + rng.nextInt(oNodes.size() - 1) : 0;
        DecisionTreeNode target      = oNodes.get(replaceIdx);
        int attempts = 0;
        while (target.isLeaf && attempts < 20) {
            replaceIdx = oNodes.size() > 1 ? 1 + rng.nextInt(oNodes.size() - 1) : 0;
            target     = oNodes.get(replaceIdx);
            attempts++;
        }
        DecisionTreeNode donorBranch = cloneTree(pNodes.get(rng.nextInt(pNodes.size())));

        replaceNodeInTree(offspring, target, donorBranch);

        // Reject offspring that exceeds depth limit — return parent clone instead
        if (offspring.depth() > maxDepth) return cloneTree(parent1);

        return offspring;
    }

    //  MUTATION — Point mutation (as specified in assignment Table 1)
    /**
     * Picks one random DECISION node and randomises its condition only:
     *   - which feature to test
     *   - which operator to use (<, >, <=, >=)
     *   - what threshold to compare against
     * The tree structure (shape, depth, all children) is completely unchanged.
     */
    public static DecisionTreeNode pointMutate(DecisionTreeNode parent, Random rng) {
        DecisionTreeNode offspring = cloneTree(parent);
        List<DecisionTreeNode> allNodes = getAllNodes(offspring);

        if (allNodes.isEmpty()) return offspring;



        // Pick ANY node at random (internal or leaf)
        DecisionTreeNode target = allNodes.get(rng.nextInt(allNodes.size()));

        if (target.isLeaf) {
            // If it's a leaf, simply flip its class label (0 to 1, or 1 to 0)
            target.classLabel = (target.classLabel == 0) ? 1 : 0;
        } else {
            // If it's an internal node, randomize its condition (Feature, Operator, Threshold)
            double[][] ranges = {
                    {0, 5},   // 0 age
                    {0, 2},   // 1 menopause
                    {0, 14},  // 2 tumorSize
                    {0, 14},  // 3 invNodes
                    {0, 2},   // 4 nodeCaps
                    {1, 3},   // 5 degMalig
                    {0, 1},   // 6 breast
                    {0, 5},   // 7 breastQuad
                    {0, 2},   // 8 irradiat
            };

            int newFeature = rng.nextInt(DecisionTreeNode.FEATURE_NAMES.length);
            String newOperator = DecisionTreeNode.OPERATORS[rng.nextInt(DecisionTreeNode.OPERATORS.length)];

            double min = ranges[newFeature][0];
            double max = ranges[newFeature][1];
            double newThreshold = (double) Math.round(min + rng.nextDouble() * (max - min));

            target.featureIndex = newFeature;
            target.operator = newOperator;
            target.threshold = newThreshold;
        }

        return offspring;
    }

    //  TREE UTILITIES
    public static DecisionTreeNode cloneTree(DecisionTreeNode node) {
        if (node == null) return null;
        if (node.isLeaf) {
            DecisionTreeNode copy = new DecisionTreeNode(node.classLabel);
            copy.fitness = node.fitness;
            return copy;
        }
        DecisionTreeNode copy = new DecisionTreeNode(
                node.featureIndex, node.operator, node.threshold);
        copy.fitness = node.fitness;
        copy.yes     = cloneTree(node.yes);
        copy.no      = cloneTree(node.no);
        return copy;
    }

    public static List<DecisionTreeNode> getAllNodes(DecisionTreeNode root) {
        List<DecisionTreeNode> list = new ArrayList<>();
        if (root == null) return list;
        list.add(root);
        if (!root.isLeaf) {
            list.addAll(getAllNodes(root.yes));
            list.addAll(getAllNodes(root.no));
        }
        return list;
    }

    public static boolean replaceNodeInTree(DecisionTreeNode current,DecisionTreeNode target,DecisionTreeNode replacement) {
        if (current == null || current.isLeaf) return false;
        if (current.yes == target) { current.yes = replacement; return true; }
        if (current.no  == target) { current.no  = replacement; return true; }
        return replaceNodeInTree(current.yes, target, replacement) || replaceNodeInTree(current.no,  target, replacement);
    }

    // Helper: pure F1 for reporting/termination (no parsimony)
    public static double computeF1(DecisionTreeNode tree, List<Patient> patients) {
        int tp = 0, fp = 0, fn = 0;
        for (Patient p : patients) {
            int actual = p.getClassLabel(), pred = tree.classify(p.getFeatures());
            if      (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 1) fn++;
        }
        double prec = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double rec  = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        return (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
    }

    // Helper: balanced accuracy = (recall_class1 + recall_class0) / 2
    public static double computeBalancedAcc(DecisionTreeNode tree, List<Patient> patients) {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (Patient p : patients) {
            int actual = p.getClassLabel(), pred = tree.classify(p.getFeatures());
            if      (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 1) fn++;
            else                               tn++;
        }
        double rec0 = (tn + fp) > 0 ? (double) tn / (tn + fp) : 0.0;
        double rec1 = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        return (rec0 + rec1) / 2.0;
    }

    /** Adds a prefix string to every line — Java 8 safe (no String.indent) */
    private static String indentLines(String text, String prefix) {
        String[] lines = text.split("\n");
        StringBuilder sb = new StringBuilder();
        for (String line : lines) sb.append(prefix).append(line).append("\n");
        return sb.toString().trim();
    }

    //  MAIN

    public static void Seedtester() {
        String trainFile = "Breast_train.csv";
        String testFile  = "Breast_test.csv";

        List<Patient> trainPatients = new ArrayList<>();
        List<Patient> testPatients  = new ArrayList<>();

        // ── Load Data ──────────────────
        try {
            trainPatients = loadCSV(trainFile);
            System.out.println("Loaded " + trainPatients.size() + " training patients from: " + trainFile);
        } catch (IOException e) {
            System.err.println("Cannot read training file: " + e.getMessage());
            return;
        }

        try {
            testPatients = loadCSV(testFile);
            System.out.println("Loaded " + testPatients.size() + " test patients from: " + testFile);
        } catch (IOException e) {
            System.out.println("No test file found at: " + testFile);
        }

        // ── Summary table tracking ─────────────
        List<String> summaryResults = new ArrayList<>();
        String divider = "+--------+--------------------+------------+------------+------------+------------+";
        String header  = "| Run    | Seed               | TrainAcc%  | TrainF1    | TestAcc%   | TestF1     |";
        summaryResults.add(divider);
        summaryResults.add(header);
        summaryResults.add(divider);

        // ── Setup Interactive Scanner ──────────────────
        java.util.Scanner scanner = new java.util.Scanner(System.in);
        Random seedGen = new Random();
        int runCount = 0;

        System.out.println("\n============================================================");
        System.out.println("  Interactive Seed Runner");
        System.out.println("  Press ENTER to run a new seed  |  Type 'C' to stop");
        System.out.println("============================================================");

        while (true) {
            System.out.print("\nPress ENTER to run a new seed (or 'C' to finish): ");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("c")) {
                System.out.println("\nStopping seed search...");
                break;
            }

            // Generate a seed with a variable length (between 1 and 15 digits long)
            int digits = 1 + seedGen.nextInt(15);
            long maxLimit = (long) Math.pow(10, digits);
            long seed = (long) (seedGen.nextDouble() * maxLimit);

            runCount++;
            System.out.println("\n>>> Run #" + runCount + " | Seed: " + seed);

            // Run Genetic Programming
            DecisionTreeNode bestTree = GeneticProgramming(
                    trainPatients, seed,
                    100, 200, 4, 5, 5, 0.70, 0.30
            );

            // Calculate Metrics
            double trainAcc = accuracy(bestTree, trainPatients);
            double trainF1  = computeF1(bestTree, trainPatients);
            double testAcc  = testPatients.isEmpty() ? 0 : accuracy(bestTree, testPatients);
            double testF1   = testPatients.isEmpty() ? 0 : computeF1(bestTree, testPatients);


            // Print full metrics to console for this specific run
            printMetrics(bestTree, trainPatients, "Training");
            if (!testPatients.isEmpty()) {
                printMetrics(bestTree, testPatients, "Test");
            }

            // Save results to summary
            summaryResults.add(String.format("| %-6d | %-18s | %-10.2f | %-10.4f | %-10.2f | %-10.4f |",
                    runCount, String.valueOf(seed), trainAcc * 100, trainF1, testAcc * 100, testF1));
        }

        scanner.close();

        // ── Final summary ──────────────────────────────────────
        System.out.println("\n==================================================");
        System.out.println("          FINAL SEED SUMMARY");
        System.out.println("==================================================");
        for (String row : summaryResults) {
            System.out.println(row);
        }
        System.out.println("==================================================");
        System.out.println("Total runs: " + runCount);
    }

    // ═══════════════════════════════════════════════════════════
    //  MAIN  —  unified entry point for both GP algorithms
    // ═══════════════════════════════════════════════════════════
    public static void main(String[] args) {
        java.util.Scanner scanner = new java.util.Scanner(System.in);

        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       Genetic Programming — Breast Cancer Classifier     ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.println("║  1. Decision Tree GP                                     ║");
        System.out.println("║  2. Arithmetic GP                                        ║");
        System.out.println("║  3. Both (runs sequentially, prints combined table)      ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.print("Choose algorithm [1/2/3]: ");
        String choice = scanner.nextLine().trim();

        // ── Results holders for Table 2 ────────────────────────
        double dtTrainAcc = 0, dtTestAcc = 0, dtTestF1 = 0, dtRuntime = 0;
        double arTrainAcc = 0, arTestAcc = 0, arTestF1 = 0, arRuntime = 0;
        boolean ranDT = false, ranAR = false;

        // ═══════════════════════════════════════════════════════
        //  DECISION TREE GP — own file + seed menu
        // ═══════════════════════════════════════════════════════
        if (choice.equals("1") || choice.equals("3")) {
            System.out.println();
            System.out.println("════════════════════════════════════════════════");
            System.out.println("  Decision Tree GP — Parameters");
            System.out.println("════════════════════════════════════════════════");

            System.out.print("  Training file         [Breast_train.csv]: ");
            String dtTrainFile = scanner.nextLine().trim();
            if (dtTrainFile.isEmpty()) dtTrainFile = "Breast_train.csv";

            System.out.print("  Test file             [Breast_test.csv]:  ");
            String dtTestFile = scanner.nextLine().trim();
            if (dtTestFile.isEmpty()) dtTestFile = "Breast_test.csv";

            System.out.print("  Seed                  [56982693090]:               ");
            String dtSeedIn = scanner.nextLine().trim();
            long dtSeed = dtSeedIn.isEmpty() ? 56982693090L : Long.parseLong(dtSeedIn);

            System.out.print("  Generations           [100]:              ");
            String genIn = scanner.nextLine().trim();
            int dtGenerations = genIn.isEmpty() ? 100 : Integer.parseInt(genIn);

            System.out.print("  Population size       [200]:              ");
            String popIn = scanner.nextLine().trim();
            int dtPopSize = popIn.isEmpty() ? 200 : Integer.parseInt(popIn);

            System.out.print("  Initial tree depth     2-[7]:                ");
            String depthIn = scanner.nextLine().trim();
            int dtInitialDepth = depthIn.isEmpty() ? 7 : Integer.parseInt(depthIn);

            System.out.print("  Tournament size       [7]:                ");
            String tournIn = scanner.nextLine().trim();
            int dtTournamentSize = tournIn.isEmpty() ? 7 : Integer.parseInt(tournIn);

            System.out.print("  Max tree depth        [5]:                ");
            String maxDepthIn = scanner.nextLine().trim();
            int dtMaxDepth = maxDepthIn.isEmpty() ? 5 : Integer.parseInt(maxDepthIn);

            System.out.print("  Crossover rate %      [85]:               ");
            String cxIn = scanner.nextLine().trim();
            double dtCrossoverRate = (cxIn.isEmpty() ? 85.0 : Double.parseDouble(cxIn)) / 100.0;

            System.out.print("  Mutation rate %       [15]:               ");
            String mutIn = scanner.nextLine().trim();
            double dtMutationRate = (mutIn.isEmpty() ? 15.0 : Double.parseDouble(mutIn)) / 100.0;

            System.out.println();
            System.out.println("  Confirmed:");
            System.out.println("    Train       : " + dtTrainFile);
            System.out.println("    Test        : " + dtTestFile);
            System.out.println("    Seed        : " + dtSeed);
            System.out.println("    Generations : " + dtGenerations);
            System.out.println("    Population  : " + dtPopSize);
            System.out.println("    Init Depth  : " + dtInitialDepth);
            System.out.println("    Tournament  : " + dtTournamentSize);
            System.out.println("    Max Depth   : " + dtMaxDepth);
            System.out.println("    Crossover   : " + (int)(dtCrossoverRate * 100) + "%");
            System.out.println("    Mutation    : " + (int)(dtMutationRate * 100) + "%");
            System.out.println();

            List<Patient> trainPatients = new ArrayList<>();
            List<Patient> testPatients  = new ArrayList<>();

            try {
                trainPatients = loadCSV(dtTrainFile);
                System.out.println("  Loaded " + trainPatients.size() + " training patients.");
            } catch (IOException e) {
                System.err.println("  Cannot read training file: " + e.getMessage());
                scanner.close();
                return;
            }
            try {
                testPatients = loadCSV(dtTestFile);
                System.out.println("  Loaded " + testPatients.size() + " test patients.");
            } catch (IOException e) {
                System.out.println("  No test file found — test evaluation skipped.");
            }

            System.out.println();
            long dtStart = System.currentTimeMillis();

            // Call GeneticProgramming with the new dynamic parameters
            DecisionTreeNode bestTree = GeneticProgramming(
                    trainPatients, dtSeed, dtGenerations, dtPopSize,
                    dtInitialDepth, dtTournamentSize, dtMaxDepth,
                    dtCrossoverRate, dtMutationRate
            );

            dtRuntime = (System.currentTimeMillis() - dtStart) / 1000.0;

            dtTrainAcc = accuracy(bestTree, trainPatients) * 100;
            dtTestAcc  = testPatients.isEmpty() ? 0 : accuracy(bestTree, testPatients) * 100;
            dtTestF1   = testPatients.isEmpty() ? 0 : computeF1(bestTree, testPatients);

            System.out.println("Best evolved Decision Tree:");
            System.out.println(bestTree);
            printMetrics(bestTree, trainPatients, "Decision Tree — Training");
            if (!testPatients.isEmpty()) printMetrics(bestTree, testPatients, "Decision Tree — Test");
            System.out.println("  Seed used: " + dtSeed + "  (reuse to replicate)");

            ranDT = true;
        }

        // ═══════════════════════════════════════════════════════
        //  ARITHMETIC GP — own file + seed menu
        // ═══════════════════════════════════════════════════════
        if (choice.equals("2") || choice.equals("3")) {
            System.out.println();
            System.out.println("════════════════════════════════════════════════");
            System.out.println("  Arithmetic GP — Parameters");
            System.out.println("════════════════════════════════════════════════");

            System.out.print("  Training file         [Breast_train.csv]: ");
            String arTrainFile = scanner.nextLine().trim();
            if (arTrainFile.isEmpty()) arTrainFile = "Breast_train.csv";

            System.out.print("  Test file             [Breast_test.csv]:  ");
            String arTestFile = scanner.nextLine().trim();
            if (arTestFile.isEmpty()) arTestFile = "Breast_test.csv";

            System.out.print("  Seed                  [500]:               ");
            String arSeedIn = scanner.nextLine().trim();
            long arSeed = arSeedIn.isEmpty() ? 500L : Long.parseLong(arSeedIn);

            System.out.print("  Number of runs        [1]:                ");
            String runsIn = scanner.nextLine().trim();
            int numberOfRuns = runsIn.isEmpty() ? 1 : Integer.parseInt(runsIn);

            System.out.print("  Initial tree depth    [3]:                ");
            String depthIn = scanner.nextLine().trim();
            int initialTreeDepth = depthIn.isEmpty() ? 3 : Integer.parseInt(depthIn);

            System.out.print("  Tournament size       [15]:                ");
            String tournIn = scanner.nextLine().trim();
            int tournamentSize = tournIn.isEmpty() ? 15 : Integer.parseInt(tournIn);

            System.out.print("  Max offspring depth   [5]:                ");
            String offIn = scanner.nextLine().trim();
            int maxOffspringDepth = offIn.isEmpty() ? 5 : Integer.parseInt(offIn);

            System.out.print("  Crossover rate %      [85]:               ");
            String cxIn = scanner.nextLine().trim();
            double crossoverRate = (cxIn.isEmpty() ? 85.0 : Double.parseDouble(cxIn)) / 100.0;

            System.out.print("  Mutation rate %       [15]:               ");
            String mutIn = scanner.nextLine().trim();
            double mutationRate = (mutIn.isEmpty() ? 15.0 : Double.parseDouble(mutIn)) / 100.0;

            System.out.print("  Mutation offspring depth [7]:             ");
            String modIn = scanner.nextLine().trim();
            int mutationOffspringDepth = modIn.isEmpty() ? 7 : Integer.parseInt(modIn);

            if (crossoverRate + mutationRate > 1.0) {
                System.out.println("  ERROR: crossover + mutation rates must not exceed 100%.");
                scanner.close();
                return;
            }

            System.out.println();
            System.out.println("  Confirmed:");
            System.out.println("    Train       : " + arTrainFile);
            System.out.println("    Test        : " + arTestFile);
            System.out.println("    Seed        : " + arSeed);
            System.out.println("    Runs        : " + numberOfRuns);
            System.out.println("    Tree depth  : " + initialTreeDepth);
            System.out.println("    Tournament  : " + tournamentSize);
            System.out.println("    Max depth   : " + maxOffspringDepth);
            System.out.println("    Crossover   : " + (int)(crossoverRate * 100) + "%");
            System.out.println("    Mutation    : " + (int)(mutationRate  * 100) + "%");
            System.out.println("    Mut depth   : " + mutationOffspringDepth);
            System.out.println();

            try {
                ArithmeticGPResult result = runArithmeticGP(
                        arTrainFile, arTestFile, arSeed, numberOfRuns,
                        initialTreeDepth, tournamentSize, maxOffspringDepth,
                        crossoverRate, mutationRate, mutationOffspringDepth
                );
                arTrainAcc = result.trainAccuracy * 100;
                arTestAcc  = result.testAccuracy  * 100;
                arTestF1   = result.testFMeasure;
                arRuntime  = result.runtimeSeconds;
                ranAR = true;
                System.out.println("  Seed used: " + arSeed + "  (reuse to replicate)");
            } catch (Exception e) {
                System.out.println("  Arithmetic GP error: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // ═══════════════════════════════════════════════════════
        //  TABLE 2  —  Comparison of Classification Performance
        // ═══════════════════════════════════════════════════════
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║         Table 2: Comparison of Classification Performance            ║");
        System.out.println("╠═════════════════╦════════════╦══════════╦════════════╦═══════════════╣");
        System.out.println("║ Algorithm       ║ Train (%)  ║ Test (%) ║ F-Measure  ║ Runtime (s)   ║");
        System.out.println("╠═════════════════╬════════════╬══════════╬════════════╬═══════════════╣");
        if (ranDT) {
            System.out.printf("║ %-15s ║ %-10.2f ║ %-8.2f ║ %-10.4f ║ %-13.2f ║%n",
                    "Decision Tree", dtTrainAcc, dtTestAcc, dtTestF1, dtRuntime);
        }
        if (ranAR) {
            System.out.printf("║ %-15s ║ %-10.2f ║ %-8.2f ║ %-10.4f ║ %-13.2f ║%n",
                    "GP Classifier", arTrainAcc, arTestAcc, arTestF1, arRuntime);
        }
        System.out.println("╚═════════════════╩════════════╩══════════╩════════════╩═══════════════╝");

        scanner.close();

    }

    // ═══════════════════════════════════════════════════════════
    //  ARITHMETIC GP RUNNER
    // ═══════════════════════════════════════════════════════════
    private static ArithmeticGPResult runArithmeticGP(
            String trainFile, String testFile, long baseSeed, int numberOfRuns,
            int initialTreeDepth, int tournamentSize, int maxOffspringDepth,
            double crossoverRate, double mutationRate, int mutationOffspringDepth
    ) throws Exception {

        ArithmeticGPTrain.runGP(
                trainFile, baseSeed, numberOfRuns, initialTreeDepth,
                tournamentSize, maxOffspringDepth, crossoverRate,
                mutationRate, mutationOffspringDepth
        );

        ArithmeticGPResult result = new ArithmeticGPResult();
        result.trainAccuracy  = ArithmeticGPTrain.lastTrainAccuracy;
        result.runtimeSeconds = ArithmeticGPTrain.lastRuntimeSeconds;

        try {
            ArithmeticGPTest.TestResult tr = ArithmeticGPTest.runTest(testFile);
            result.testAccuracy  = tr.accuracy;
            result.testFMeasure  = tr.fMeasure;
            result.runtimeSeconds = ArithmeticGPTrain.lastRuntimeSeconds + tr.runtimeSeconds;
        } catch (Exception e) {
            System.out.println("Could not evaluate arithmetic GP on test set: " + e.getMessage());
        }

        return result;
    }

    private static class ArithmeticGPResult {
        double trainAccuracy;
        double testAccuracy;
        double testFMeasure;
        double runtimeSeconds;
    }
}