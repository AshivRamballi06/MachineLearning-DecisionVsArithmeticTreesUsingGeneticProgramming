import java.io.*;
import java.util.*;

public class ArithmeticGPTrain {

    private static final int POPULATION_SIZE = 200;
    private static final int MAX_GENERATIONS = 100;

    private static final int CONST_MIN = 0;
    private static final int CONST_MAX = 10;

    private static final String BEST_MODEL_FILE = "best_arithmetic_model.txt";
    private static final String TRAINING_SUMMARY_FILE = "arithmetic_training_summary.csv";
    private static final String MODEL_FOLDER = "arithmetic_models";

    private static final String[] FEATURE_NAMES = {
            "age",
            "menopause",
            "tumor_size",
            "inv_nodes",
            "node_caps",
            "degree_of_malignancy",
            "breast",
            "breast_quad",
            "irradiat"
    };

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        printFixedSettings();

        System.out.print("Enter seed: ");
        long baseSeed = Long.parseLong(input.nextLine().trim());

        System.out.print("Enter training CSV filepath: ");
        String trainingPath = input.nextLine().trim();

        System.out.print("Enter number of runs: ");
        int numberOfRuns = Integer.parseInt(input.nextLine().trim());

        System.out.print("Enter initial tree depth: ");
        int initialTreeDepth = Integer.parseInt(input.nextLine().trim());

        System.out.print("Enter tournament size: ");
        int tournamentSize = Integer.parseInt(input.nextLine().trim());

        System.out.print("Enter max offspring depth: ");
        int maxOffspringDepth = Integer.parseInt(input.nextLine().trim());

        System.out.print("Enter crossover rate in percent: ");
        double crossoverRate = Double.parseDouble(input.nextLine().trim()) / 100.0;

        System.out.print("Enter mutation rate in percent: ");
        double mutationRate = Double.parseDouble(input.nextLine().trim()) / 100.0;

        System.out.print("Enter mutation offspring depth: ");
        int mutationOffspringDepth = Integer.parseInt(input.nextLine().trim());

        if (crossoverRate < 0 || mutationRate < 0 || crossoverRate + mutationRate > 1.0) {
            System.out.println("ERROR: Crossover rate + mutation rate must be between 0 and 100 percent.");
            return;
        }

        try {
            DataSet trainingData = loadCsv(trainingPath);

            File folder = new File(MODEL_FOLDER);
            if (!folder.exists()) {
                folder.mkdirs();
            }

            BufferedWriter summaryWriter = new BufferedWriter(new FileWriter(TRAINING_SUMMARY_FILE));
            summaryWriter.write("run,seed,generation,best_accuracy_percent,best_fmeasure,tree_size,expression,expression_with_column_names");
            summaryWriter.newLine();

            Individual overallBest = null;
            int overallBestRun = -1;
            int overallBestGeneration = -1;
            long overallBestSeed = baseSeed;

            long startTime = System.currentTimeMillis();

            for (int run = 1; run <= numberOfRuns; run++) {
                long runSeed = baseSeed + (run - 1);
                Random random = new Random(runSeed);

                System.out.println();
                System.out.println("====================================================");
                System.out.println("RUN " + run + " OF " + numberOfRuns);
                System.out.println("Seed for this run: " + runSeed);
                System.out.println("====================================================");

                ArrayList<Individual> population = createInitialPopulation(
                        random,
                        trainingData.featureCount,
                        initialTreeDepth
                );

                Individual runBest = null;
                int runBestGeneration = -1;

                for (int generation = 1; generation <= MAX_GENERATIONS; generation++) {
                    evaluatePopulation(population, trainingData);

                    Individual generationBest = getBestIndividual(population);

                    if (runBest == null || isBetter(generationBest, runBest)) {
                        runBest = generationBest.copy();
                        runBestGeneration = generation;
                    }

                    if (overallBest == null || isBetter(generationBest, overallBest)) {
                        overallBest = generationBest.copy();
                        overallBestRun = run;
                        overallBestGeneration = generation;
                        overallBestSeed = runSeed;
                    }

                    System.out.println(
                            "Run " + run +
                                    " | Generation " + generation +
                                    " | Best Accuracy: " + formatPercent(generationBest.metrics.accuracy) +
                                    " | F-measure: " + formatDouble(generationBest.metrics.fMeasure) +
                                    " | Tree Size: " + generationBest.root.size() +
                                    " | Expression: " + generationBest.root.toExpression()
                    );

                    summaryWriter.write(
                            run + "," +
                                    runSeed + "," +
                                    generation + "," +
                                    formatDouble(generationBest.metrics.accuracy * 100.0) + "," +
                                    formatDouble(generationBest.metrics.fMeasure) + "," +
                                    generationBest.root.size() + "," +
                                    csvEscape(generationBest.root.toExpression()) + "," +
                                    csvEscape(generationBest.root.toExpressionWithNames())
                    );
                    summaryWriter.newLine();

                    if (generation < MAX_GENERATIONS) {
                        population = createNextGeneration(
                                population,
                                random,
                                trainingData.featureCount,
                                tournamentSize,
                                maxOffspringDepth,
                                mutationOffspringDepth,
                                crossoverRate,
                                mutationRate
                        );
                    }
                }

                String perRunModelFile = MODEL_FOLDER + File.separator +
                        "arithmetic_model_run_" + run + "_seed_" + runSeed + ".txt";

                saveModel(
                        perRunModelFile,
                        runBest,
                        run,
                        runSeed,
                        runBestGeneration,
                        "Best model for run " + run
                );

                System.out.println();
                System.out.println("Best model for run " + run + " saved to: " + perRunModelFile);
                System.out.println("Run best generation: " + runBestGeneration);
                System.out.println("Run best accuracy: " + formatPercent(runBest.metrics.accuracy));
                System.out.println("Run best F-measure: " + formatDouble(runBest.metrics.fMeasure));
            }

            summaryWriter.close();

            long endTime = System.currentTimeMillis();
            double runtimeSeconds = (endTime - startTime) / 1000.0;

            saveModel(
                    BEST_MODEL_FILE,
                    overallBest,
                    overallBestRun,
                    overallBestSeed,
                    overallBestGeneration,
                    "Overall best arithmetic GP model"
            );

            System.out.println();
            System.out.println("====================================================");
            System.out.println("TRAINING COMPLETE");
            System.out.println("====================================================");
            System.out.println("Overall best model saved to: " + BEST_MODEL_FILE);
            System.out.println("Training summary saved to: " + TRAINING_SUMMARY_FILE);
            System.out.println("Overall best run: " + overallBestRun);
            System.out.println("Overall best seed: " + overallBestSeed);
            System.out.println("Overall best generation: " + overallBestGeneration);
            System.out.println("Overall best training accuracy: " + formatPercent(overallBest.metrics.accuracy));
            System.out.println("Overall best F-measure: " + formatDouble(overallBest.metrics.fMeasure));
            System.out.println("Overall best tree size: " + overallBest.root.size());

            System.out.println();
            System.out.println("Overall best expression:");
            System.out.println(overallBest.root.toExpression());

            System.out.println();
            System.out.println("Overall best expression with column names:");
            System.out.println(overallBest.root.toExpressionWithNames());

            System.out.println();
            System.out.println("Overall best arithmetic tree:");
            System.out.println(overallBest.root.toVisualTree());

            System.out.println("Runtime seconds: " + formatDouble(runtimeSeconds));
            System.out.println("====================================================");

        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void printFixedSettings() {
        System.out.println("===== ARITHMETIC GP CLASSIFIER SETTINGS =====");
        System.out.println();
        System.out.println("Population size: " + POPULATION_SIZE);
        System.out.println("Initial tree generation: ramped half-and-half");
        System.out.println("Selection method: tournament selection");
        System.out.println("Mutation type: point mutation");
        System.out.println("Fitness function: accuracy");
        System.out.println("Maximum generations: " + MAX_GENERATIONS);
        System.out.println("Function set: +, -, *, protected division");
        System.out.println("Constant range: " + CONST_MIN + " to " + CONST_MAX);
        System.out.println("Classification rule: expression >= 0 predicts class 1, otherwise class 0");
        System.out.println();
        System.out.println("Feature key:");
        System.out.println("CSV column 0 = class label / actual class");
        System.out.println("CSV column 1 = x0 = age");
        System.out.println("CSV column 2 = x1 = menopause");
        System.out.println("CSV column 3 = x2 = tumor_size");
        System.out.println("CSV column 4 = x3 = inv_nodes");
        System.out.println("CSV column 5 = x4 = node_caps");
        System.out.println("CSV column 6 = x5 = degree_of_malignancy");
        System.out.println("CSV column 7 = x6 = breast");
        System.out.println("CSV column 8 = x7 = breast_quad");
        System.out.println("CSV column 9 = x8 = irradiat");
        System.out.println();
        System.out.println("=============================================");
        System.out.println();
    }

    private static DataSet loadCsv(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));

        String header = reader.readLine();
        if (header == null) {
            reader.close();
            throw new IOException("CSV file is empty.");
        }

        ArrayList<double[]> features = new ArrayList<double[]>();
        ArrayList<Integer> labels = new ArrayList<Integer>();

        String line;
        int lineNumber = 1;

        while ((line = reader.readLine()) != null) {
            lineNumber++;

            if (line.trim().length() == 0) {
                continue;
            }

            String[] parts = line.split(",");

            if (parts.length < 2) {
                reader.close();
                throw new IOException("Invalid CSV row at line " + lineNumber);
            }

            int label = Integer.parseInt(parts[0].trim());
            double[] rowFeatures = new double[parts.length - 1];

            for (int i = 1; i < parts.length; i++) {
                rowFeatures[i - 1] = Double.parseDouble(parts[i].trim());
            }

            labels.add(Integer.valueOf(label));
            features.add(rowFeatures);
        }

        reader.close();

        if (features.size() == 0) {
            throw new IOException("No data rows found in CSV file.");
        }

        int featureCount = features.get(0).length;

        double[][] featureArray = new double[features.size()][featureCount];
        int[] labelArray = new int[labels.size()];

        for (int i = 0; i < features.size(); i++) {
            if (features.get(i).length != featureCount) {
                throw new IOException("Inconsistent number of columns in CSV file.");
            }

            featureArray[i] = features.get(i);
            labelArray[i] = labels.get(i).intValue();
        }

        return new DataSet(featureArray, labelArray, featureCount);
    }

    private static ArrayList<Individual> createInitialPopulation(Random random, int featureCount, int initialTreeDepth) {
        ArrayList<Individual> population = new ArrayList<Individual>();

        int minDepth = 2;
        int depthRange = Math.max(1, initialTreeDepth - minDepth + 1);

        for (int i = 0; i < POPULATION_SIZE; i++) {
            int depth = minDepth + (i % depthRange);

            Node root;

            if (i < POPULATION_SIZE / 2) {
                root = generateFullTree(random, featureCount, depth);
            } else {
                root = generateGrowTree(random, featureCount, depth);
            }

            population.add(new Individual(root));
        }

        return population;
    }

    private static Node generateFullTree(Random random, int featureCount, int depth) {
        if (depth <= 1) {
            return randomTerminal(random, featureCount);
        }

        return new FunctionNode(
                randomFunction(random),
                generateFullTree(random, featureCount, depth - 1),
                generateFullTree(random, featureCount, depth - 1)
        );
    }

    private static Node generateGrowTree(Random random, int featureCount, int depth) {
        if (depth <= 1) {
            return randomTerminal(random, featureCount);
        }

        if (depth < 4 && random.nextDouble() < 0.35) {
            return randomTerminal(random, featureCount);
        }

        return new FunctionNode(
                randomFunction(random),
                generateGrowTree(random, featureCount, depth - 1),
                generateGrowTree(random, featureCount, depth - 1)
        );
    }

    private static Node randomTerminal(Random random, int featureCount) {
        if (random.nextBoolean()) {
            return new FeatureNode(random.nextInt(featureCount));
        }

        int value = CONST_MIN + random.nextInt(CONST_MAX - CONST_MIN + 1);
        return new ConstantNode(value);
    }

    private static String randomFunction(Random random) {
        int choice = random.nextInt(4);

        if (choice == 0) {
            return "ADD";
        } else if (choice == 1) {
            return "SUB";
        } else if (choice == 2) {
            return "MUL";
        } else {
            return "DIV";
        }
    }

    private static void evaluatePopulation(ArrayList<Individual> population, DataSet dataSet) {
        for (int i = 0; i < population.size(); i++) {
            population.get(i).metrics = evaluate(population.get(i).root, dataSet);
        }
    }

    private static Metrics evaluate(Node root, DataSet dataSet) {
        int correct = 0;
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < dataSet.features.length; i++) {
            double output = root.evaluate(dataSet.features[i]);
            int predicted = output >= 0.0 ? 1 : 0;
            int actual = dataSet.labels[i];

            if (predicted == actual) {
                correct++;
            }

            if (predicted == 1 && actual == 1) {
                tp++;
            } else if (predicted == 0 && actual == 0) {
                tn++;
            } else if (predicted == 1 && actual == 0) {
                fp++;
            } else if (predicted == 0 && actual == 1) {
                fn++;
            }
        }

        double accuracy = correct / (double) dataSet.labels.length;

        double precision;
        if (tp + fp == 0) {
            precision = 0.0;
        } else {
            precision = tp / (double) (tp + fp);
        }

        double recall;
        if (tp + fn == 0) {
            recall = 0.0;
        } else {
            recall = tp / (double) (tp + fn);
        }

        double fMeasure;
        if (precision + recall == 0.0) {
            fMeasure = 0.0;
        } else {
            fMeasure = 2.0 * precision * recall / (precision + recall);
        }

        Metrics metrics = new Metrics();
        metrics.accuracy = accuracy;
        metrics.precision = precision;
        metrics.recall = recall;
        metrics.fMeasure = fMeasure;
        metrics.tp = tp;
        metrics.tn = tn;
        metrics.fp = fp;
        metrics.fn = fn;

        return metrics;
    }

    private static Individual getBestIndividual(ArrayList<Individual> population) {
        Individual best = population.get(0);

        for (int i = 1; i < population.size(); i++) {
            Individual candidate = population.get(i);

            if (isBetter(candidate, best)) {
                best = candidate;
            }
        }

        return best;
    }

    private static boolean isBetter(Individual a, Individual b) {
        if (a.metrics.accuracy > b.metrics.accuracy) {
            return true;
        }

        if (a.metrics.accuracy < b.metrics.accuracy) {
            return false;
        }

        if (a.metrics.fMeasure > b.metrics.fMeasure) {
            return true;
        }

        if (a.metrics.fMeasure < b.metrics.fMeasure) {
            return false;
        }

        return a.root.size() < b.root.size();
    }

    private static ArrayList<Individual> createNextGeneration(
            ArrayList<Individual> population,
            Random random,
            int featureCount,
            int tournamentSize,
            int maxOffspringDepth,
            int mutationOffspringDepth,
            double crossoverRate,
            double mutationRate
    ) {
        ArrayList<Individual> nextPopulation = new ArrayList<Individual>();

        while (nextPopulation.size() < POPULATION_SIZE) {
            double choice = random.nextDouble();

            Node childRoot;

            if (choice < crossoverRate) {
                Individual parent1 = tournamentSelection(population, random, tournamentSize);
                Individual parent2 = tournamentSelection(population, random, tournamentSize);

                childRoot = subtreeCrossover(
                        parent1.root,
                        parent2.root,
                        random,
                        maxOffspringDepth
                );

            } else if (choice < crossoverRate + mutationRate) {
                Individual parent = tournamentSelection(population, random, tournamentSize);

                childRoot = pointMutation(
                        parent.root,
                        random,
                        featureCount,
                        mutationOffspringDepth,
                        maxOffspringDepth
                );

            } else {
                Individual parent = tournamentSelection(population, random, tournamentSize);
                childRoot = parent.root.copy();
            }

            nextPopulation.add(new Individual(childRoot));
        }

        return nextPopulation;
    }

    private static Individual tournamentSelection(ArrayList<Individual> population, Random random, int tournamentSize) {
        Individual best = null;

        for (int i = 0; i < tournamentSize; i++) {
            Individual candidate = population.get(random.nextInt(population.size()));

            if (best == null || isBetter(candidate, best)) {
                best = candidate;
            }
        }

        return best;
    }

    private static Node subtreeCrossover(Node parent1, Node parent2, Random random, int maxOffspringDepth) {
        for (int attempt = 0; attempt < 10; attempt++) {
            Node child = parent1.copy();

            ArrayList<Integer> path1 = randomPath(child, random);
            ArrayList<Integer> path2 = randomPath(parent2, random);

            Node donorSubtree = getSubtree(parent2, path2).copy();
            child = replaceSubtree(child, path1, donorSubtree);

            if (child.depth() <= maxOffspringDepth) {
                return child;
            }
        }

        return parent1.copy();
    }

    private static Node pointMutation(
            Node parent,
            Random random,
            int featureCount,
            int mutationOffspringDepth,
            int maxOffspringDepth
    ) {
        for (int attempt = 0; attempt < 10; attempt++) {
            Node child = parent.copy();

            ArrayList<Integer> path = randomPath(child, random);
            Node selected = getSubtree(child, path);

            Node replacement;

            if (selected instanceof FunctionNode && random.nextBoolean()) {
                FunctionNode functionNode = (FunctionNode) selected;
                String newOperator = randomFunction(random);

                while (newOperator.equals(functionNode.operator)) {
                    newOperator = randomFunction(random);
                }

                replacement = new FunctionNode(
                        newOperator,
                        functionNode.left.copy(),
                        functionNode.right.copy()
                );
            } else {
                replacement = generateGrowTree(random, featureCount, mutationOffspringDepth);
            }

            child = replaceSubtree(child, path, replacement);

            if (child.depth() <= maxOffspringDepth) {
                return child;
            }
        }

        return parent.copy();
    }

    // Called programmatically from Assignment3.main
    public static double lastTrainAccuracy = 0;
    public static double lastTrainFMeasure = 0;
    public static double lastRuntimeSeconds = 0;

    public static void runGP(
            String trainingPath, long baseSeed, int numberOfRuns,
            int initialTreeDepth, int tournamentSize, int maxOffspringDepth,
            double crossoverRate, double mutationRate, int mutationOffspringDepth
    ) throws Exception {
        File folder = new File(MODEL_FOLDER);
        if (!folder.exists()) folder.mkdirs();

        BufferedWriter summaryWriter = new BufferedWriter(new FileWriter(TRAINING_SUMMARY_FILE));
        summaryWriter.write("run,seed,generation,best_accuracy_percent,best_fmeasure,tree_size,expression,expression_with_column_names");
        summaryWriter.newLine();

        DataSet trainingData = loadCsv(trainingPath);

        Individual overallBest = null;
        int overallBestRun = -1;
        int overallBestGeneration = -1;
        long overallBestSeed = baseSeed;
        long startTime = System.currentTimeMillis();

        for (int run = 1; run <= numberOfRuns; run++) {
            long runSeed = baseSeed + (run - 1);
            Random random = new Random(runSeed);

            System.out.println("====================================================");
            System.out.println("RUN " + run + " OF " + numberOfRuns + " | Seed: " + runSeed);
            System.out.println("====================================================");

            ArrayList<Individual> population = createInitialPopulation(
                    random, trainingData.featureCount, initialTreeDepth);

            Individual runBest = null;
            int runBestGeneration = -1;

            for (int generation = 1; generation <= MAX_GENERATIONS; generation++) {
                evaluatePopulation(population, trainingData);
                Individual generationBest = getBestIndividual(population);

                if (runBest == null || isBetter(generationBest, runBest)) {
                    runBest = generationBest.copy();
                    runBestGeneration = generation;
                }
                if (overallBest == null || isBetter(generationBest, overallBest)) {
                    overallBest = generationBest.copy();
                    overallBestRun = run;
                    overallBestGeneration = generation;
                    overallBestSeed = runSeed;
                }

                System.out.println(
                        "Run " + run + " | Gen " + generation +
                                " | Accuracy: " + formatPercent(generationBest.metrics.accuracy) +
                                " | F-measure: " + formatDouble(generationBest.metrics.fMeasure) +
                                " | Size: " + generationBest.root.size() +
                                " | Expr: " + generationBest.root.toExpression()
                );

                summaryWriter.write(run + "," + runSeed + "," + generation + "," +
                        formatDouble(generationBest.metrics.accuracy * 100.0) + "," +
                        formatDouble(generationBest.metrics.fMeasure) + "," +
                        generationBest.root.size() + "," +
                        csvEscape(generationBest.root.toExpression()) + "," +
                        csvEscape(generationBest.root.toExpressionWithNames()));
                summaryWriter.newLine();

                if (generation < MAX_GENERATIONS) {
                    population = createNextGeneration(population, random,
                            trainingData.featureCount, tournamentSize, maxOffspringDepth,
                            mutationOffspringDepth, crossoverRate, mutationRate);
                }
            }

            String perRunFile = MODEL_FOLDER + File.separator +
                    "arithmetic_model_run_" + run + "_seed_" + runSeed + ".txt";
            saveModel(perRunFile, runBest, run, runSeed, runBestGeneration,
                    "Best model for run " + run);
        }

        summaryWriter.close();
        long endTime = System.currentTimeMillis();
        lastRuntimeSeconds = (endTime - startTime) / 1000.0;
        lastTrainAccuracy  = overallBest != null ? overallBest.metrics.accuracy : 0;
        lastTrainFMeasure  = overallBest != null ? overallBest.metrics.fMeasure : 0;

        saveModel(BEST_MODEL_FILE, overallBest, overallBestRun, overallBestSeed,overallBestGeneration, "Overall best arithmetic GP model");

        System.out.println("====================================================");
        System.out.println("ARITHMETIC GP COMPLETE");
        System.out.println("Best training accuracy : " + formatPercent(overallBest.metrics.accuracy));
        System.out.println("Best F-measure         : " + formatDouble(overallBest.metrics.fMeasure));
        System.out.println("Best expression        : " + overallBest.root.toExpression());
        System.out.println("Runtime                : " + formatDouble(lastRuntimeSeconds) + "s");
        System.out.println("====================================================");
    }

    private static ArrayList<Integer> randomPath(Node root, Random random) {
        ArrayList<ArrayList<Integer> > allPaths = new ArrayList<ArrayList<Integer> >();
        collectPaths(root, new ArrayList<Integer>(), allPaths);
        return allPaths.get(random.nextInt(allPaths.size()));
    }

    private static void collectPaths(Node node, ArrayList<Integer> currentPath, ArrayList<ArrayList<Integer> > paths) {
        paths.add(new ArrayList<Integer>(currentPath));

        if (node instanceof FunctionNode) {
            FunctionNode functionNode = (FunctionNode) node;

            currentPath.add(Integer.valueOf(0));
            collectPaths(functionNode.left, currentPath, paths);
            currentPath.remove(currentPath.size() - 1);

            currentPath.add(Integer.valueOf(1));
            collectPaths(functionNode.right, currentPath, paths);
            currentPath.remove(currentPath.size() - 1);
        }
    }

    private static Node getSubtree(Node root, ArrayList<Integer> path) {
        Node current = root;

        for (int i = 0; i < path.size(); i++) {
            int direction = path.get(i).intValue();

            FunctionNode functionNode = (FunctionNode) current;

            if (direction == 0) {
                current = functionNode.left;
            } else {
                current = functionNode.right;
            }
        }

        return current;
    }

    private static Node replaceSubtree(Node root, ArrayList<Integer> path, Node replacement) {
        if (path.size() == 0) {
            return replacement;
        }

        Node newRoot = root.copy();
        Node current = newRoot;

        for (int i = 0; i < path.size() - 1; i++) {
            int direction = path.get(i).intValue();
            FunctionNode functionNode = (FunctionNode) current;

            if (direction == 0) {
                current = functionNode.left;
            } else {
                current = functionNode.right;
            }
        }

        int lastDirection = path.get(path.size() - 1).intValue();
        FunctionNode parent = (FunctionNode) current;

        if (lastDirection == 0) {
            parent.left = replacement;
        } else {
            parent.right = replacement;
        }

        return newRoot;
    }

    private static void saveModel(
            String filePath,
            Individual model,
            int run,
            long seed,
            int generation,
            String description
    ) throws IOException {
        BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8")
        );

        writer.write("MODEL_VERSION=1");
        writer.newLine();

        writer.write("MODEL_TYPE=ARITHMETIC_GP");
        writer.newLine();

        writer.write("DESCRIPTION=" + description);
        writer.newLine();

        writer.write("RUN=" + run);
        writer.newLine();

        writer.write("SEED=" + seed);
        writer.newLine();

        writer.write("GENERATION=" + generation);
        writer.newLine();

        writer.write("TRAINING_ACCURACY=" + formatDouble(model.metrics.accuracy));
        writer.newLine();

        writer.write("TRAINING_ACCURACY_PERCENT=" + formatDouble(model.metrics.accuracy * 100.0));
        writer.newLine();

        writer.write("TRAINING_FMEASURE=" + formatDouble(model.metrics.fMeasure));
        writer.newLine();

        writer.write("TREE_SIZE=" + model.root.size());
        writer.newLine();

        writer.write("FUNCTION_SET=+,-,*,protected_division");
        writer.newLine();

        writer.write("FEATURE_KEY:");
        writer.newLine();
        writer.write("CSV column 0 = class label / actual class");
        writer.newLine();
        writer.write("CSV column 1 = x0 = age");
        writer.newLine();
        writer.write("CSV column 2 = x1 = menopause");
        writer.newLine();
        writer.write("CSV column 3 = x2 = tumor_size");
        writer.newLine();
        writer.write("CSV column 4 = x3 = inv_nodes");
        writer.newLine();
        writer.write("CSV column 5 = x4 = node_caps");
        writer.newLine();
        writer.write("CSV column 6 = x5 = degree_of_malignancy");
        writer.newLine();
        writer.write("CSV column 7 = x6 = breast");
        writer.newLine();
        writer.write("CSV column 8 = x7 = breast_quad");
        writer.newLine();
        writer.write("CSV column 9 = x8 = irradiat");
        writer.newLine();

        writer.write("CLASSIFICATION_RULE=expression >= 0 predicts class 1 else class 0");
        writer.newLine();

        writer.write("PREFIX=" + model.root.toPrefix());
        writer.newLine();

        writer.write("EXPRESSION=" + model.root.toExpression());
        writer.newLine();

        writer.write("EXPRESSION_WITH_COLUMN_NAMES=" + model.root.toExpressionWithNames());
        writer.newLine();

        writer.newLine();
        writer.write("ARITHMETIC_TREE:");
        writer.newLine();
        writer.write(model.root.toVisualTree());
        writer.newLine();

        writer.close();
    }

    private static String csvEscape(String value) {
        String escaped = value.replace("\"", "\"\"");
        return "\"" + escaped + "\"";
    }

    private static String formatPercent(double value) {
        return String.format(Locale.US, "%.2f%%", value * 100.0);
    }

    private static String formatDouble(double value) {
        return String.format(Locale.US, "%.6f", value);
    }

    private static class DataSet {
        double[][] features;
        int[] labels;
        int featureCount;

        DataSet(double[][] features, int[] labels, int featureCount) {
            this.features = features;
            this.labels = labels;
            this.featureCount = featureCount;
        }
    }

    private static class Individual {
        Node root;
        Metrics metrics;

        Individual(Node root) {
            this.root = root;
            this.metrics = new Metrics();
        }

        Individual copy() {
            Individual copy = new Individual(root.copy());
            copy.metrics = metrics.copy();
            return copy;
        }
    }

    private static class Metrics {
        double accuracy;
        double precision;
        double recall;
        double fMeasure;
        int tp;
        int tn;
        int fp;
        int fn;

        Metrics copy() {
            Metrics copy = new Metrics();
            copy.accuracy = accuracy;
            copy.precision = precision;
            copy.recall = recall;
            copy.fMeasure = fMeasure;
            copy.tp = tp;
            copy.tn = tn;
            copy.fp = fp;
            copy.fn = fn;
            return copy;
        }
    }

    private static abstract class Node {
        abstract double evaluate(double[] features);
        abstract Node copy();
        abstract int depth();
        abstract int size();
        abstract String toExpression();
        abstract String toExpressionWithNames();
        abstract String toPrefix();
        abstract String visualLabel();

        String toVisualTree() {
            StringBuilder builder = new StringBuilder();
            buildVisualTree(builder, "", true, true);
            return builder.toString();
        }

        private void buildVisualTree(StringBuilder builder, String prefix, boolean isTail, boolean isRoot) {
            if (isRoot) {
                builder.append(visualLabel()).append(System.lineSeparator());
            } else {
                builder.append(prefix);
                builder.append(isTail ? "└── " : "├── ");
                builder.append(visualLabel()).append(System.lineSeparator());
            }

            if (this instanceof FunctionNode) {
                FunctionNode functionNode = (FunctionNode) this;

                String childPrefix;

                if (isRoot) {
                    childPrefix = "";
                } else {
                    childPrefix = prefix + (isTail ? "    " : "│   ");
                }

                functionNode.left.buildVisualTree(builder, childPrefix, false, false);
                functionNode.right.buildVisualTree(builder, childPrefix, true, false);
            }
        }
    }

    private static class FeatureNode extends Node {
        int index;

        FeatureNode(int index) {
            this.index = index;
        }

        double evaluate(double[] features) {
            return features[index];
        }

        Node copy() {
            return new FeatureNode(index);
        }

        int depth() {
            return 1;
        }

        int size() {
            return 1;
        }

        String toExpression() {
            return "x" + index;
        }

        String toExpressionWithNames() {
            if (index >= 0 && index < FEATURE_NAMES.length) {
                return FEATURE_NAMES[index];
            }

            return "x" + index;
        }

        String toPrefix() {
            return "F" + index;
        }

        String visualLabel() {
            return "x" + index;
        }
    }

    private static class ConstantNode extends Node {
        int value;

        ConstantNode(int value) {
            this.value = value;
        }

        double evaluate(double[] features) {
            return value;
        }

        Node copy() {
            return new ConstantNode(value);
        }

        int depth() {
            return 1;
        }

        int size() {
            return 1;
        }

        String toExpression() {
            return String.valueOf(value);
        }

        String toExpressionWithNames() {
            return String.valueOf(value);
        }

        String toPrefix() {
            return "C" + value;
        }

        String visualLabel() {
            return String.valueOf(value);
        }
    }

    private static class FunctionNode extends Node {
        String operator;
        Node left;
        Node right;

        FunctionNode(String operator, Node left, Node right) {
            this.operator = operator;
            this.left = left;
            this.right = right;
        }

        double evaluate(double[] features) {
            double a = left.evaluate(features);
            double b = right.evaluate(features);

            if (operator.equals("ADD")) {
                return a + b;
            } else if (operator.equals("SUB")) {
                return a - b;
            } else if (operator.equals("MUL")) {
                return a * b;
            } else {
                if (Math.abs(b) < 0.000000001) {
                    return 1.0;
                }
                return a / b;
            }
        }

        Node copy() {
            return new FunctionNode(operator, left.copy(), right.copy());
        }

        int depth() {
            int leftDepth = left.depth();
            int rightDepth = right.depth();
            return 1 + Math.max(leftDepth, rightDepth);
        }

        int size() {
            return 1 + left.size() + right.size();
        }

        String toExpression() {
            String symbol = getSymbol();
            return "(" + left.toExpression() + " " + symbol + " " + right.toExpression() + ")";
        }

        String toExpressionWithNames() {
            String symbol = getSymbol();
            return "(" + left.toExpressionWithNames() + " " + symbol + " " + right.toExpressionWithNames() + ")";
        }

        String toPrefix() {
            return operator + "(" + left.toPrefix() + "," + right.toPrefix() + ")";
        }

        String visualLabel() {
            return getSymbol();
        }

        private String getSymbol() {
            if (operator.equals("ADD")) {
                return "+";
            } else if (operator.equals("SUB")) {
                return "-";
            } else if (operator.equals("MUL")) {
                return "*";
            } else {
                return "/";
            }
        }
    }
}