import java.io.*;
import java.util.*;

public class ArithmeticGPTest {

    private static final String MODEL_PATH = "best_arithmetic_model.txt";
    private static final String OUTPUT_PATH = "arithmetic_test_predictions.csv";

    private static final String[] CLASS_NAMES = {
            "no-recurrence-events",
            "recurrence-events"
    };

    // Result holder returned to Assignment3.main
    public static class TestResult {
        public double accuracy;
        public double fMeasure;
        public double trainAccuracy;
        public double runtimeSeconds;
    }

    // Called programmatically from Assignment3.main
    public static TestResult runTest(String testFilePath) throws Exception {
        long start = System.currentTimeMillis();

        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            throw new Exception("Model file not found: " + modelFile.getAbsolutePath()
                    + ". Run Arithmetic GP training first.");
        }

        SavedModel model = loadModel(MODEL_PATH);
        DataSet testingData = loadCsv(testFilePath);

        Metrics testMetrics = evaluateAndSavePredictions(model.root, testingData, OUTPUT_PATH);

        System.out.println("------------------------------------------------------------");
        System.out.println("Arithmetic GP — Test Results:");
        System.out.printf("  Accuracy  : %.4f  (%.2f%%)%n",
                testMetrics.accuracy, testMetrics.accuracy * 100);
        System.out.printf("  Precision : %.4f%n", testMetrics.precision);
        System.out.printf("  Recall    : %.4f%n", testMetrics.recall);
        System.out.printf("  F-Measure : %.4f%n", testMetrics.fMeasure);
        System.out.printf("  TP: %d  |  FP: %d  |  TN: %d  |  FN: %d%n",
                testMetrics.tp, testMetrics.fp, testMetrics.tn, testMetrics.fn);
        System.out.println("------------------------------------------------------------");

        TestResult result = new TestResult();
        result.accuracy       = testMetrics.accuracy;
        result.fMeasure       = testMetrics.fMeasure;
        result.trainAccuracy  = Double.parseDouble(model.trainingAccuracyPercent) / 100.0;
        result.runtimeSeconds = (System.currentTimeMillis() - start) / 1000.0;
        return result;
    }

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        System.out.println("_____________________________________________________");
        System.out.println("ARITHMETIC GP TESTING PROGRAM");
        System.out.println("This program loads the saved best arithmetic GP model");
        System.out.println("and classifies the unseen test CSV file.");
        System.out.println("_____________________________________________________");
        System.out.println();

        System.out.println("Saved model filepath: " + MODEL_PATH);
        System.out.println("Output predictions filepath: " + OUTPUT_PATH);
        System.out.println();

        System.out.print("Enter test CSV filepath: ");
        String testingPath = input.nextLine().trim();

        try {
            File modelFile = new File(MODEL_PATH);

            if (!modelFile.exists()) {
                System.out.println("ERROR: Model file was not found.");
                System.out.println("Expected model path: " + modelFile.getAbsolutePath());
                return;
            }

            SavedModel model = loadModel(MODEL_PATH);
            DataSet testingData = loadCsv(testingPath);

            System.out.println();
            System.out.println("_____________________________________________________");
            System.out.println("DEBUG MODEL CHECK");
            System.out.println("_____________________________________________________");
            System.out.println("Model absolute path: " + modelFile.getAbsolutePath());
            System.out.println("Model last modified: " + new Date(modelFile.lastModified()));
            System.out.println("Loaded prefix hash: " + model.prefix.hashCode());
            System.out.println("Loaded prefix:");
            System.out.println(model.prefix);
            System.out.println();
            System.out.println("Loaded expression:");
            System.out.println(model.expression);
            System.out.println();
            System.out.println("Loaded training accuracy: " + model.trainingAccuracyPercent + "%");
            System.out.println("Loaded training F-measure: " + model.trainingFMeasure);
            System.out.println("_____________________________________________________");

            Metrics testMetrics = evaluateAndSavePredictions(
                    model.root,
                    testingData,
                    OUTPUT_PATH
            );

            System.out.println();
            System.out.println("_____________________________________________________");
            System.out.println("TESTING COMPLETE");
            System.out.println("_____________________________________________________");
            System.out.println("Model file: " + MODEL_PATH);
            System.out.println("Testing file: " + testingPath);
            System.out.println("Predictions saved to: " + OUTPUT_PATH);
            System.out.println();
            System.out.println("Test accuracy: " + formatPercent(testMetrics.accuracy));
            System.out.println("Test F-measure: " + formatDouble(testMetrics.fMeasure));
            System.out.println("Precision: " + formatDouble(testMetrics.precision));
            System.out.println("Recall: " + formatDouble(testMetrics.recall));
            System.out.println();
            System.out.println("Confusion Matrix:");
            System.out.println("TP: " + testMetrics.tp);
            System.out.println("TN: " + testMetrics.tn);
            System.out.println("FP: " + testMetrics.fp);
            System.out.println("FN: " + testMetrics.fn);
            System.out.println();
            System.out.println("Correct test predictions: " + testMetrics.correct);
            System.out.println("Total test rows: " + testMetrics.totalRows);
            System.out.println("Predicted class 0 count: " + testMetrics.predictedClassZero);
            System.out.println("Predicted class 1 count: " + testMetrics.predictedClassOne);
            System.out.println("_____________________________________________________");

        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static SavedModel loadModel(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));

        String line;
        String prefix = null;
        String expression = "";
        String trainingAccuracyPercent = "";
        String trainingFMeasure = "";

        while ((line = reader.readLine()) != null) {
            if (line.startsWith("PREFIX=")) {
                prefix = line.substring("PREFIX=".length()).trim();
            } else if (line.startsWith("EXPRESSION=")) {
                expression = line.substring("EXPRESSION=".length()).trim();
            } else if (line.startsWith("TRAINING_ACCURACY_PERCENT=")) {
                trainingAccuracyPercent = line.substring("TRAINING_ACCURACY_PERCENT=".length()).trim();
            } else if (line.startsWith("TRAINING_FMEASURE=")) {
                trainingFMeasure = line.substring("TRAINING_FMEASURE=".length()).trim();
            }
        }

        reader.close();

        if (prefix == null || prefix.length() == 0) {
            throw new IOException("Model file does not contain a PREFIX= line.");
        }

        PrefixParser parser = new PrefixParser(prefix);

        SavedModel model = new SavedModel();
        model.root = parser.parse();
        model.prefix = prefix;
        model.expression = expression;
        model.trainingAccuracyPercent = trainingAccuracyPercent;
        model.trainingFMeasure = trainingFMeasure;

        return model;
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

    private static Metrics evaluateAndSavePredictions(
            Node root,
            DataSet dataSet,
            String outputPath
    ) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));

        writer.write("row,actual,predicted,actual_label,predicted_label,raw_output,correct");
        writer.newLine();

        int correct = 0;
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        int predictedClassZero = 0;
        int predictedClassOne = 0;

        for (int i = 0; i < dataSet.features.length; i++) {
            double output = root.evaluate(dataSet.features[i]);
            int predicted = output >= 0.0 ? 1 : 0;
            int actual = dataSet.labels[i];

            if (predicted == 0) {
                predictedClassZero++;
            } else {
                predictedClassOne++;
            }

            boolean isCorrect = predicted == actual;

            if (isCorrect) {
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

            writer.write(
                    (i + 1) + "," +
                            actual + "," +
                            predicted + "," +
                            getClassName(actual) + "," +
                            getClassName(predicted) + "," +
                            formatDouble(output) + "," +
                            isCorrect
            );
            writer.newLine();
        }

        writer.close();

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
        metrics.correct = correct;
        metrics.totalRows = dataSet.labels.length;
        metrics.predictedClassZero = predictedClassZero;
        metrics.predictedClassOne = predictedClassOne;

        return metrics;
    }

    private static String getClassName(int classValue) {
        if (classValue >= 0 && classValue < CLASS_NAMES.length) {
            return CLASS_NAMES[classValue];
        }

        return "unknown";
    }

    private static String formatPercent(double value) {
        return String.format(Locale.US, "%.2f%%", value * 100.0);
    }

    private static String formatDouble(double value) {
        return String.format(Locale.US, "%.6f", value);
    }

    private static class SavedModel {
        Node root;
        String prefix;
        String expression;
        String trainingAccuracyPercent;
        String trainingFMeasure;
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

    private static class Metrics {
        double accuracy;
        double precision;
        double recall;
        double fMeasure;
        int tp;
        int tn;
        int fp;
        int fn;
        int correct;
        int totalRows;
        int predictedClassZero;
        int predictedClassOne;
    }

    private static abstract class Node {
        abstract double evaluate(double[] features);
    }

    private static class FeatureNode extends Node {
        int index;

        FeatureNode(int index) {
            this.index = index;
        }

        double evaluate(double[] features) {
            return features[index];
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
    }

    private static class PrefixParser {
        private String text;
        private int position;

        PrefixParser(String text) {
            this.text = text;
            this.position = 0;
        }

        Node parse() {
            Node node = parseNode();
            skipSpaces();

            if (position != text.length()) {
                throw new RuntimeException("Unexpected text at position " + position);
            }

            return node;
        }

        private Node parseNode() {
            skipSpaces();

            String word = parseWord();

            if (word.length() == 0) {
                throw new RuntimeException("Expected token at position " + position);
            }

            if (word.charAt(0) == 'F') {
                int index = Integer.parseInt(word.substring(1));
                return new FeatureNode(index);
            }

            if (word.charAt(0) == 'C') {
                int value = Integer.parseInt(word.substring(1));
                return new ConstantNode(value);
            }

            if (
                    word.equals("ADD") ||
                            word.equals("SUB") ||
                            word.equals("MUL") ||
                            word.equals("DIV")
            ) {
                expect('(');
                Node left = parseNode();
                expect(',');
                Node right = parseNode();
                expect(')');

                return new FunctionNode(word, left, right);
            }

            throw new RuntimeException("Unknown token: " + word);
        }

        private String parseWord() {
            skipSpaces();

            StringBuilder builder = new StringBuilder();

            while (position < text.length()) {
                char c = text.charAt(position);

                if (Character.isLetterOrDigit(c)) {
                    builder.append(c);
                    position++;
                } else {
                    break;
                }
            }

            return builder.toString();
        }

        private void expect(char expected) {
            skipSpaces();

            if (position >= text.length() || text.charAt(position) != expected) {
                throw new RuntimeException("Expected '" + expected + "' at position " + position);
            }

            position++;
        }

        private void skipSpaces() {
            while (position < text.length() && Character.isWhitespace(text.charAt(position))) {
                position++;
            }
        }
    }
}