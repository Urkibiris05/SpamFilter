import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

public class KalitateEstimazioa {
    private DataProcessor dataProcessor;
    private Sailkatzailea sailkatzailea;

    public KalitateEstimazioa() {
        this.dataProcessor = new DataProcessor();
        this.sailkatzailea = new Sailkatzailea();
    }

    public Classifier sailkatzaileEstandarraSortu(Instances trainData) throws Exception {
        System.out.println("MLP sortzen...");

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setLearningRate(0.05);
        mlp.setMomentum(0.1);
        mlp.setHiddenLayers("5");
        mlp.setBatchSize("100");

        return mlp;
    }


    public void holdOut(Instances trainData, Instances devData,Classifier mlp, String path) throws Exception {
        System.out.println("Hold Out gauzatzen...");

        System.out.println("Training data: " + trainData.numInstances());
        System.out.println("Dev data: " + devData.numInstances());

        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(mlp, devData);
        System.out.println(eval.toSummaryString("\n=== Eval Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("\n=== Class Details ===\n"));
        System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        metrikak(eval, trainData, path);
    }
    public void stratifiedRepeatedHoldOut(String rawTrainPath, String rawDevPath, int repeats, double trainRatio, int seed, String tempDirPath, String pathOut, Classifier mlp) throws Exception {

        DataSource sourceTrain = new DataSource(rawTrainPath);
        Instances train = sourceTrain.getDataSet();
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }

        DataSource sourceDev = new DataSource(rawDevPath);
        Instances dev = sourceDev.getDataSet();
        if (dev.classIndex() == -1) {
            dev.setClassIndex(dev.numAttributes() - 1);
        }

        Instances dataTotala = new Instances(train);
        dataTotala.addAll(dev);
        if (dataTotala.classIndex() == -1) {
            dataTotala.setClassIndex(dataTotala.numAttributes() - 1);
        }

        Path tempDir = Path.of(tempDirPath);
        Files.createDirectories(tempDir);

        double pSum = 0.0;
        double rSum = 0.0;
        double fSum = 0.0;
        String[] parametroOptimoak = null;


        try (PrintWriter pw = new PrintWriter(new File(pathOut))) {
            pw.println("Stratified Repeated Hold-Out (raw train+dev)");
            pw.println("Repeats: " + repeats);
            pw.println("Train ratio: " + trainRatio);
            pw.println();

            for (int rep = 0; rep < repeats; rep++) {
                long repSeed = (long) seed + rep;
                // Repeated hold-out con Resample sin reemplazo.
                Instances[] split = splitResample(dataTotala, trainRatio, (int) repSeed);
                Instances rawSplitTrain = split[0];
                Instances rawSplitDev = split[1];

                String repRawTrainPath = tempDir.resolve("rep_" + rep + "_train_raw.arff").toString();
                String repRawDevPath = tempDir.resolve("rep_" + rep + "_dev_raw.arff").toString();
                String repBekTrainPath = tempDir.resolve("rep_" + rep + "_train_bek.arff").toString();
                String repBekDevPath = tempDir.resolve("rep_" + rep + "_dev_bek.arff").toString();
                String repFilterPath = tempDir.resolve("rep_" + rep + "_filter.model").toString();

                gordeArff(rawSplitTrain, repRawTrainPath);
                gordeArff(rawSplitDev, repRawDevPath);

                // Errepikapen bakoitzean bektorizazioa berreraiki
                dataProcessor.bektorizatu(repRawTrainPath, repBekTrainPath, repFilterPath, true);
                dataProcessor.bektorizatu(repRawDevPath, repBekDevPath, repFilterPath, false);

                Instances trainBek = new DataSource(repBekTrainPath).getDataSet();
                Instances devBek = new DataSource(repBekDevPath).getDataSet();
                if (trainBek.classIndex() == -1) trainBek.setClassIndex(trainBek.numAttributes() - 1);
                if (devBek.classIndex() == -1) devBek.setClassIndex(devBek.numAttributes() - 1);



                Evaluation eval = new Evaluation(trainBek);
                eval.evaluateModel(mlp, devBek);

                int spamIdx = trainBek.classAttribute().indexOfValue("spam");
                if (spamIdx < 0) {
                    throw new IllegalArgumentException("'spam' klasea ez da aurkitu dataset-ean");
                }


                double p = eval.precision(spamIdx);
                double r = eval.recall(spamIdx);
                double f = eval.fMeasure(spamIdx);

                pSum += p;
                rSum += r;
                fSum += f;

                System.out.println("[Rep " + (rep + 1) + "/" + repeats + "] train=" + trainBek.numInstances() + " dev=" + devBek.numInstances());
                System.out.println(eval.toSummaryString("\n=== Eval Summary ===\n", false));

                pw.println("Rep " + (rep + 1));
                pw.println(String.format(Locale.US, "Precision(spam): %.6f", p));
                pw.println(String.format(Locale.US, "Recall(spam): %.6f", r));
                pw.println(String.format(Locale.US, "F1(spam): %.6f", f));
                pw.println();
            }

            pw.println("---- Mean ----");
            pw.println(String.format(Locale.US, "Precision(spam) mean: %.6f", pSum / repeats));
            pw.println(String.format(Locale.US, "Recall(spam) mean: %.6f", rSum / repeats));
            pw.println(String.format(Locale.US, "F1(spam) mean: %.6f", fSum / repeats));
        }
    }

    private Instances[] splitEstratifikatua(Instances data, double trainRatio, long seed) {
        Instances trainSplit = new Instances(data, 0);
        Instances devSplit = new Instances(data, 0);

        Random random = new Random(seed);
        for (int classIdx = 0; classIdx < data.numClasses(); classIdx++) {
            Instances classSubset = new Instances(data, 0);
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if ((int) inst.classValue() == classIdx) {
                    classSubset.add(inst);
                }
            }

            classSubset.randomize(random);
            // No replacement
            int trainCount = (int) Math.round(classSubset.numInstances() * trainRatio);
            if (classSubset.numInstances() > 1) {
                trainCount = Math.max(1, Math.min(trainCount, classSubset.numInstances() - 1));
            }

            for (int i = 0; i < classSubset.numInstances(); i++) {
                if (i < trainCount) {
                    trainSplit.add(classSubset.instance(i));
                } else {
                    devSplit.add(classSubset.instance(i));
                }
            }
        }

        trainSplit.randomize(random);
        devSplit.randomize(random);
        return new Instances[]{trainSplit, devSplit};
    }

    private Instances[] splitResample(Instances data, double trainRatio, int seed) throws Exception {
        Resample trainResample = new Resample();
        trainResample.setNoReplacement(true);
        trainResample.setInvertSelection(false);
        trainResample.setRandomSeed(seed);
        trainResample.setSampleSizePercent(trainRatio * 100.0);
        trainResample.setInputFormat(data);
        Instances trainSplit = Filter.useFilter(data, trainResample);

        Resample devResample = new Resample();
        devResample.setNoReplacement(true);
        devResample.setInvertSelection(true);
        devResample.setRandomSeed(seed);
        devResample.setSampleSizePercent(trainRatio * 100.0);
        devResample.setInputFormat(data);
        Instances devSplit = Filter.useFilter(data, devResample);

        return new Instances[]{trainSplit, devSplit};
    }

    private void gordeArff(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }

    private Classifier sortuMLPParametroekin(String[] parametroak) {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setSeed(42);

        // Sailkatzailea.sailkatzaileaSortu-n erabiltzen diren parametro berdinak aplikatu
        if (parametroak != null && parametroak.length >= 3) {
            mlp.setHiddenLayers(parametroak[0]);
            mlp.setLearningRate(Double.parseDouble(parametroak[1]));
            mlp.setMomentum(Double.parseDouble(parametroak[2]));
        } else {
            mlp.setHiddenLayers("5");
            mlp.setLearningRate(0.05);
            mlp.setMomentum(0.1);
        }

        mlp.setValidationSetSize(20);
        mlp.setValidationThreshold(15);
        return mlp;
    }


    private void metrikak(Evaluation eval, Instances data, String pathOut) throws Exception {

        int spamIdx = 1;

        PrintWriter pw = new PrintWriter(new File(pathOut));
        pw.println("Precision: " + eval.precision(spamIdx));
        pw.println("Recall: " + eval.recall(spamIdx));
        pw.println("F-Score: " + eval.fMeasure(spamIdx));
        pw.close();
    }

}
