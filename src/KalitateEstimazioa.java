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


    public void ezZintzoa(Instances train, Instances dev, Classifier mlp, String path) throws Exception {
        System.out.println("Ez-zintzoa gauzatzen...");

        System.out.println("Training data: " + train.numInstances());
        System.out.println("Dev data: " + dev.numInstances());
        Instances dataGuztia =new  Instances(train);
        dataGuztia.addAll(dev);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(mlp, dataGuztia);
        System.out.println(eval.toSummaryString("\n=== Eval Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("\n=== Class Details ===\n"));
        System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        metrikak(eval, train, path);
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


    public void stratifiedRepeatedHoldOut(Instances train, Instances dev, int repeats, double trainRatio, int seed, String tempDirPath, String pathOut, Classifier mlp) throws Exception {

        Instances dataTotala = new Instances(train);
        dataTotala.addAll(dev);
        if (dataTotala.classIndex() == -1) {
            dataTotala.setClassIndex(dataTotala.numAttributes() - 1);
        }
        //Direktorio tenporala sortzeko
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

                int spamIdx = 1;

                double p = eval.precision(spamIdx);
                double r = eval.recall(spamIdx);
                double f = eval.fMeasure(spamIdx);

                pSum += p;
                rSum += r;
                fSum += f;

                System.out.println("[Rep " + (rep + 1) + "/" + repeats + "] train=" + trainBek.numInstances() + " dev=" + devBek.numInstances());
                System.out.println(eval.toSummaryString("\n=== Eval Summary ===\n", false));
                System.out.println(eval.toMatrixString());

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


    private Instances[] splitResample(Instances data, double trainRatio, int seed) throws Exception {
        Resample rs = new Resample();
        rs.setNoReplacement(true);
        rs.setInvertSelection(false);
        rs.setRandomSeed(seed);
        rs.setSampleSizePercent(trainRatio * 100.0);
        rs.setInputFormat(data);
        Instances trainSplit = Filter.useFilter(data, rs);

        rs.setInvertSelection(true);
        rs.setInputFormat(data);
        Instances devSplit = Filter.useFilter(data, rs);

        return new Instances[]{trainSplit, devSplit};
    }

    private void gordeArff(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
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
