import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

public class KalitateEstimazioa {
    private DataProcessor dataProcessor;
    private Sailkatzailea sailkatzailea;

    /**
     * Eraikitzailea.
     *
     * <p>Ebaluazio-fluxuan behar diren laguntzaileak hasieratzen ditu:
     * bektorizaziorako {@link DataProcessor} eta sailkatzailearen utilitateetarako
     * {@link Sailkatzailea}.</p>
     */
    public KalitateEstimazioa() {
        this.dataProcessor = new DataProcessor();
        this.sailkatzailea = new Sailkatzailea();
    }


    /**
     * Ebaluazio ez-zintzoa exekutatzen du (train + dev multzo berdinean ebaluatuz).
     *
     * <p>Metodo honek datu-fuga eragin dezake eta, ondorioz, metrika optimistak ematea.
     * Konparazio akademikorako edo baseline gisa erabil daiteke, baina ez da gomendagarria
     * kalitate errealaren estimaziorako.</p>
     *
     * @param train entrenamendurako instantziak
     * @param dev garapenerako instantziak
     * @param path metrikak idazteko irteerako fitxategiaren bidea
     * @throws Exception ebaluazioan edo fitxategi-idazketan errorea badago
     */
    public void ezZintzoa(Instances train, Instances dev, String path) throws Exception {
        System.out.println("Ez-zintzoa gauzatzen...");

        System.out.println("Training data: " + train.numInstances());
        System.out.println("Dev data: " + dev.numInstances());
        Instances dataGuztia =new  Instances(train);
        dataGuztia.addAll(dev);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataGuztia);
        saver.setFile(new File("src/data/arff/trainDev.arff"));
        saver.writeBatch();
        dataProcessor.bektorizatu("src/data/arff/trainDev.arff","src/data/arff/trainDevBek.arff","src/data/model/trainDevMultiFilter.model",true);
        DataSource source = new DataSource("src/data/arff/trainDevBek.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        Evaluation eval = new Evaluation(data);
        Classifier mlp = sailkatzailea.sailkatzaileOptimoaSortuHardCoded();
        mlp.buildClassifier(data);
        eval.evaluateModel(mlp, data);
        System.out.println(eval.toSummaryString("\n=== Eval Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("\n=== Class Details ===\n"));
        System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        metrikak(eval, data, path);
    }


    /**
     * Hold-out ebaluazio estandarra egiten du.
     *
     * @param trainData entrenamendu multzoa (eredua aurrez entrenatuta dagoela suposatzen da)
     * @param devData balidazio/garapen multzoa
     * @param path metrikak idazteko irteerako fitxategiaren bidea
     * @throws Exception ebaluazioan edo fitxategi-idazketan errorea badago
     */
    public void holdOut(Instances trainData, Instances devData, String path) throws Exception {
        System.out.println("Hold Out gauzatzen...");

        System.out.println("Training data: " + trainData.numInstances());
        System.out.println("Dev data: " + devData.numInstances());

        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainData);
        saver.setFile(new File("src/data/arff/train.arff"));
        saver.writeBatch();
        dataProcessor.bektorizatu("src/data/arff/train.arff","src/data/arff/trainBek.arff","src/data/model/trainMultiFilter.model",true);
        DataSource source1 = new DataSource("src/data/arff/trainBek.arff");
        Instances train = source1.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        saver = new ArffSaver();
        saver.setInstances(devData);
        saver.setFile(new File("src/data/arff/dev.arff"));
        saver.writeBatch();
        dataProcessor.bektorizatu("src/data/arff/dev.arff","src/data/arff/devBek.arff","src/data/model/trainMultiFilter.model",false);
        DataSource source2 = new DataSource("src/data/arff/devBek.arff");
        Instances dev = source2.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);


        Classifier mlp = sailkatzailea.sailkatzaileOptimoaSortuHardCoded();
        mlp.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(mlp, dev);
        System.out.println(eval.toSummaryString("\n=== Eval Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("\n=== Class Details ===\n"));
        System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        metrikak(eval, train, path);
    }


    /**
     * Stratified Repeated Hold-Out egiten du train+dev raw bateratuta.
     *
     * <p>Errepikapen bakoitzean prozesua hau da:
     * split estratifikatua -> train/dev raw ARFF gordetzea -> train filtroz bektorizatzea
     * -> dev multzoa filtro berarekin eraldatzea -> modeloa entrenatu eta ebaluatzea.</p>
     *
     * @param train hasierako train raw instantziak
     * @param dev hasierako dev raw instantziak
     * @param repeats errepikapen kopurua
     * @param trainRatio split bakoitzean train proportzioa (0-1)
     * @param seed ausazko hasierako hazia
     * @param tempDirPath aldi baterako ARFF/model fitxategiak gordetzeko karpeta
     * @param pathOut errepikapen eta batezbesteko metrikak idazteko fitxategia
     * @throws Exception split, bektorizazio, entrenamendu edo idazketan errorea badago
     */
    public void stratifiedRepeatedHoldOut(Instances train, Instances dev, int repeats, double trainRatio, int seed, String tempDirPath, String pathOut) throws Exception {

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

                Classifier mlp = sailkatzailea.sailkatzaileOptimoaSortuHardCoded();
                mlp.buildClassifier(trainBek);

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


    /**
     * Resample filtroarekin train/dev split bat sortzen du ordezkapenik gabe.
     *
     * <p>Lehen pasean train zatia aukeratzen du; bigarrenean, {@code invertSelection=true}
     * erabilita osagarria (dev zatia) lortzen da.</p>
     *
     * @param data zatitu beharreko datu-multzoa
     * @param trainRatio train-erako proportzioa (0-1)
     * @param seed ausazko hazia
     * @return bi posizioko arraya: [0] train split, [1] dev split
     * @throws Exception filtroaren exekuzioan errorea badago
     */
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

    /**
     * {@link Instances} objektu bat ARFF fitxategi gisa gordetzen du.
     *
     * @param data gorde beharreko instantziak
     * @param outputPath irteerako ARFF fitxategiaren bidea
     * @throws Exception idazketan errorea badago
     */
    private void gordeArff(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }


    /**
     * Spam klasearen metrika nagusiak fitxategi batera idazten ditu.
     *
     * @param eval egindako ebaluazioaren emaitza
     * @param data erabilitako datu multzoa (une honetan ez da zuzenean erabiltzen)
     * @param pathOut irteerako fitxategiaren bidea
     * @throws Exception fitxategia sortu edo idaztean errorea badago
     */
    private void metrikak(Evaluation eval, Instances data, String pathOut) throws Exception {

        int spamIdx = 1;

        PrintWriter pw = new PrintWriter(new File(pathOut));
        pw.println("Precision: " + eval.precision(spamIdx));
        pw.println("Recall: " + eval.recall(spamIdx));
        pw.println("F-Score: " + eval.fMeasure(spamIdx));
        pw.println(eval.toMatrixString());
        pw.close();
    }

}
