import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Scanner;

public class SpamFilter {

    private static final DataProcessor dataProcessor = new DataProcessor();
    private static final Sailkatzailea sailkatzailea = new Sailkatzailea();
    private static final KalitateEstimazioa kalitateEstimazioa = new KalitateEstimazioa();
    private static final Iragarpenak iragarpenak = new Iragarpenak();
    private static final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        exekutatuMenua();
    }

    private static void exekutatuMenua() {
        boolean martxan = true;

        while (martxan) {
            inprimatuMenua();
            String aukera = eskatuTestua("Aukera hautatu");

            try {
                switch (aukera) {
                    case "1":
                        sms2ArffInteraktiboa();
                        break;
                    case "2":
                        instantziakAztertuInteraktiboa();
                        break;
                    case "3":
                        bektorizatuInteraktiboa();
                        break;
                    case "4":
                        parametroBilatzaileaInteraktiboa();
                        break;
                    case "5":
                        parametroBilatzaileaV2Interaktiboa();
                        break;
                    case "6":
                        parametroEkorketaInteraktiboa();
                        break;
                    case "7":
                        modeloOptimoaSortuInteraktiboa();
                        break;
                    case "8":
                        kalitateaEstimatuInteraktiboa();
                        break;
                    /*case "9":
                        pipelineOsoaInteraktiboa();
                        break;

                     */
                    case "10":
                        iragarpenak();
                    case "0":
                        martxan = false;
                        System.out.println("Agur! Programa amaituta.");
                        break;
                    default:
                        System.out.println("Aukera baliogabea. Saiatu berriro.");
                }
            } catch (Exception e) {
                System.out.println("Errorea funtzionalitatea exekutatzean: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private static void iragarpenak() throws Exception {
        String testBekPath = eskatuTestua("Bek TEST Bektorizatuta path-a");
        String testPath = eskatuTestua("TEST path-a");
        String modelPath = eskatuTestua("Kargatu beharreko model path-a (.model)");
        String iragarpenakOut = eskatuTestua("Iragarpenak gordetzeko fitxategia");

        Classifier modeloa = (Classifier) SerializationHelper.read(modelPath);
        Instances testBek = kargatuInstantziak(testBekPath);
        Instances test = kargatuInstantziak(testPath);
        iragarpenak.Iragarpenak(testBek, test, modeloa,iragarpenakOut);
    }

    private static void inprimatuMenua() {
        System.out.println("\n================= SPAM FILTER MENUA =================");
        System.out.println("1) TXT -> ARFF (sms2Arff)");
        System.out.println("2) Instantziak aztertu");
        System.out.println("3) Bektorizatu datu sorta");
        System.out.println("4) Parametro bilatzailea (V1)");
        System.out.println("5) Parametro bilatzailea (V2)");
        System.out.println("6) Parametro Ekorketa");
        System.out.println("7) Modelo optimoa sortu");
        System.out.println("8) Kalitate estimazioa (hold-out)");
        System.out.println("9) Pipeline osoa (preprozesatu + bektorizatu + entrenatu + ebaluatu)");
        System.out.println("10) Iragarpenak");
        System.out.println("0) Irten");
        System.out.println("=====================================================");
    }

    private static void sms2ArffInteraktiboa() throws Exception {
        String txtPath = eskatuTestua("Sartu TXT sarrerako path-a");
        String arffPath = eskatuTestua("Sartu ARFF irteerako path-a");
        boolean blind = eskatuBaiEz("Class blind da? (bai/ez)");
        dataProcessor.sms2Arff(txtPath, arffPath, blind);
    }

    private static void instantziakAztertuInteraktiboa() throws Exception {
        String dataPath = eskatuTestua("Sartu aztertu nahi duzun ARFF path-a");
        String etapaIzena = eskatuLehenetsia("Etapa izena", "ETAPA");
        dataProcessor.instantziakAztertu(dataPath, etapaIzena);
    }

    private static void bektorizatuInteraktiboa() throws Exception {
        String rawDataPath = eskatuTestua("Raw ARFF path-a");
        String bekDataPath = eskatuTestua("Bektorizatutako ARFF path-a");
        String filterModelPath = eskatuTestua("Filtro/modelo path-a (adib. multiFilter.model)");
        boolean isTrain = eskatuBaiEz("Train multzoa da? (bai/ez)");
        dataProcessor.bektorizatu(rawDataPath, bekDataPath, filterModelPath, isTrain);
    }

    private static void parametroBilatzaileaInteraktiboa() throws Exception {
        String rawTrainPath = eskatuTestua("Raw TRAIN ARFF path-a");
        dataProcessor.parametroBilatzailea(rawTrainPath);
    }

    private static void parametroBilatzaileaV2Interaktiboa() throws Exception {
        String rawTrainPath = eskatuTestua("Raw TRAIN ARFF path-a");
        String rawDevPath = eskatuTestua("Raw DEV ARFF path-a");
        dataProcessor.parametroBilatzaileaV2(rawTrainPath, rawDevPath);
    }

    private static void parametroEkorketaInteraktiboa() throws Exception {
        String[] arff = new String[2];
        arff[0] = eskatuTestua("Bektorizatutako TRAIN ARFF path-a");
        arff[1] = eskatuTestua("Bektorizatutako DEV ARFF path-a");

        Instances[] instantziak = new Instances[2];
        instantziak = sailkatzailea.arffKargatu(arff);
        sailkatzailea.parametroOptimoakLortu(instantziak);
    }

    private static void modeloOptimoaSortuInteraktiboa() throws Exception {
        String hl = eskatuTestua("HiddenLayers parametroa");
        String lr = eskatuTestua("LearningRate parametroa");
        String m = eskatuTestua("Momentum parametroa");
        String[] parametroak = new String[3];
        parametroak[0] = hl;
        parametroak[1] = lr;
        parametroak[2] = m;
        String rawTrainPath = eskatuTestua("Raw TRAIN ARFF path-a");
        String rawDevPath = eskatuTestua("Raw DEV ARFF path-a");
        String rawDataPath = eskatuTestua("Raw DATA ARFF path-a");
        String bekDataPath = eskatuTestua("Bek DATA ARFF path-a");
        String filterModelPath = eskatuTestua("Filtro/modelo path-a (adib. multiFilter.model)");
        String outputPath = eskatuTestua("Modelo optimoa gordetzeko path-a");

        sailkatzailea.sailkatzaileaSortu(parametroak, rawTrainPath, rawDevPath, rawDataPath, bekDataPath, filterModelPath, outputPath);
    }

    private static void kalitateaEstimatuInteraktiboa() throws Exception {
        String modelPath = eskatuTestua("Kargatu beharreko model path-a (.model)");
        String metricsOutPath = eskatuTestua("Metrikak gordetzeko fitxategia");

        boolean erabiliStratifiedRepeatedHoldOut = eskatuBaiEz("Stratified Repeated Hold-Out erabili nahi duzu? (bai/ez)");
        Classifier modeloa = (Classifier) SerializationHelper.read(modelPath);

        if (erabiliStratifiedRepeatedHoldOut) {

            String trainPath = eskatuTestua("TRAIN ARFF path-a");
            String devPath = eskatuTestua("DEV ARFF path-a");

            Instances train = kargatuInstantziak(trainPath);
            Instances dev = kargatuInstantziak(devPath);

            int repeats = Integer.parseInt(eskatuLehenetsia("Errepikapen kopurua", "10"));
            Double ratio = Double.parseDouble(eskatuLehenetsia("Train ratio (0-1)", "0.8"));
            System.out.println("Datuen split bakoitza karpeta tenporal batean gordeko da (src/data/tmp) ");

            kalitateEstimazioa.stratifiedRepeatedHoldOut(train, dev,repeats, ratio,42,"src/data/tmp",metricsOutPath,modeloa);
        }
        else {

            String bekTrainPath = eskatuTestua("Bektorizatutako TRAIN ARFF path-a");
            String bekDevPath = eskatuTestua("Bektorizatutako DEV ARFF path-a");

            Instances trainBek = kargatuInstantziak(bekTrainPath);
            Instances devBek = kargatuInstantziak(bekDevPath);

            boolean erabiliEzZintzoa = eskatuBaiEz("Ez zintzoa erabili nahi duzu? (bai/ez)");
            if (erabiliEzZintzoa) {
                kalitateEstimazioa.ezZintzoa(trainBek, devBek, modeloa, metricsOutPath);
            } else {
                kalitateEstimazioa.holdOut(trainBek, devBek, modeloa, metricsOutPath);
                System.out.println("Kalitate metrikak gordeta: " + metricsOutPath);
            }
        }
    }


    /*private static void pipelineOsoaInteraktiboa() throws Exception {
        System.out.println("\n--- PIPELINE OSOA ---");
5
        boolean preprozesatu = eskatuBaiEz("TXT fitxategietatik ARFF sortu nahi duzu? (bai/ez)");

        String rawTrainPath = eskatuLehenetsia("Raw TRAIN ARFF path-a", "src/data/arff/SMS_SpamCollection.train.arff");
        String rawDevPath = eskatuLehenetsia("Raw DEV ARFF path-a", "src/data/arff/SMS_SpamCollection.dev.arff");
        String rawTestPath = eskatuLehenetsia("Raw TEST ARFF path-a", "src/data/arff/SMS_SpamCollection.test_blind.arff");

        if (preprozesatu) {
            String txtTrainPath = eskatuLehenetsia("TRAIN TXT path-a", "src/data/txt/SMS_SpamCollection.train.txt");
            String txtDevPath = eskatuLehenetsia("DEV TXT path-a", "src/data/txt/SMS_SpamCollection.dev.txt");
            String txtTestPath = eskatuLehenetsia("TEST TXT path-a", "src/data/txt/SMS_SpamCollection.test_blind.txt");

            dataProcessor.sms2Arff(txtTrainPath, rawTrainPath, false);
            dataProcessor.sms2Arff(txtDevPath, rawDevPath, false);
            dataProcessor.sms2Arff(txtTestPath, rawTestPath, true);
        }

        boolean erabiliStratifiedRepeatedHoldOut = eskatuBaiEz("Stratified Repeated Hold-Out erabili nahi duzu? (bai/ez)");
        String metricsOutPath = eskatuLehenetsia("Metriken irteera path-a", "src/data/results/metrikak.txt");

        if (erabiliStratifiedRepeatedHoldOut) {
            int repeats = Integer.parseInt(eskatuLehenetsia("Errepikapen kopurua", "10"));
            double trainRatio = Double.parseDouble(eskatuLehenetsia("Train ratio (0-1)", "0.8"));
            int seed = Integer.parseInt(eskatuLehenetsia("Hasierako hazia (seed)", "42"));
            String tempDirPath = eskatuLehenetsia("Aldi baterako fitxategien karpeta", "src/data/tmp");

            System.out.println("\n[SRHO] Stratified Repeated Hold-Out egiten...");
            kalitateEstimazioa.stratifiedRepeatedHoldOut(
                    rawTrainPath,
                    rawDevPath,
                    repeats,
                    trainRatio,
                    seed,
                    tempDirPath,
                    metricsOutPath
            );

            System.out.println("Pipeline osoa amaituta (SRHO modua).");
            System.out.println(" - Metrikak: " + metricsOutPath);
            return;
        }

        String bekTrainPath = eskatuLehenetsia("BEK TRAIN ARFF path-a", "src/data/arff/SMS_SpamCollection.bektrain.arff");
        String bekDevPath = eskatuLehenetsia("BEK DEV ARFF path-a", "src/data/arff/SMS_SpamCollection.bekdev.arff");
        String bekTestPath = eskatuLehenetsia("BEK TEST ARFF path-a", "src/data/arff/SMS_SpamCollection.bektest_blind.arff");
        String multiFilterPath = eskatuLehenetsia("MultiFilter path-a", "src/data/model/multiFilter.model");
        String modelOutPath = eskatuLehenetsia("Azken model path-a", "src/data/model/final.model");

        System.out.println("\n[1/4] Raw datuak bektorizatzen...");
        dataProcessor.bektorizatu(rawTrainPath, bekTrainPath, multiFilterPath, true);
        dataProcessor.bektorizatu(rawDevPath, bekDevPath, multiFilterPath, false);
        dataProcessor.bektorizatu(rawTestPath, bekTestPath, multiFilterPath, false);

        System.out.println("[2/4] Parametro optimoak bilatzen...");
        Instances train = kargatuInstantziak(bekTrainPath);
        Instances dev = kargatuInstantziak(bekDevPath);
        String[] parametroak = sailkatzailea.parametroOptimoakLortu(new Instances[]{train, dev, null});

        System.out.println("[3/4] Eredua entrenatzen eta gordetzen...");
        Classifier modeloa = sortuMLP(parametroak);
        modeloa.buildClassifier(train);
        SerializationHelper.write(modelOutPath, modeloa);

        System.out.println("[4/4] Kalitate estimazioa egiten...");
        kalitateEstimazioa.holdOut(train, dev, modeloa, metricsOutPath);

        System.out.println("Pipeline osoa amaituta.");
        System.out.println(" - Modeloa: " + modelOutPath);
        System.out.println(" - Metrikak: " + metricsOutPath);
    }
     */

    //CUIDADO
    private static Instances kargatuInstantziak(String path) throws Exception {
        Instances data = new DataSource(path).getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    //CUIDADO
    /*private static Classifier sortuMLP(String[] parametroak) {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setNominalToBinaryFilter(false);
        mlp.setSeed(42);
        mlp.setHiddenLayers(parametroak[0]);
        mlp.setLearningRate(Double.parseDouble(parametroak[1]));
        mlp.setMomentum(Double.parseDouble(parametroak[2]));
        mlp.setValidationSetSize(20);
        mlp.setValidationThreshold(15);
        return mlp;
    }
    */
    private static String eskatuTestua(String mezua) {
        System.out.print(mezua + ": ");
        return scanner.nextLine().trim();
    }

    private static String eskatuLehenetsia(String mezua, String balioLehenetsia) {
        System.out.print(mezua + " [" + balioLehenetsia + "]: ");
        String sarrera = scanner.nextLine().trim();
        return sarrera.isEmpty() ? balioLehenetsia : sarrera;
    }

    private static boolean eskatuBaiEz(String mezua) {
        while (true) {
            String balioa = eskatuTestua(mezua).toLowerCase();
            if ("bai".equals(balioa) || "ba".equals(balioa) || "y".equals(balioa) || "yes".equals(balioa)) {
                return true;
            }
            if ("ez".equals(balioa) || "e".equals(balioa) || "n".equals(balioa) || "no".equals(balioa)) {
                return false;
            }
            System.out.println("Mesedez, erantzun bai/ez.");
        }
    }

}
