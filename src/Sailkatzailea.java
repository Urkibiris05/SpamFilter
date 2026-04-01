import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;

/**
 * SMS Spam sailkapenerako modeloak kudeatzen dituen klasea.
 * Parametroen ekorketa eta eredu finalaren sorkuntza barne hartzen ditu.
 * @author Asier S
 * @version 1.0
 */
public class Sailkatzailea {
    private DataProcessor dataProcessor = new DataProcessor();

    /**
     * Entrenamendu eta garapen datuak (ARFF) kargatzen ditu fitxategi bideetatik.
     * @param args Fitxategien bideak dituen array-a (args[0]: train, args[1]: dev).
     * @return Bi Instances objektu dituen array-a [0: train, 1: dev].
     * @throws Exception Fitxategiak irakurtzean akatsen bat gertatzen bada.
     */
    public Instances[] arffKargatu(String[] args) throws Exception{
        try {
            Instances[] arff = new Instances[2];
            String trainFile = args[0];
            String devFile = args[1];
            DataSource sourceTrain = new DataSource(trainFile);
            Instances train = sourceTrain.getDataSet();
            if (train.classIndex() == -1){
                train.setClassIndex(train.numAttributes() - 1);
            }
            arff[0] = train;
            DataSource sourceDev = new DataSource(devFile);
            Instances dev = sourceDev.getDataSet();
            if (dev.classIndex() == -1){
                dev.setClassIndex(dev.numAttributes() - 1);
            }
            arff[1] = dev;
            return arff;
        } catch ( IOException e ) {
            System.out.println("ERROREA: arff fitxategiak kargatzen.");
            return null;
        }
    }

    /**
     * Parametroen ekorketa egiten du Precision metrika maximizatzeko.
     * Hidden layers, learning rate eta momentum konbinazio desberdinak probatzen ditu.
     * @param instantziak Ebaluaziorako erabiliko diren datuak (train eta dev).
     * @return Parametro optimoen array-a precision arabera (metrika guztien arabera egiteko imprimatutako nahasmen matrizei erreparatu) [0: HiddenLayers, 1: LearningRate, 2: Momentum].
     * @throws Exception Iterazio bakoitzean sailkatzaileak huts egiten badu.
     */
    public String[] parametroOptimoakLortu(Instances[] instantziak) throws Exception{
        try{
            Instances train = instantziak[0];
            Instances dev = instantziak[1];

            System.out.println("======================================================");
            System.out.println("              📊 PARAMETRO EKORKETA ");
            System.out.println("======================================================");

            String[] balioParametroa = new String[3];
            String[] hiddenLayersBalioak = {"5", "10", "15", "5,10", "10, 5", "5, 5, 5", "5,10,15", "15,10,5"};
            double[] learningRatesBalioak = {0.01, 0.05, 0.1, 0.2, 0.3};
            double[] momentumsBalioak = {0.1, 0.2};
            String bestH = "";
            Double bestL = 0.0;
            Double bestM = 0.0;
            Double bestPrecision = 0.0;

            //Klase spam-aren indizea
            int idxSpam = 0;
            int i = 1;

            for (String h : hiddenLayersBalioak) {
                for (Double l : learningRatesBalioak){
                    for (Double m : momentumsBalioak){

                        long hasiera = System.nanoTime();
                        System.out.println(i + " : proba abian dago. Ebaluazioa hurrengoa da:");
                        i++;
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setNominalToBinaryFilter(false);
                        //mlp.setNormalizeAttributes(true); // Aparicion datos (0-1)
                        //mlp.setDecay(true); //Reduce LR / Epoch
                        mlp.setSeed(42);
                        mlp.setHiddenLayers(h);
                        mlp.setLearningRate(l);
                        mlp.setMomentum(m);

                        mlp.setValidationSetSize(20);
                        mlp.setValidationThreshold(15); //Numero de veces que no tiene que mejorar para que pare

                        mlp.buildClassifier(train);

                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(mlp, dev);
                        double currentPrecision = eval.precision(idxSpam);
                        System.out.println(" ----- PROBAKO DENBORA -----");
                        double segundoak = (System.nanoTime() - hasiera) / 1000000000.0;
                        System.out.println("Exekuzioan emandako doenbora: "+segundoak);
                        System.out.println(" ----- PROBAKO PARAMETROAK -----");
                        System.out.println("HL: "+h+" | LR: "+l+" | M: "+m);
                        System.out.println(eval.toMatrixString());
                        System.out.println(eval.toClassDetailsString());

                        if (currentPrecision > bestPrecision){
                            bestPrecision = currentPrecision;
                            bestH = h;
                            bestL = l;
                            bestM = m;
                        }
                    }
                }
            }
            balioParametroa[0] = bestH;
            balioParametroa[1] = Double.toString(bestL);
            balioParametroa[2] = Double.toString(bestM);
            System.out.println(" ------ PARAMETRO OPTIMOAK -------");
            System.out.println("HL: "+bestH+" | LR: "+bestL+" | M: "+bestM+" | Precision: "+bestPrecision);
            return balioParametroa;
        } catch ( IOException e){
            System.out.println("ERROREA: parametro ekorketa egitean.");
            return null;
        }
    }

    /**
     * Parametro optimoak erabiliz, modelo optimoa sortzen du nahi diren datu sortekin eta diskoan gordetzen du.
     * @param parametroak Aukeratutako HL, LR eta Momentum balioak.
     * @param trainPath Entrenamendu datuen bidea.
     * @param devPath Garapen datuen bidea.
     * @param rawDataPath Datu guztiak batuko diren ARFF fitxategiaren bidea.
     * @param bekDataPath Bektorizazioaren ondoren lortuko den fitxategia baita non gordeko den.
     * @param filterPath Erabiliko den iragazkiaren bidea.
     * @param outputPath Eredua (.model) gordeko den bidea.
     * @return Entrenatutako modelo bat bueltatuko du ezarritako parametroen arabera.
     * @throws Exception Fitxategiak idaztean edo eredua eraikitzean akatsen bat badago.
     */
    public Classifier sailkatzaileaSortu(String[] parametroak, String trainPath, String devPath, String rawDataPath, String bekDataPath, String filterPath, String outputPath) throws Exception{
        try {
            DataSource sourceTrain = new DataSource(trainPath);
            Instances train = sourceTrain.getDataSet();
            if (train.classIndex() == -1) {
                train.setClassIndex(train.numAttributes() - 1);
            }
            boolean devBai = true;
            Instances dev = null;
            if (devPath.equals("")) {
                devBai = false;
            }
            if (devBai) {
                DataSource sourceDev = new DataSource(devPath);
                    dev = sourceDev.getDataSet();
                if (dev.classIndex() == -1) {
                    dev.setClassIndex(dev.numAttributes() - 1);
                }
            }

            Instances dataTotala = new Instances(train);
            if (devBai && dev != null) {
                dataTotala.addAll(dev);
            }

            // Bektorizazioa
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataTotala);
            saver.setFile(new File(rawDataPath));
            saver.writeBatch();
            dataProcessor.bektorizatu(rawDataPath, bekDataPath, filterPath, true);
            DataSource sourceTotala = new DataSource(bekDataPath);
            Instances bekDataTotala = sourceTotala.getDataSet();
            if (bekDataTotala.classIndex() == -1){
                bekDataTotala.setClassIndex(bekDataTotala.numAttributes()-1);
            }

            //.model finala sortu (Parametro hoberenekin)
            String hl = parametroak[0];
            Double lr = Double.parseDouble(parametroak[1]);
            Double m = Double.parseDouble(parametroak[2]);
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setNominalToBinaryFilter(false);
            mlp.setSeed(42);
            mlp.setHiddenLayers(hl);
            mlp.setLearningRate(lr);
            mlp.setMomentum(m);

            mlp.setValidationSetSize(20);
            mlp.setValidationThreshold(15);

            mlp.buildClassifier(bekDataTotala);

            // .model gorde entregatzeko
            SerializationHelper.write(outputPath, mlp);
            System.out.println("[OK] Modeloa ondo gorde da hemen: " + outputPath);
            return mlp;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}
