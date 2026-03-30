import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;

public class Sailkatzailea {
    private DataProcessor dataProcessor = new DataProcessor();

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

    //NO SABER COMO SE HACE
    public void erregresioLineala(Instances[] instantziak) throws Exception {
        Instances train = instantziak[0];
        Instances dev = instantziak[1];

        System.out.println("======================================================");
        System.out.println("              📊 ERREGRESIO LINEALA ");
        System.out.println("======================================================");

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setNominalToBinaryFilter(true);  //Mirar que hace
        mlp.setHiddenLayers("1");
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.1);
        mlp.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(mlp, dev);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());
    }

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

                        System.out.println(i + " : proba abian dago. Ebaluazioa hurrengoa da:");
                        i++;
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setNominalToBinaryFilter(false);
                        //mlp.setNormalizeAttributes(true); // Aparicion datos (0-1)
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

    public Classifier sailkatzaileaSortu(String[] parametroak, String trainPath, String devPath, String rawDataPath, String bekDataPath, String dicFilePath, String outputPath) throws Exception{
        try {
            DataSource sourceTrain = new DataSource(trainPath);
            Instances train = sourceTrain.getDataSet();
            if (train.classIndex() == -1){
                train.setClassIndex(train.numAttributes() - 1);
            }
            DataSource sourceDev = new DataSource(devPath);
            Instances dev = sourceDev.getDataSet();
            if (dev.classIndex() == -1){
                dev.setClassIndex(dev.numAttributes() - 1);
            }
            Instances dataTotala = new Instances(train);
            dataTotala.addAll(dev);

            //Bektorizazioa
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataTotala);
            saver.setFile(new File(rawDataPath));
            saver.writeBatch();
            dataProcessor.bektorizatu(rawDataPath, bekDataPath, dicFilePath, true);
            DataSource sourceTotala = new DataSource(bekDataPath);
            Instances bekDataTotala = sourceTotala.getDataSet();
            if (bekDataTotala.classIndex() == -1){
                bekDataTotala.setClassIndex(0);
            }


            //.model finala sortu (Parametro hoberenekin)
            String hl = parametroak[0];
            Double lr = Double.parseDouble(parametroak[1]);
            Double m = Double.parseDouble(parametroak[2]);
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setHiddenLayers(hl);
            mlp.setLearningRate(lr);
            mlp.setMomentum(m);

            mlp.setValidationSetSize(20);
            mlp.setValidationThreshold(5);

            mlp.buildClassifier(dataTotala);

            //.model gorde entregatzeko
            SerializationHelper.write(outputPath, mlp);
            return mlp;
        } catch ( IOException e) {
            System.out.println("ERROREA: .model KalitateaEstimatzeko sortzean.");
            return null;
        }
    }

}
