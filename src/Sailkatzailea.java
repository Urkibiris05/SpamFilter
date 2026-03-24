import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;

public class Sailkatzailea {

    public Instances[] arffKargatu(String[] args) throws Exception{
        try {
            Instances[] arff = new Instances[3];
            String trainFile = args[0];
            String devFile = args[1];
            String testFile = args[2];
            DataSource sourceTrain = new DataSource(trainFile);
            Instances train = sourceTrain.getDataSet();
            if (train.classIndex() == -1){
                train.setClassIndex(0);
            }
            arff[0] = train;
            DataSource sourceDev = new DataSource(devFile);
            Instances dev = sourceDev.getDataSet();
            if (dev.classIndex() == -1){
                dev.setClassIndex(0);
            }
            arff[1] = dev;
            DataSource sourceTest = new DataSource(testFile);
            Instances test = sourceTest.getDataSet();
            if (test.classIndex() == -1){
                test.setClassIndex(0);
            }
            arff[2] = dev;
            return arff;
        } catch ( IOException e ) {
            System.out.println("ERROREA: arff fitxategiak kargatzen.");
            return null;
        }
    }

    public String[] parametroOptimoakLortu(Instances[] arff) throws Exception{
        try{
            Instances train = arff[0];
            Instances dev = arff[1];
            String[] balioParametroa = new String[4];
            String[] hiddenLayersBalioak = {" ", "a", "o", "i", "5", "10", "20", "10, 5", "a, a"};
            double[] learningRatesBalioak = {0.01,0.05, 0.1, 0.3};
            double[] momentumsBalioak = {0.1, 0.2};
            int[] epochBalioak = {100, 300, 500};
            String bestH = "";
            Double bestL = 0.0;
            Double bestM = 0.0;
            Double bestPrecision = 0.0;
            int bestEpoch = 0;

            //Klase spam-aren indizea
            int idxSpam = 0;

            for (int e : epochBalioak) {
                for (String h : hiddenLayersBalioak){
                    for (Double l : learningRatesBalioak){
                        for (Double m : momentumsBalioak){
                            MultilayerPerceptron mlp = new MultilayerPerceptron();
                            mlp.setNominalToBinaryFilter(true);
                            mlp.setDecay(true);
                            mlp.setTrainingTime(e);
                            mlp.setHiddenLayers(h);
                            mlp.setLearningRate(l);
                            mlp.setMomentum(m);

                            mlp.buildClassifier(train);

                            Evaluation eval = new Evaluation(train);
                            eval.evaluateModel(mlp, dev);
                            double currentPrecision = eval.precision(idxSpam);
                            System.out.println("HL: "+h+" | LR: "+l+" | M: "+m+" | Epoch: "+e+" | Precision: "+currentPrecision);

                            if (currentPrecision < bestPrecision){
                                bestPrecision = currentPrecision;
                                bestH = h;
                                bestL = l;
                                bestM = m;
                                bestEpoch = e;
                            }
                        }
                    }
                }
            }
            balioParametroa[0] = bestH;
            balioParametroa[1] = Double.toString(bestL);
            balioParametroa[2] = Double.toString(bestM);
            balioParametroa[3] = Integer.toString(bestEpoch);
            System.out.println(" ------ PARAMETRO OPTIMOAK -------");
            System.out.println("HL: "+bestH+" | LR: "+bestL+" | M: "+bestM+" | Epoch: "+bestEpoch+" | Precision: "+bestPrecision);
            return balioParametroa;
        } catch ( IOException e){
            System.out.println("ERROREA: parametro ekorketa egitean.");
            return null;
        }
    }

    public String[] parametroOptimoakLortuHoldOut(Instances[] arff) throws Exception{
        try{
            // Datu sorta oso bat sortu Train + Dev
            Instances datuak1 = arff[0];
            Instances datuak2 = arff[1];
            Instances data = new Instances(datuak1);
            data.addAll(datuak2);

            String[] balioParametroa = new String[4];
            String[] hiddenLayersBalioak = {" ", "a", "o", "i", "5", "10", "20", "10, 5", "a, a"};
            double[] learningRatesBalioak = {0.01,0.05, 0.1, 0.3};
            double[] momentumsBalioak = {0.1, 0.2};
            String bestH = "";
            Double bestL = 0.0;
            Double bestM = 0.0;
            Double bestPrecision = 0.0;
            int bestEpoch = 0;
            int e = 1;

            //Klase spam-aren indizea
            int idxSpam = 0;
            for (String h : hiddenLayersBalioak){
                for (Double l : learningRatesBalioak){
                    for (Double m : momentumsBalioak){
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setNominalToBinaryFilter(true);
                        mlp.setDecay(true);
                        mlp.setTrainingTime(1000);
                        mlp.setHiddenLayers(h);
                        mlp.setLearningRate(l);
                        mlp.setMomentum(m);

                        // EarlyStopping
                        mlp.setValidationSetSize(10);  //Erreserbatutako % dev egiteko
                        mlp.setValidationThreshold(20); //Errore margina stop egin arte

                        mlp.buildClassifier(data);

                        Evaluation eval = new Evaluation(data);
                        eval.evaluateModel(mlp, data);
                        double currentPrecision = eval.precision(idxSpam);
                        System.out.println("HL: "+h+" | LR: "+l+" | M: "+m+" | Epoch: "+e+" | Precision: "+currentPrecision);

                        if (currentPrecision < bestPrecision){
                            bestPrecision = currentPrecision;
                            bestH = h;
                            bestL = l;
                            bestM = m;
                            bestEpoch = e;
                        }
                        e++;
                    }
                }
            }
            balioParametroa[0] = bestH;
            balioParametroa[1] = Double.toString(bestL);
            balioParametroa[2] = Double.toString(bestM);
            balioParametroa[3] = Integer.toString(bestEpoch);
            System.out.println(" ------ PARAMETRO OPTIMOAK -------");
            System.out.println("HL: "+bestH+" | LR: "+bestL+" | M: "+bestM+" | Epoch: "+bestEpoch+" | Precision: "+bestPrecision);
            return balioParametroa;
        } catch ( IOException e){
            System.out.println("ERROREA: parametro ekorketa egitean.");
            return null;
        }
    }

    public Classifier sailkatzaileaKalitateaSortu(String[] parametroak, Instances train, Instances dev, String path) throws Exception{
        try {
            //.model finala sortu (Parametro hoberenekin)
            String hl = parametroak[0];
            Double lr = Double.parseDouble(parametroak[1]);
            Double m = Double.parseDouble(parametroak[2]);
            int epoch = Integer.parseInt(parametroak[3]);
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setNominalToBinaryFilter(true);
            mlp.setDecay(true);
            mlp.setTrainingTime(epoch);
            mlp.setHiddenLayers(hl);
            mlp.setLearningRate(lr);
            mlp.setMomentum(m);

            // Train + Dev fusionatu
            Instances trainDev = new Instances(train);
            trainDev.addAll(dev);
            mlp.buildClassifier(trainDev);

            //.model gorde entregatzeko
            String outputPath = path;
            SerializationHelper.write(outputPath, mlp);
            return null;
        } catch ( IOException e) {
            System.out.println("ERROREA: .model KalitateaEstimatzeko sortzean.");
            return null;
        }
    }

    public Classifier sailkatzaileaFinalaSortu(String[] parametroak, Instances train, Instances dev, Instances test, String path) throws Exception{
        try {
            //.model finala sortu (Parametro hoberenekin)
            String hl = parametroak[0];
            Double lr = Double.parseDouble(parametroak[1]);
            Double m = Double.parseDouble(parametroak[2]);
            int epoch = Integer.parseInt(parametroak[3]);
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setNominalToBinaryFilter(true);
            mlp.setDecay(true);
            mlp.setTrainingTime(epoch);
            mlp.setHiddenLayers(hl);
            mlp.setLearningRate(lr);
            mlp.setMomentum(m);

            // Train + Dev + Test fusionatu
            Instances trainDevTest = new Instances(train);
            trainDevTest.addAll(dev);
            trainDevTest.addAll(test);
            mlp.buildClassifier(trainDevTest);

            //.model gorde entregatzeko
            String outputPath = path;
            SerializationHelper.write(outputPath, mlp);
            return null;
        } catch ( IOException e) {
            System.out.println("ERROREA: .model Entregatzeko sortzean.");
            return null;
        }
    }
}
