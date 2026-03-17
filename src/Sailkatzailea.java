import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;

public class Sailkatzailea {

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

    public String[] parametroOptimoakLortu(Instances[] arff) throws Exception{
        try{
            Instances train = arff[0];
            Instances dev = arff[1];
            String[] balioParametroa = new String[3];
            String[] hiddenLayersBalioak = {" ","a","i","o","t","a, o","o, o","t, o*2","i, a, o"};
            double[] learningRatesBalioak = {0.01,0.05, 0.1, 0.3};
            double[] momentumsBalioak = {0.1, 0.2};
            String bestH = "";
            Double bestL = 0.0;
            Double bestM = 0.0;
            Double bestAccuracy = 0.0;
            for (String h : hiddenLayersBalioak){
                for (Double l : learningRatesBalioak){
                    for (Double m : momentumsBalioak){
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setNominalToBinaryFilter(true);
                        mlp.setDecay(true);
                        mlp.setTrainingTime(500);
                        mlp.setHiddenLayers(h);
                        mlp.setLearningRate(l);
                        mlp.setMomentum(m);

                        mlp.buildClassifier(train);

                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(mlp, dev);
                        double currentAccuracy = eval.pctCorrect();
                        System.out.println("HL: "+h+" | LR: "+l+" | M: "+m+" | Accuracy: "+currentAccuracy);

                        if (currentAccuracy < bestAccuracy){
                            bestAccuracy = currentAccuracy;
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
            System.out.println("HL: "+bestH+" | LR: "+bestL+" | M: "+bestM+" | Accuracy: "+bestAccuracy);
            return balioParametroa;
        } catch ( IOException e){
            System.out.println("ERROREA: parametro ekorketa egitean.");
            return null;
        }
    }
}
