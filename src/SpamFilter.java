

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SpamFilter {

    private static DataProcessor dataProcessor = new DataProcessor();
    //SMS2Arff sms2Arff = new SMS2Arff();
    private static Sailkatzailea sailkatzailea = new Sailkatzailea();

    public static void main(String[] args) throws Exception {
        String rawTrainPath = "data/arff/SMS_SpamCollection.train.arff";
        DataSource source =  new DataSource(rawTrainPath);
        Instances rawTrain = source.getDataSet();
        if (rawTrain.classIndex() == -1) rawTrain.setClassIndex(rawTrain.numAttributes() - 1);

        Instances bekData = dataProcessor.bektorizatu(rawTrain);
        dataProcessor.instantziakAztertu(bekData, "LEHEN FROGA");

        Instances[] arff = sailkatzailea.arffKargatu(bekData); // Parametro arff
        String[] parametroak = sailkatzailea.parametroOptimoakLortu(arff);



    }

}
