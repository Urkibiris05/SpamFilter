import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SpamFilter {

    private static DataProcessor dataProcessor = new DataProcessor();
    //SMS2Arff sms2Arff = new SMS2Arff();
    private static Sailkatzailea sailkatzailea = new Sailkatzailea();

    public static void main(String[] args) throws Exception {
        String rawTrainPath = "src/data/arff/SMS_SpamCollection.train.arff";
        String bekTrainPath = "src/data/arff/SMS_SpamCollection.bektrain.arff";


        // dataProcessor.sms2Arff(rawTrainPath);
        // dataProcessor.bektorizatu(rawTrainPath, bekTrainPath);
        //  dataProcessor.instantziakAztertu(bekTrainPath, "LEHEN FROGA");
        dataProcessor.parametroBilatzailea(rawTrainPath);

        /*
        Instances[] arff = sailkatzailea.arffKargatu(bekData); // Parametro arff
        String[] parametroak = sailkatzailea.parametroOptimoakLortu(arff);
        */


    }

}
