import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SpamFilter {

    private static DataProcessor dataProcessor = new DataProcessor();
    //SMS2Arff sms2Arff = new SMS2Arff();
    private static Sailkatzailea sailkatzailea = new Sailkatzailea();

    public static void main(String[] args) throws Exception {
        String rawTrainPath = "src/data/arff/SMS_SpamCollection.train.arff";
        String bekTrainPath = "src/data/arff/SMS_SpamCollection.bektrain.arff";
        String rawDevPath = "src/data/arff/SMS_SpamCollection.dev.arff";
        String bekDevPath = "src/data/arff/SMS_SpamCollection.bekdev.arff";
        String rawTestPath = "src/data/arff/SMS_SpamCollection.test_blind.arff";
        String bekTestPath = "src/data/arff/SMS_SpamCollection.bektest_blind.arff";

        String multiFilterPath = "src/data/model/multiFilter.model";

        // dataProcessor.sms2Arff(rawTrainPath);
        dataProcessor.bektorizatu(rawTrainPath, bekTrainPath, multiFilterPath, true);
        dataProcessor.bektorizatu(rawDevPath, bekDevPath, multiFilterPath, false);
        dataProcessor.bektorizatu(rawTestPath, bekTestPath, multiFilterPath, false);
        // dataProcessor.instantziakAztertu(bekTrainPath, "LEHEN FROGA");
        //dataProcessor.parametroBilatzailea(rawTrainPath);
        //dataProcessor.parametroBilatzaileaV2(rawTrainPath, rawDevPath);

        /*
        Instances[] arff = sailkatzailea.arffKargatu(bekData); // Parametro arff
        String[] parametroak = sailkatzailea.parametroOptimoakLortu(arff);
        */


    }

}
