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

        String dicFilePath = "src/data/txt/train_dictionary.txt";

        // dataProcessor.sms2Arff(rawTrainPath);
        //dataProcessor.bektorizatu(rawTrainPath, bekTrainPath, dicFilePath, true);
        //dataProcessor.bektorizatu(rawDevPath, bekDevPath, dicFilePath, false);
        //dataProcessor.bektorizatu(rawTestPath, bekTestPath, dicFilePath, false);
        // dataProcessor.instantziakAztertu(bekTrainPath, "LEHEN FROGA");
        // dataProcessor.parametroBilatzailea(rawTrainPath);

        /*
        Instances[] arff = sailkatzailea.arffKargatu(bekData); // Parametro arff
        String[] parametroak = sailkatzailea.parametroOptimoakLortu(arff);
        */

        String[] arff = new String[3];
        arff[0] = bekTrainPath;
        arff[1] = bekDevPath;
        arff[2] = bekTestPath;

        Instances[] instantziak = new Instances[3];
        instantziak = sailkatzailea.arffKargatu(arff);
        String[] parametroak = new String[4];
        parametroak = sailkatzailea.parametroOptimoakLortu(instantziak);
        //sailkatzailea.sailkatzaileaSortu(parametroak, rawTrainPath, rawDevPath, "/dicFIlePath", "outputPath");
        //sailkatzailea.erregresioLineala(instantziak);
    }

}
