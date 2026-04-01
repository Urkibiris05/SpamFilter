import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Iragarpenak {
        DataProcessor dataProcessor = new DataProcessor();
    public Iragarpenak() {

    }

    /**
     * Test instantzien iragarpenak kalkulatu eta txt fitxategian idazten ditu.
     *
     * <p>Metodo honek test multzo bektorizatua eta jatorrizko test multzoa erabiltzen ditu:
     * bektorizatua klasifikazioak egiteko eta jatorrizko testua mezu originala lortzeko.</p>
     *
     * <p><b>Garrantzitsua:</b> {@code testBek} eta {@code test} instantzien ordena bera izan behar da.</p>
     *
     *
     * @param test test multzo jatorrizoa (mezuak lortzeko)
     * @param mlp entrenatu nahi den Weka Classifier-a
     * @param out txt irteera fitxategiaren bidea
     * @throws Exception irakurketa, idazketa edo klasifikazioan errorea gertatzen bada
     */
    public void Iragarpenak(String test, Classifier mlp, String multiFilterPath, String out) throws Exception {

        dataProcessor.sms2Arff(test, "SMS_SpamCollection.test_blind.arff",true);
        System.out.println("ARFF fitxategia hemen sortu da: ./SMS_SpamCollection.test_blind.arff");

        dataProcessor.bektorizatu("SMS_SpamCollection.test_blind.arff",
                        "SMS_SpamCollection.bektest_blind.arff",
                        multiFilterPath,false);

        ConverterUtils.DataSource sourceBekTest = new ConverterUtils.DataSource("SMS_SpamCollection.bektest_blind.arff");
        Instances testBek = sourceBekTest.getDataSet();

        if (testBek.classIndex() == -1) {
            testBek.setClassIndex(testBek.numAttributes() - 1);
        }

        //Irteera fitxategia
        PrintWriter writer = new PrintWriter(new FileWriter(out));
        writer.println("Instantzia,Iragarpena");

        System.out.println("[INFO] Iragarpenak egiten...");

        //Instantziak iteratu
        for (int i = 0; i < testBek.numInstances(); i++) {


            Instance instBek = testBek.instance(i);


            double clsLabel = mlp.classifyInstance(instBek);

            //Predikzioa lortu
            String prediction = testBek.classAttribute().value((int) clsLabel);

            writer.println((i + 1) +  "\"," + prediction);
        }

        writer.close();
        System.out.println("[OK] Iragarpenak fitxategi honetan gorde dira: " + out);
    }
}



