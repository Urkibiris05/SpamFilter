import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Iragarpenak {

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
     * @param testBek test multzo bektorizatua (ereduak prozesatzeko erabiltzen du)
     * @param test test multzo jatorrizoa (mezuak lortzeko)
     * @param mlp entrenatu nahi den Weka Classifier-a
     * @param out txt irteera fitxategiaren bidea
     * @throws Exception irakurketa, idazketa edo klasifikazioan errorea gertatzen bada
     */
    public void Iragarpenak(Instances testBek, Instances test, Classifier mlp, String out) throws Exception {

        if (testBek.classIndex() == -1) {
            testBek.setClassIndex(testBek.numAttributes() - 1);
        }

        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() - 1);
        }

        //Irteera fitxategia
        PrintWriter writer = new PrintWriter(new FileWriter(out));
        writer.println("Instantzia,Mezua,Iragarpena,Konfiantza");

        System.out.println("[INFO] Iragarpenak egiten...");

        //Instantziak iteratu
        for (int i = 0; i < testBek.numInstances(); i++) {

            Instance instHasi = test.instance(i);

            Instance instBek = testBek.instance(i);

            //Mezua lortu
            String mezua = instHasi.stringValue(0);

            double clsLabel = mlp.classifyInstance(instBek);

            double[] distribution = mlp.distributionForInstance(instBek);
            double confidence = distribution[(int) clsLabel];

            //Predikzioa lortu
            String prediction = testBek.classAttribute().value((int) clsLabel);

            String mezuaEscaped = mezua.replace("\"", "\"\"");
            writer.println((i + 1) + ",\"" + mezuaEscaped + "\"," + prediction + "," + String.format("%.4f", confidence));
        }

        writer.close();
        System.out.println("[OK] Iragarpenak fitxategi honetan gorde dira: " + out);
    }
}



