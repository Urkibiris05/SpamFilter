import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Iragarpenak {

    public Iragarpenak() {

    }

    public void Iragarpenak(Instances testBek, Instances test, Classifier mlp, String out) throws Exception {
        // 1. Configurar los índices de clase en ambos conjuntos de datos
        if (testBek.classIndex() == -1) {
            testBek.setClassIndex(testBek.numAttributes() - 1);
        }
        // 'test' es el conjunto original (el que tiene el atributo Text como String)
        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() - 1);
        }

        // 2. Preparar el escritor de archivos
        PrintWriter writer = new PrintWriter(new FileWriter(out));
        writer.println("Instantzia,Mezua,Iragarpena,Konfiantza");

        System.out.println("[INFO] Iragarpenak egiten...");

        // 3. Iterar sobre las instancias
        // Usamos testBek.numInstances() para asegurar que recorremos las que el modelo va a procesar
        for (int i = 0; i < testBek.numInstances(); i++) {

            // INSTANCIA ORIGINAL (para el texto)
            Instance instOriginal = test.instance(i);
            // INSTANCIA VECTORIZADA (para el modelo)
            Instance instVectorizada = testBek.instance(i);

            // Extraer el texto original del atributo 0 de la instancia NO vectorizada
            String mezuaOriginala = instOriginal.stringValue(0);

            // CLASIFICAR usando la instancia vectorizada (la que el MLP entiende)
            double clsLabel = mlp.classifyInstance(instVectorizada);

            // Obtener confianza
            double[] distribution = mlp.distributionForInstance(instVectorizada);
            double confidence = distribution[(int) clsLabel];

            // Nombre de la clase predicha (ham/spam)
            String prediction = testBek.classAttribute().value((int) clsLabel);

            // Escribir en el CSV
            String mezuaEscaped = mezuaOriginala.replace("\"", "\"\"");
            writer.println((i + 1) + ",\"" + mezuaEscaped + "\"," + prediction + "," + String.format("%.4f", confidence));
        }

        writer.close();
        System.out.println("[OK] Iragarpenak fitxategi honetan gorde dira: " + out);
    }
}



