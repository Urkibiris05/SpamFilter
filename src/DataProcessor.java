import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stemmers.Stemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.Tokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.io.PrintWriter;
import java.util.Scanner;

public class DataProcessor {

    public void sms2Arff(String pTxtPath, String pArffPath, boolean isClassBlind) throws Exception {
        String txtPath = pTxtPath;
        String arffPath = pArffPath;

        try (Scanner sc = new Scanner(new File(txtPath), "UTF-8");
             PrintWriter pw = new PrintWriter(new File(arffPath), "UTF-8")) {

            pw.println("@relation sms_spam_proiektua\n");
            pw.println("@attribute Text string");
            pw.println("@attribute class_label {ham, spam}\n");
            pw.println("@data");

            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                if (line.trim().isEmpty()) continue;

                String label;
                String text;

                if (isClassBlind) {
                    text = line.trim();
                    label = "?";
                } else {
                    String[] parts = line.split("\t");
                    if (parts.length < 2) continue;
                    label = parts[0].trim();
                    text = parts[1].trim();
                }

                text = text.replace("\"", "'");

                pw.println("\"" + text + "\"," + label);
            }
            System.out.println("Fitxategia ondo sortu da: " + arffPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void instantziakAztertu(String dataPath, String etapaIzena) throws Exception {
        Instances data = new DataSource(dataPath).getDataSet();

        // Asegurar que la clase está definida (suele ser el último atributo en los datos crudos)
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

        System.out.println("======================================================");
        System.out.println("📊 DATU GORDINEN ANALISIA - ETAPA: " + etapaIzena.toUpperCase());
        System.out.println("======================================================");
        System.out.println("Instantzia kopuru osoa (SMS): " + data.numInstances());
        System.out.println("Atributuen kopurua (Zutabeak): " + data.numAttributes());

        // Si hay clase y es nominal (Spam/Ham), mostramos la distribución
        if (data.classIndex() != -1 && data.classAttribute().isNominal()) {
            System.out.println("Klase banaketa:");
            weka.core.AttributeStats estatistikak = data.attributeStats(data.classIndex());

            for (int i = 0; i < data.classAttribute().numValues(); i++) {
                String klaseaBalioa = data.classAttribute().value(i);
                int kopurua = estatistikak.nominalCounts[i];
                double ehunekoa = (double) kopurua / data.numInstances() * 100;

                // Ignorar el valor "?" del Test Blind en el recuento si aparece
                if (kopurua > 0 || !klaseaBalioa.equals("?")) {
                    System.out.printf("  - %s: %d instantzia (%.2f%%)\n", klaseaBalioa, kopurua, ehunekoa);
                }
            }
        } else {
            System.out.println("Klase atributua: DEFINITU GABE edo TEST BLIND");
        }
        System.out.println("======================================================\n");
    }

    public void bektorizazioaAztertu(String dataPath, String etapaIzena) throws Exception {
        Instances data = new DataSource(dataPath).getDataSet();
        if (data.classIndex() == -1) data.setClassIndex(0);

        System.out.println("======================================================");
        System.out.println("🔎 BEKTORIZAZIOAREN ANALISIA - ETAPA: " + etapaIzena.toUpperCase());
        System.out.println("======================================================");

        // Restamos 1 porque un atributo es la clase, el resto son palabras
        int hiztegiTamaina = data.numAttributes() - 1;
        System.out.println("Hiztegiaren tamaina (Sortutako hitzak): " + hiztegiTamaina);

        // Análisis InfoGain (Solo si la clase tiene valores conocidos, en Test Blind no se puede)
        if (data.classIndex() != -1 && data.numInstances() > 0 && !data.instance(0).isMissing(data.classIndex())) {
            System.out.println("\nTop 10 atributu (hitz) onenak (Information Gain arabera):");
            inprimatuTop10InfoGain(data);
        } else {
            System.out.println("\nEzin da InfoGain kalkulatu klaseak ez duelako baliorik (Test Blind da?).");
        }
        System.out.println("======================================================\n");
    }

    public void bektorizatu(String rawDataPath, String bekDataPath, String dicFilePath, boolean isDicNull) throws Exception {
        Instances rawData = new DataSource(rawDataPath).getDataSet();
        if (rawData.classIndex() == -1) rawData.setClassIndex(rawData.numAttributes() - 1);

        Instances bekData = null;
        int wordsToKeep = 1000;


        if (isDicNull){
            StringToWordVector filter = new StringToWordVector();
            File dicFile = new File(dicFilePath);
            dicFile.createNewFile();
            filter.setDictionaryFileToSaveTo(dicFile);

            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setTFTransform(true);
            filter.setIDFTransform(true);
            // filter.setWordsToKeep(wordsToKeep);

            // Stemmer-a
            IteratedLovinsStemmer stemmer = new IteratedLovinsStemmer();
            filter.setStemmer(stemmer);

            // Tokenizer
            AlphabeticTokenizer tokenizer = new AlphabeticTokenizer();
            filter.setTokenizer(tokenizer);

            // Stop word-ak (esanahi handirik ematen ez duten hitz arruntak)
            // Wekak automatikoki ezabatuko ditu "the", "a", "an", "in"... bezalako hitzak
            Rainbow stopWords = new Rainbow();
            filter.setStopwordsHandler(stopWords);

            // Datu sorta bektorizatua itzuli
            filter.setInputFormat(rawData);
            bekData = Filter.useFilter(rawData, filter);

        } else {
            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            File dicFile = new File(dicFilePath);
            dicFile.createNewFile();
            filter.setDictionaryFile(dicFile);

            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setTFTransform(true);
            filter.setIDFTransform(true);
            // filter.setWordsToKeep(wordsToKeep);

            // Stemmer-a
            IteratedLovinsStemmer stemmer = new IteratedLovinsStemmer();
            filter.setStemmer(stemmer);

            // Tokenizer
            AlphabeticTokenizer tokenizer = new AlphabeticTokenizer();
            filter.setTokenizer(tokenizer);

            // Stop word-ak (esanahi handirik ematen ez duten hitz arruntak)
            // Wekak automatikoki ezabatuko ditu "the", "a", "an", "in"... bezalako hitzak
            Rainbow stopWords = new Rainbow();
            filter.setStopwordsHandler(stopWords);

            // Datu sorta bektorizatua itzuli
            filter.setInputFormat(rawData);
            bekData = Filter.useFilter(rawData, filter);
        }

        ArffSaver saver = new ArffSaver();
        File bekDataFile = new File(bekDataPath);
        if (!bekDataFile.exists()) bekDataFile.createNewFile();
        saver.setFile(bekDataFile);
        saver.setInstances(bekData);
        saver.writeBatch();
    }



    // =========================================================================
    // 🚀 LABORATORIO AUTOMÁTICO: BUSCADOR DE PARÁMETROS
    // =========================================================================

    public void parametroBilatzailea(String rawDataPath) throws Exception {
        Instances rawData = new DataSource(rawDataPath).getDataSet();
        if (rawData.classIndex() == -1) rawData.setClassIndex(rawData.numAttributes() - 1);

        System.out.println("\n======================================================");
        System.out.println("🚀 PARAMETRO BILATZAILE AUTOMATIKOA (ESPERIMENTUAK)");
        System.out.println("======================================================\n");

        // ---------------------------------------------------------
        // ESPERIMENTUA A: Adierazpen Matematikoa (Nola kontatu hitzak)
        // ---------------------------------------------------------
        System.out.println(">>> ESPERIMENTUA A: Adierazpen Matematikoa <<<");

        // A1. BoW Bitarra (Dago edo ez dago)
        StringToWordVector filterA1 = createBaseFilter();
        filterA1.setOutputWordCounts(false);
        runAndEvaluate(rawData, filterA1, "A1. Bag of Words Bitarra (Agerpena bakarrik)");

        // A2. TF (Maiztasuna - Zenbat aldiz agertzen da)
        StringToWordVector filterA2 = createBaseFilter();
        filterA2.setOutputWordCounts(true);
        filterA2.setTFTransform(false);
        filterA2.setIDFTransform(false);
        runAndEvaluate(rawData, filterA2, "A2. Term Frequency (TF - Maiztasuna)");

        // A3. TF-IDF (Garrantziaren pisu matematikoa)
        StringToWordVector filterA3 = createBaseFilter();
        filterA3.setOutputWordCounts(true);
        filterA3.setTFTransform(true);
        filterA3.setIDFTransform(true);
        runAndEvaluate(rawData, filterA3, "A3. TF-IDF (Gure oinarri edo baseline berria)");


        // ---------------------------------------------------------
        // ESPERIMENTUA B: N-Gramak (Testuingurua harrapatu)
        // ---------------------------------------------------------
        System.out.println("\n>>> ESPERIMENTUA B: Testuingurua (N-Gramak) <<<");
        NGramTokenizer ngram = null;
        for (int i=2; i<=4; i++){
            StringToWordVector filterB = createBaseFilter();
            filterB.setOutputWordCounts(true); filterB.setTFTransform(true); filterB.setIDFTransform(true);

            // Tokenizatzailea aldatu bi-gramak lortzeko (hitz pareak)
            ngram = new NGramTokenizer();
            ngram.setNGramMinSize(1);
            ngram.setNGramMaxSize(i); // Gehienez 2 hitz elkarrekin (adib. "call now")
            filterB.setTokenizer(ngram);

            runAndEvaluate(rawData, filterB, "B1. TF-IDF + Bi-gramak (1-"+i+" hitz)");
        }



        // ---------------------------------------------------------
        // ESPERIMENTUA C: Stemming-aren Eragina (Erroak elkartu)
        // ---------------------------------------------------------
        System.out.println("\n>>> ESPERIMENTUA C: Stemming-aren Eragina <<<");

        StringToWordVector filterC = createBaseFilter();
        filterC.setOutputWordCounts(true); filterC.setTFTransform(true); filterC.setIDFTransform(true);
        filterC.setTokenizer(ngram); // Bi-gramak mantenduko ditugu onak badira

        // Stemmer-a gehitu
        IteratedLovinsStemmer stemmer = new IteratedLovinsStemmer();
        filterC.setStemmer(stemmer);

        runAndEvaluate(rawData, filterC, "C1. TF-IDF + Bi-gramak + Lovins Stemmer");

        // ---------------------------------------------------------
        // ESPERIMENTUA D: Grid Search (Stemmer eta Tokenizer konbinazioak)
        // ---------------------------------------------------------
        System.out.println("\n>>> ESPERIMENTUA D: Konbinazioen Sareko Bilaketa (Grid Search) <<<");

        // 1. Probatu nahi ditugun Tokenizatzaileak
        Tokenizer[] tokenizers = {
                new WordTokenizer(),
                new AlphabeticTokenizer()
        };
        String[] tokIzenak = {"WordTokenizer (Zenbakiak mantendu)", "AlphabeticTokenizer (Letrak bakarrik)"};

        // 2. Probatu nahi ditugun Stemmer-ak
        Stemmer[] stemmers = {
                null, // Stemmer-ik ez erabili
                new LovinsStemmer(),
                new IteratedLovinsStemmer()
        };
        String[] stemIzenak = {"Ez (Null)", "Lovins", "IteratedLovins"};

        // 3. Konbinazio guztiak probatzeko begizta (Loop)
        for (int t = 0; t < tokenizers.length; t++) {
            for (int s = 0; s < stemmers.length; s++) {

                StringToWordVector filterD = createBaseFilter();
                // Gure baseline onena aplikatu (TF-IDF)
                filterD.setOutputWordCounts(true);
                filterD.setTFTransform(true);
                filterD.setIDFTransform(true);

                // Parametroak ezarri
                filterD.setTokenizer(tokenizers[t]);
                if (stemmers[s] != null) {
                    filterD.setStemmer(stemmers[s]);
                }

                String esperimentuIzenburua = "D - Tok: " + tokIzenak[t] + " | Stem: " + stemIzenak[s];
                runAndEvaluate(rawData, filterD, esperimentuIzenburua);
            }
        }
    }

    /**
     * Oinarrizko filtroa sortzen du, konfigurazio komunak ezarriz (minuskulak, stopwords...)
     */
    private StringToWordVector createBaseFilter() {
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);

        // StopWords ezinbestekoa da zaborra kentzeko
        Rainbow stopWords = new Rainbow();
        filter.setStopwordsHandler(stopWords);

        // Nahiko handia uzten dugu 1000 hitz onenak ikusteko
        filter.setWordsToKeep(Integer.MAX_VALUE);
        return filter;
    }

    /**
     * Filtro bat aplikatzen du eta zuzenean InfoGain-a kalkulatzen du pantailaratzeko
     */
    private void runAndEvaluate(Instances rawData, StringToWordVector filter, String expName) throws Exception {
        System.out.println("\n--- " + expName + " ---");

        // 1. Bektorizazioa aplikatu
        filter.setInputFormat(rawData);
        Instances vectorData = Filter.useFilter(rawData, filter);

        // Klasearen indizea ziurtatu
        Attribute classAttr = vectorData.attribute("class_label");
        if (classAttr != null) vectorData.setClassIndex(classAttr.index());
        else vectorData.setClassIndex(0);

        System.out.println("Hiztegi tamaina: " + (vectorData.numAttributes() - 1) + " atributu");

        // 2. InfoGain kalkulatu eta inprimatu
        inprimatuTop10InfoGain(vectorData);

    }

    // ---------------------------------------------------------
    // METODO AUXILIAR (El que hace el cálculo real)
    // ---------------------------------------------------------
    private void inprimatuTop10InfoGain(Instances data) {
        try {
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            eval.buildEvaluator(data);

            Ranker ranker = new Ranker();
            int[] topAtributuak = ranker.search(eval, data);

            int inprimatzekoKopurua = Math.min(10, topAtributuak.length);
            int count = 0;
            for (int i = 0; i < topAtributuak.length && count < inprimatzekoKopurua; i++) {
                int attrIndex = topAtributuak[i];
                if (attrIndex == data.classIndex()) continue; // Saltarnos la clase

                double infoGainBalioa = eval.evaluateAttribute(attrIndex);
                String hitza = data.attribute(attrIndex).name();
                System.out.printf("  %d. %-15s (InfoGain: %.4f)\n", (count+1), hitza, infoGainBalioa);
                count++;
            }
        } catch (Exception e) {
            System.out.println("  Ezin izan da InfoGain kalkulatu: " + e.getMessage());
        }
    }
}
