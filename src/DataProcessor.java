import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SerializationHelper;
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
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.io.PrintWriter;
import java.util.Random;
import java.util.Scanner;

/**
 * SMSak Wekarekin sailkatzeko proiektuko aurreprozesatze eta analisi utilitateak.
 *
 * <p>Klase honek TXT->ARFF bihurketa-fluxua, instantzien analisia,
 * testuaren bektorizazioa eta parametro/filtro bilaketa-esperimentuak biltzen ditu.</p>
 */
public class DataProcessor {

    /**
     * SMSen TXT fitxategi bat ARFF formatura bihurtzen du.
     *
     * @param pTxtPath sarrerako TXT fitxategiaren bidea
     * @param pArffPath irteerako ARFF fitxategiaren bidea
     * @param isClassBlind true bada, TXTak ez du klase-etiketarik ("?" gordetzen da)
     * @throws Exception irakurketa/idazketa errorea gertatzen bada
     */
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

    /**
     * ARFF dataset baten oinarrizko estatistikak kontsolan erakusten ditu.
     *
     * @param dataPath aztertu beharreko ARFFaren bidea
     * @param etapaIzena logerako etaparen izen deskribatzailea
     * @throws Exception dataseta ezin bada kargatu
     */
    public void instantziakAztertu(String dataPath, String etapaIzena) throws Exception {
        Instances data = new DataSource(dataPath).getDataSet();

        // Asegurar que la clase está definida (suele ser el último atributo en los datos crudos)
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

        System.out.println("======================================================");
        System.out.println(" DATU GORDINEN ANALISIA - ETAPA: " + etapaIzena.toUpperCase());
        System.out.println("======================================================");
        System.out.println("Instantzia kopuru osoa (SMS): " + data.numInstances());
        System.out.println("Atributuen kopurua (Zutabeak): " + data.numAttributes());

        // Si hay clase y es nominal (Spam/Ham), mostramos la distribución
        if (data.classIndex() != -1 && data.classAttribute().isNominal()) {
            System.out.println("Klase banaketa:");
            AttributeStats estatistikak = data.attributeStats(data.classIndex());

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

    /**
     * Testu-dataset bat bektorizatzen du eta, dagokionean, emaitzako ARFFa eta filtroa gordetzen ditu.
     *
     * @param rawDataPath sarrerako ARFFaren bidea (testu gordina)
     * @param bekDataPath irteerako ARFF bektorizatuaren bidea
     * @param filterModelPath MultiFilter serializatua gorde/kargatzeko bidea
     * @param isTrain true bada filtroa entrenatu eta gorde; false bada dagoena berrerabili
     * @throws Exception kargatzean, filtratzean edo irteera idaztean huts egiten bada
     */
    public void bektorizatu(String rawDataPath, String bekDataPath, String filterModelPath, boolean isTrain) throws Exception {
        Instances rawData = new DataSource(rawDataPath).getDataSet();
        if (rawData.classIndex() == -1) rawData.setClassIndex(rawData.numAttributes() - 1);

        Instances bekData = null;

        if (isTrain) {
            System.out.println("  [TRAIN] datu sorta bektorizatzen eta filtroak aplikatzen...");

            // 1. STWV konfiguratu, V1 esperimentuetatik lortutako oinarrizko bektorizazioa
            StringToWordVector stwv = new StringToWordVector();
            stwv.setLowerCaseTokens(true);
            stwv.setOutputWordCounts(true);
            stwv.setTFTransform(true);
            stwv.setIDFTransform(true);
            stwv.setWordsToKeep(1500);
            stwv.setStemmer(new IteratedLovinsStemmer());
            stwv.setTokenizer(new AlphabeticTokenizer());
            stwv.setStopwordsHandler(new Rainbow());

            // 2. V2 esperimentuetatik lortutako atributuen hautapena konfiguratu
            AttributeSelection as = new AttributeSelection();
            InfoGainAttributeEval infoGain = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();
            ranker.setNumToSelect(500);
            as.setEvaluator(infoGain);
            as.setSearch(ranker);

            // 3. MultiFilter erabiliz bi filtroak bateratu
            MultiFilter multiFilter = new MultiFilter();
            multiFilter.setFilters(new Filter[]{stwv, as});

            // 4. Train multzoa bektorizatu filtro bateratuarekin (filtroak hiztegia ikasi)
            multiFilter.setInputFormat(rawData);
            bekData = Filter.useFilter(rawData, multiFilter);

            // 5. Filtro bateratua gorde gainontzeko datu sortak hiztegi berarekin bektorizatzeko
            SerializationHelper.write(filterModelPath, multiFilter);
            System.out.println("  [TRAIN] Filtro bateratua zuzen gordeta hurrengo helbidean: " + filterModelPath);

        } else {
            System.out.println("  [TEST/DEV] Filtro bateratua kargatzen eta datu berriak bektorizatzen...");

            // 1. Jada sortutako filtro bateratua kargatu
            MultiFilter multiFilter = (MultiFilter) SerializationHelper.read(filterModelPath);

            // 2. Filtroa zuzenean aplikatu
            bekData = Filter.useFilter(rawData, multiFilter);
        }

        // Ziurtatu instantzien klase atributua (class_label) amaieran mantentzen dela
        if (bekData.classIndex() == -1) {
            bekData.setClassIndex(bekData.numAttributes() - 1);
        }

        // Lortutako ARFF fitxategia gorde
        ArffSaver saver = new ArffSaver();
        File bekDataFile = new File(bekDataPath);
        if (!bekDataFile.exists()) bekDataFile.createNewFile();
        saver.setFile(bekDataFile);
        saver.setInstances(bekData);
        saver.writeBatch();

        System.out.println("  [OK] ARFF fitxategia zuzen sortuta: " + bekDataPath + " (" + (bekData.numAttributes() - 1) + " atributos)");
    }



    // =========================================================================
    // PARAMETRO BILATZAILE AUTOMATIKOA - ESPERIMENTUAK
    // =========================================================================

    /**
     * Testu-errepresentazioaren eta tokenizazio/stemming konfigurazioen oinarrizko esperimentuak exekutatzen ditu.
     *
     * @param rawDataPath entrenamendurako ARFF gordinaren bidea
     * @throws Exception datuak kargatzean edo filtroak aplikatzean erroreak badaude
     */
    public void parametroBilatzailea(String rawDataPath) throws Exception {
        Instances rawData = new DataSource(rawDataPath).getDataSet();
        if (rawData.classIndex() == -1) rawData.setClassIndex(rawData.numAttributes() - 1);

        System.out.println("\n======================================================");
        System.out.println("PARAMETRO BILATZAILE AUTOMATIKOA (ESPERIMENTUAK)");
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
     * Esperimentuetarako konfigurazio komunarekin StringToWordVector oinarrizko bat sortzen du.
     *
     * @return esperimentu bakoitzerako konfigura daitekeen oinarrizko filtroa
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
     * Bektorizazio-filtro bat aplikatu eta InfoGain bidez atributuen laburpena erakusten du.
     *
     * @param rawData sarrerako dataset gordina
     * @param filter aplikatu beharreko bektorizazio-filtroa
     * @param expName kontsolako trazetarako esperimentuaren izena
     * @throws Exception filtratzeak edo atributuen ebaluazioak huts egiten badu
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

    /**
     * Klasea ez diren eta InfoGain handiena duten 10 atributuak inprimatzen ditu.
     *
     * @param data klase-atributua definituta duen dataset bektorizatua
     */
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

    // =========================================================================
    // LABORATORIO V2: VECTORIZACIÓN + SELECCIÓN + MLP BASE
    // =========================================================================

    /**
     * TRAIN/DEV gaineko atributu-hautaketako laborategia exekutatzen du, bektorizazio konfigurazio finkoarekin.
     *
     * @param rawTrainPath TRAIN ARFF gordinaren bidea
     * @param rawDevPath DEV ARFF gordinaren bidea
     * @throws Exception dataseten kargak huts egiten badu
     */
    public void parametroBilatzaileaV2(String rawTrainPath, String rawDevPath) throws Exception {
        Instances trainRaw = new DataSource(rawTrainPath).getDataSet();
        if (trainRaw.classIndex() == -1) trainRaw.setClassIndex(trainRaw.numAttributes() - 1);

        Instances devRaw = new DataSource(rawDevPath).getDataSet();
        if (devRaw.classIndex() == -1) devRaw.setClassIndex(devRaw.numAttributes() - 1);

        System.out.println("\n======================================================");
        System.out.println("LABORATORIO V5: GRID SEARCH DE ATTRIBUTE SELECTION");
        System.out.println("======================================================\n");

        // --- 1. CONFIGURACIÓN BASE (Vectorización inamovible) ---
        StringToWordVector stwv = new StringToWordVector();
        stwv.setLowerCaseTokens(true);
        stwv.setOutputWordCounts(true);
        stwv.setTFTransform(true);
        stwv.setIDFTransform(true);
        stwv.setWordsToKeep(1500);
        stwv.setTokenizer(new AlphabeticTokenizer());
        stwv.setStemmer(new IteratedLovinsStemmer());
        stwv.setStopwordsHandler(new Rainbow());

        // =========================================================
        // FAMILIA A: EVALUADORES INDIVIDUALES (Requieren Ranker)
        // =========================================================
        System.out.println(">>> FAMILIA A: Evaluadores Individuales + Ranker <<<");

        ASEvaluation[] singleEvals = {
                new InfoGainAttributeEval(),
                new GainRatioAttributeEval(),
                new SymmetricalUncertAttributeEval(),
                new ReliefFAttributeEval()
        };
        String[] singleNames = {"InfoGain", "GainRatio", "SymmetricalUncertainty", "ReliefF (Basado en instancias)"};
        int[] topKValues = {100, 300, 500, 750, 1000}; // Probaremos a quedarnos con el Top 100 y Top 300

        for (int i = 0; i < singleEvals.length; i++) {
            for (int k : topKValues) {
                AttributeSelection filterA = new AttributeSelection();
                Ranker ranker = new Ranker();
                ranker.setNumToSelect(k);

                filterA.setEvaluator(singleEvals[i]);
                filterA.setSearch(ranker);

                String expName = "A - Eval: " + singleNames[i] + " | Búsqueda: Ranker (Top " + k + ")";
                ejecutarPipelineCompleto(trainRaw, devRaw, stwv, filterA, expName);
            }
        }

        // =========================================================
        // FAMILIA B: EVALUADORES DE SUBCONJUNTOS (CFS)
        // =========================================================
        System.out.println("\n>>> FAMILIA B: Evaluadores de Subconjuntos (CFS) + Búsquedas Complejas <<<");

        CfsSubsetEval cfsEval = new CfsSubsetEval();

        // 1. GreedyStepwise (Hacia adelante - Forward)
        GreedyStepwise greedyForward = new GreedyStepwise();
        greedyForward.setSearchBackwards(false);

        // 2. BestFirst (Busca en varias direcciones saltando ramas)
        BestFirst bestFirst = new BestFirst();

        ASSearch[] subsetSearches = {greedyForward, bestFirst};
        String[] searchNames = {"GreedyStepwise (Forward)", "BestFirst"};

        for (int j = 0; j < subsetSearches.length; j++) {
            AttributeSelection filterB = new AttributeSelection();
            filterB.setEvaluator(cfsEval);
            filterB.setSearch(subsetSearches[j]);

            String expName = "B - Eval: CFS | Búsqueda: " + searchNames[j];
            ejecutarPipelineCompleto(trainRaw, devRaw, stwv, filterB, expName);
        }
    }

    /**
     * Pipeline oso bat exekutatzen du: TRAIN bektorizatu, aukerako hautaketa aplikatu,
     * MLP bat entrenatu eta DEVen ebaluatu, transformazio-iragazki bera erabiliz.
     *
     * @param trainRaw TRAIN dataset gordina
     * @param devRaw DEV dataset gordina
     * @param stwv bektorizazioaren oinarrizko filtroa
     * @param asFilter atributu-hautaketako filtroa (null izan daiteke)
     * @param nombrePrueba logetarako etiketa deskribatzailea
     */
    private void ejecutarPipelineCompleto(Instances trainRaw,
                                          Instances devRaw,
                                          StringToWordVector stwv,
                                          AttributeSelection asFilter,
                                          String nombrePrueba) {
        try {
            System.out.println("\n▶ INICIANDO: " + nombrePrueba);
            long startTime = System.currentTimeMillis();

            // 1. APLICAR FILTROS SOLO AL TRAIN (Aprender el vocabulario)
            System.out.println("  [1/3] Vectorizando y seleccionando atributos en TRAIN...");
            stwv.setInputFormat(trainRaw);
            Instances trainVectorizado = Filter.useFilter(trainRaw, stwv);
            Instances trainFinal = trainVectorizado;

            if (asFilter != null) {
                asFilter.setInputFormat(trainVectorizado);
                trainFinal = Filter.useFilter(trainVectorizado, asFilter);
            }

            int numAtributos = trainFinal.numAttributes() - 1;
            System.out.println("  [ℹ] Tamaño del diccionario REAL seleccionado: " + numAtributos + " atributos.");

            // 2. Crear el clasificador final (MLP)
            System.out.println("  [2/3] Entrenando Multilayer Perceptron con TRAIN...");
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setHiddenLayers("5");
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.1);
            mlp.buildClassifier(trainFinal);

            // 3. PASAR EL DEV POR EL MISMO EMBUDO (Aplicar diccionario)
            System.out.println("  [3/3] Evaluando modelo con DEV...");
            Instances devVectorizado = Filter.useFilter(devRaw, stwv);
            Instances devFinal = devVectorizado;
            if (asFilter != null) {
                devFinal = Filter.useFilter(devVectorizado, asFilter);
            }

            // 4. Evaluar con Cross-Validation usando los DATOS CRUDOS (Raw Data)
            Evaluation eval = new Evaluation(trainFinal);
            eval.evaluateModel(mlp, devFinal);

            long endTime = System.currentTimeMillis();
            long tiempoSegundos = (endTime - startTime) / 1000;

            // Extraer índices de las clases (asegúrate de que los nombres coinciden con tu .arff)
            int spamIndex = trainFinal.classAttribute().indexOfValue("spam");
            int hamIndex = trainFinal.classAttribute().indexOfValue("ham");

            if (spamIndex == -1 || hamIndex == -1) {
                System.out.println("  [!] Aviso: No se encontraron las clases 'spam' o 'ham' con ese nombre exacto. Revisa mayúsculas/minúsculas.");
            }

            System.out.println("\n  RESULTADOS DETALLADOS (Sin Data Leakage):");
            System.out.printf("     Accuracy (Precisión Global): %.2f%%\n", eval.pctCorrect());

            if (spamIndex != -1 && hamIndex != -1) {
                System.out.println("\n     [Métricas clase SPAM (El objetivo principal)]");
                System.out.printf("      - Precision: %.4f (De todos los que marqué como Spam, ¿cuántos lo eran de verdad?)\n", eval.precision(spamIndex));
                System.out.printf("      - Recall:    %.4f (De todos los Spam reales que había, ¿cuántos logré cazar?)\n", eval.recall(spamIndex));
                System.out.printf("      - F-Measure: %.4f (Media armónica entre Precision y Recall)\n", eval.fMeasure(spamIndex));
                System.out.printf("      - AUC (ROC): %.4f (Área bajo la curva. >0.95 es excelente)\n", eval.areaUnderROC(spamIndex));

                System.out.println("\n     [Métricas clase HAM (Mensajes legítimos)]");
                System.out.printf("      - Precision: %.4f\n", eval.precision(hamIndex));
                System.out.printf("      - Recall:    %.4f\n", eval.recall(hamIndex));
                System.out.printf("      - F-Measure: %.4f\n", eval.fMeasure(hamIndex));
            }

            System.out.println("\n" + eval.toMatrixString("     📉 MATRIZ DE CONFUSIÓN"));

            System.out.println("     Tiempo total del proceso: " + tiempoSegundos + " segundos.");
            System.out.println("------------------------------------------------------\n");

        } catch (Exception e) {
            System.out.println("  Error en la prueba: " + e.getMessage());
            e.printStackTrace();
        }
    }
}



