# SpamFilter CLI

Fitxategi honetan, "SpamFilter.jar" komando-lineako aplikazioaren erabilera eta komandoak azaltzen dira. 
Aplikazio honek SMS spam detekzioarekin lotutako hainbat prozesu exekutatzeko aukera ematen du, 
hala nola datuen prestaketa, bektorizazioa, parametro bilaketa, entrenamendua eta ebaluazioa.

## Oinarrizko erabilera

```bash
java -jar SpamFilter.jar --run <komando1,komando2,...> [--gakoa balioa ...]
```

`--gakoa=balioa` formatua ere erabil dezakezu.

Laguntza:

```bash
java -jar SpamFilter.jar --help
```

## KONTUZ! 
Helbide bezala sartzen diren irteera fitxategi guztiak ez dira zertan sortuta egon behar, baina bai haien direktorio gurasoa.
Adbz. /home/\$USER/datuak/data.arff irteera fitxategia ez da sortuta egon behar, baina /home/$USER/datuak direktorioa existitu behar da fitxategia zuzen sortzeko.

## Komando erabilgarriak
### Aukera orokorrak

- `--run`: exekutatu beharreko komandoen zerrenda, komaz bereizita (adib. `sms2arff,vectorize`).
- `--help`: programaren laguntza erakusten du.
- Argumentuen formatua: `--gakoa balioa` eta `--gakoa=balioa` onartzen dira.

### Komando zehatzak

- `sms2arff`
  - Zer egiten du: SMSen TXT fitxategi bat ARFF bihurtzen du.
  - `--sms2arff.txt` [Beharrezkoa]: sarrerako TXTaren bidea.
  - `--sms2arff.arff` [Beharrezkoa]: irteerako ARFFaren bidea.
  - `--sms2arff.blind` [Aukerakoa]: `true|false`, TXTak klase-etiketarik ez duen adierazten du (`lehenetsita: false`).

- `analyze`
  - Zer egiten du: ARFF baten oinarrizko estatistikak erakusten ditu (instantziak, atributuak, klase-banaketak).
  - `--analyze.data` [Beharrezkoa]: aztertu beharreko ARFFaren bidea.
  - `--analyze.stage` [Aukerakoa]: kontsolan inprimatzeko etaparen izena (`lehenetsia: ETAPA`).

- `vectorize`
  - Zer egiten du: bektorizazio pipeline-a aplikatzen du eta bektorizatutako ARFF bat sortzen du.
  - `--vectorize.raw` [Beharrezkoa]: sarrerako ARFF gordinaren bidea (testua).
  - `--vectorize.bek` [Beharrezkoa]: irteerako ARFF bektorizatuaren bidea.
  - `--vectorize.filter` [Beharrezkoa]: gordetzeko/kargatzeko filtroaren modeloaren bidea (`.model`).
  - `--vectorize.train` [Aukerakoa]: `true|false`; `true` bada filtroa entrenatu eta gordetzen du, `false` bada lehendik dagoena berrerabiltzen du (`lehenetsia: false`).

- `param-search`
  - Zer egiten du: aurreprozesatze/oinarriko parametro bilaketako esperimentuak exekutatzen ditu.
  - `--param-search.rawTrain` [Beharrezkoa]: train ARFF gordinaren bidea.

- `param-search-v2`
  - Zer egiten du: parametro/atributu bilaketaren V2 bertsioa exekutatzen du.
  - `--param-search-v2.rawTrain` [Beharrezkoa]: train ARFF gordinaren bidea.

- `sweep`
  - Zer egiten du: sailkatzailearen parametro-ekorketa exekutatzen du, train/dev jada bektorizatuta daudenean.
  - `--sweep.trainBek` [Beharrezkoa]: train ARFF bektorizatuaren bidea.
  - `--sweep.devBek` [Beharrezkoa]: dev ARFF bektorizatuaren bidea.

- `train-optimal`
  - Zer egiten du: azken modelo bat entrenatu eta gordetzen du, hiperparametro zehatzak erabiliz.
  - `--train-optimal.hl` [Beharrezkoa]: MLPren hidden layers (adib. `10` edo `5,10`).
  - `--train-optimal.lr` [Beharrezkoa]: MLPren learning rate.
  - `--train-optimal.m` [Beharrezkoa]: MLPren momentum.
  - `--train-optimal.train` [Beharrezkoa]: train ARFF gordinaren bidea.
  - `--train-optimal.dev` [Aukerakoa]: dev ARFF gordinaren bidea.
  - `--train-optimal.rawData` [Beharrezkoa]: aldi baterako ARFF konbinatu gordinaren bidea.
  - `--train-optimal.bekData` [Beharrezkoa]: ARFF konbinatu bektorizatuaren bidea.
  - `--train-optimal.filter` [Beharrezkoa]: bektorizazio filtro/modeloaren bidea.
  - `--train-optimal.out` [Beharrezkoa]: entrenatutako modeloaren irteerako bidea (`.model`).

- `quality`
  - Zer egiten du: kalitate-ebaluazioak exekutatu eta metrikak gordetzen ditu.
  - `--quality.out` [Beharrezkoa]: metriken irteerako fitxategiaren bidea.
  - `--quality.mode` [Aukerakoa]: ebaluazio modua `holdout|unfair|srho` (`lehenetsia: holdout`).
  - `--quality.train` [Beharrezkoa]: train ARFF gordinaren bidea.
  - `--quality.dev` [Beharrezkoa]: dev ARFF gordinaren bidea.
  - `quality.mode=srho` bada, gainera:
    - `--quality.repeats` [Aukerakoa]: errepikapen kopurua (`lehenetsia: 10`).
    - `--quality.ratio` [Aukerakoa]: train proportzioa split bakoitzean (`lehenetsia: 0.8`).
    - `--quality.seed` [Aukerakoa]: hasierako hazia (`lehenetsia: 42`).
    - `--quality.tmp` [Aukerakoa]: tarteko fitxategietarako aldi baterako karpeta (`lehenetsia: src/data/tmp`).

- `predict`
  - Zer egiten du: testeko iragarpenak sortu eta TXT fitxategian gordetzen ditu.
  - `--predict.test` [Beharrezkoa]: testeko TXT fitxategiaren bidea (mezuak lerroz-lerro).
  - `--predict.model` [Beharrezkoa]: iragartzeko erabiliko den modeloaren bidea (`.model`).
  - `--predict.filter` [Beharrezkoa]: bektorizaziorako erabiliko den `multiFilter` modeloaren bidea (`.model`).
  - `--predict.out` [Beharrezkoa]: iragarpenen irteerako fitxategiaren bidea.

## Adibideak

### 1) Komando bakarra

```bash
java -jar SpamFilter.jar \
  --run sms2arff \
  --sms2arff.txt src/data/txt/SMS_SpamCollection.train.txt \
  --sms2arff.arff src/data/arff/SMS_SpamCollection.train.arff \
  --sms2arff.blind false
```

### 2) Hainbat komando exekuzio bakarrean

```bash
java -jar SpamFilter.jar \
  --run sms2arff,vectorize \
  --sms2arff.txt src/data/txt/SMS_SpamCollection.train.txt \
  --sms2arff.arff src/data/arff/SMS_SpamCollection.train.arff \
  --sms2arff.blind false \
  --vectorize.raw src/data/arff/SMS_SpamCollection.train.arff \
  --vectorize.bek src/data/arff/SMS_SpamCollection.bektrain.arff \
  --vectorize.filter src/data/model/multiFilter.model \
  --vectorize.train true
```

### 3) Hold-out ebaluazioa

```bash
java -jar SpamFilter.jar \
  --run quality \
  --quality.mode holdout \
  --quality.train src/data/arff/SMS_SpamCollection.train.arff \
  --quality.dev src/data/arff/SMS_SpamCollection.dev.arff \
  --quality.out src/data/results/metrikakHO.txt
```

### 4) Unfair ebaluazioa

```bash
java -jar SpamFilter.jar \
  --run quality \
  --quality.mode unfair \
  --quality.train src/data/arff/SMS_SpamCollection.train.arff \
  --quality.dev src/data/arff/SMS_SpamCollection.dev.arff \
  --quality.out src/data/results/metrikakEZ.txt
```

### 5) SRHO ebaluazioa

```bash
java -jar SpamFilter.jar \
  --run quality \
  --quality.mode srho \
  --quality.train src/data/arff/SMS_SpamCollection.train.arff \
  --quality.dev src/data/arff/SMS_SpamCollection.dev.arff \
  --quality.repeats 10 \
  --quality.ratio 0.8 \
  --quality.seed 42 \
  --quality.tmp src/data/tmp \
  --quality.out src/data/results/metrikakSRHO.txt
```

### 6) Iragarpenak (predict)

```bash
java -jar SpamFilter.jar \
  --run predict \
  --predict.test src/data/txt/SMS_SpamCollection.test_blind.txt \
  --predict.model src/data/model/final.model \
  --predict.filter src/data/model/multiFilter.model \
  --predict.out src/data/results/iragarpenak.txt
```

## Bideak

Bide erlatiboak komandoa exekutatzen duzun direktoriotik ebazten dira.

`SpamFilter`-en erro direktoriotik exekutatzen baduzu, `src/data/...` bezalako bideek bide absoluturik gabe funtzionatzen dute.

