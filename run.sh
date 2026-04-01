#!/usr/bin/env bash
set -euo pipefail

# ======================================================
# ERABILTZAILEAK ALDATZEKO ATALA
# ======================================================
# Script hau dagoen karpeta (normalean ez ukitu behar)
PROIEKTU_KARPETA="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Java komandoa (adib.: java edo /ruta/osora/java)
JAVA_KOMANDOA="java"

# JAR fitxategiaren bidea
JAR_BIDEA="$PROIEKTU_KARPETA/SpamFilter.jar"

# Datu karpetak (behar baduzu, hemen aldatu behin bakarrik)
DATU_KARPETA="$PROIEKTU_KARPETA/src/data"
TXT_KARPETA="$DATU_KARPETA/txt"
ARFF_KARPETA="$DATU_KARPETA/arff"
MODEL_KARPETA="$DATU_KARPETA/model"
EMAITZA_KARPETA="$DATU_KARPETA/results"

# Wekarekin bateragarritasunerako Java aukera gehigarria
JAVA_AUKERAK=(--add-opens java.base/java.lang=ALL-UNNAMED)

if [[ ! -f "$JAR_BIDEA" ]]; then
  echo "[ERROREA] Ez da fitxategia aurkitu: $JAR_BIDEA"
  echo "Jar-a bide horretan jarri edo aldatu JAR_BIDEA balioa scriptaren hasieran."
  exit 1
fi

# EGIKARIARAZTEKO ADIBIDEAK: Komentatu/deskomentatu egikaritu nahi dituzun funtzionalitateak.
# =========================
# 1) sms2arff
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run sms2arff \
#   --sms2arff.txt "$TXT_KARPETA/SMS_SpamCollection.train.txt" \
#   --sms2arff.arff "$ARFF_KARPETA/SMS_SpamCollection.train.arff" \
#   --sms2arff.blind false

# =========================
# 2) analyze
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run analyze \
#   --analyze.data "$ARFF_KARPETA/SMS_SpamCollection.train.arff" \
#   --analyze.stage ETAPA

# =========================
# 3) vectorize
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run vectorize \
#   --vectorize.raw "$ARFF_KARPETA/SMS_SpamCollection.train.arff" \
#   --vectorize.bek "$ARFF_KARPETA/SMS_SpamCollection.bektrain.arff" \
#   --vectorize.filter "$MODEL_KARPETA/TrainMultiFilter.model" \
#   --vectorize.train true

# =========================
# 4) sweep
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run sweep \
#   --sweep.trainBek "$ARFF_KARPETA/SMS_SpamCollection.bektrain.arff" \
#   --sweep.devBek "$ARFF_KARPETA/SMS_SpamCollection.bekdev.arff"

# =========================
# 5) param-search
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run param-search \
#   --param-search.rawTrain "$ARFF_KARPETA/SMS_SpamCollection.train.arff"

# =========================
# 6) param-search-v2
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run param-search-v2 \
#   --param-search-v2.rawTrain "$ARFF_KARPETA/SMS_SpamCollection.train.arff"

# =========================
# 7) train-optimal
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run train-optimal \
#   --train-optimal.hl 10,5 \
#   --train-optimal.lr 0.01 \
#   --train-optimal.m 0.1 \
#   --train-optimal.train "$ARFF_KARPETA/SMS_SpamCollection.train.arff" \
#   --train-optimal.dev "$ARFF_KARPETA/SMS_SpamCollection.dev.arff" \
#   --train-optimal.rawData "$ARFF_KARPETA/SMS_SpamCollection.data.arff" \
#   --train-optimal.bekData "$ARFF_KARPETA/SMS_SpamCollection.bekdata.arff" \
#   --train-optimal.filter "$MODEL_KARPETA/TrainDevMultiFilter.model" \
#   --train-optimal.out "$MODEL_KARPETA/MLP_Bezero(Train+Dev).model"

# =========================
# 8) quality (holdout)
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run quality \
#   --quality.mode holdout \
#   --quality.model "$MODEL_KARPETA/MLP_Train.model" \
#   --quality.trainBek "$ARFF_KARPETA/SMS_SpamCollection.bektrain.arff" \
#   --quality.devBek "$ARFF_KARPETA/SMS_SpamCollection.bekdev.arff" \
#   --quality.out "$EMAITZA_KARPETA/metrikakHO.txt"

# =========================
# 8b) quality (unfair)
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run quality \
#   --quality.mode unfair \
#   --quality.model "$MODEL_KARPETA/MLP_Train.model" \
#   --quality.trainBek "$ARFF_KARPETA/SMS_SpamCollection.bektrain.arff" \
#   --quality.devBek "$ARFF_KARPETA/SMS_SpamCollection.bekdev.arff" \
#   --quality.out "$EMAITZA_KARPETA/metrikakEZ.txt"

# =========================
# 8c) quality (srho)
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run quality \
#   --quality.mode srho \
#   --quality.model "$MODEL_KARPETA/MLP_Train.model" \
#   --quality.train "$ARFF_KARPETA/SMS_SpamCollection.train.arff" \
#   --quality.dev "$ARFF_KARPETA/SMS_SpamCollection.dev.arff" \
#   --quality.repeats 10 \
#   --quality.ratio 0.8 \
#   --quality.seed 42 \
#   --quality.tmp "$DATU_KARPETA/tmp" \
#   --quality.out "$EMAITZA_KARPETA/metrikakSRHO.txt"

# =========================
# 9) predict
# =========================
# "$JAVA_KOMANDOA" "${JAVA_AUKERAK[@]}" -jar "$JAR_BIDEA" --run predict \
#  --predict.test "$TXT_KARPETA/SMS_SpamCollection.test_blind.txt" \
#  --predict.model "$MODEL_KARPETA/MLP_Bezero(Train+Dev).model" \
#  --predict.filter "$MODEL_KARPETA/TrainDevMultiFilter.model" \
#  --predict.out "$EMAITZA_KARPETA/iragarpenak.txt"
