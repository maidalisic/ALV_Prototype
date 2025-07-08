# ALV – AI-powered Logfile Validator

ALV ist ein FastAPI-Backend, das **Anomalien erkennt** und **Fehler klassifiziert**  
in CI/CD-Logs (Schwerpunkt Embedded-Software). Zwei Analyse-Engines stehen zur Wahl:

1. **Local Model** – Isolation Forest + TF-IDF-Bigrams, offline – trainierbar auf eigenen Logs  
2. **ChatGPT** – OpenAI API ( GPT-4o-mini ) für schnelle Ad-hoc-Analysen

---

## Highlights

| Feature | Details |
|---------|---------|
| **OpenAPI 3** | Automatische Swagger UI unter `/docs` |
| **Pluggable ML-Pipelines** | Isolation Forest ↔ ChatGPT, Regex-/ML-Classifier |
| **Preprocessing** | Timestamp-Stripping, Hex-Filter, Lower-casing |
| **Global Threshold** | µ − 2 σ-Grenze, in Modell-Datei persistiert |
| **Model Versioning** | `app/models/model_YYYYMMDDhhmmss.joblib` |
| **Helper-Scripts** | `train_from_dir.py`, `train_and_test.py`, `eval_pr.py` |

---

## Ordnerstruktur

```text
ALV/
├─ app/               FastAPI-Code
│  ├─ main.py         API-Routes
│  ├─ schemas.py      Pydantic-I/O-Modelle
│  └─ service/        Trainer · Analyser · Classifier
├─ scripts/           CLI-Hilfsprogramme
├─ logs/              Beispiel-Logs  (train_clean, test)
├─ tests/             Pytest-Suite  (+ Mini-Log-Fixtures)
├─ requirements.txt
└─ README.md
```

---

## Schnellstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-…   # nur falls ChatGPT-Mode genutzt wird
uvicorn app.main:app --reload
```

Swagger-UI: <http://127.0.0.1:8000/docs>

---

### Train / Analyse per cURL

```bash
# Modell trainieren (lokal, 5 % Outlier, 200 Trees)
curl -F "files=@logs/train_clean/build_ok.log" \
     -F "files=@logs/train_clean/deploy_ok.log" \
     "http://127.0.0.1:8000/train?contamination=0.05&n_estimators=200"

# Log mit lokalem Modell + Klassifikation prüfen
curl -F "file=@logs/test/segfault.log" \
     "http://127.0.0.1:8000/analyse?mode=local&classify=true" | jq
```

---

## Umgebungsvariablen

| Variable         | Beschreibung                                  |
|------------------|-----------------------------------------------|
| `OPENAI_API_KEY` | API-Key für ChatGPT-Analyse (optional)        |
| `ALV_MODELS_DIR` | alternatives Ablage-Verzeichnis für Modelle   |

---

## Tests

```bash
pytest -q
```

Die Suite trainiert ein Kurzeit-Modell, ruft beide Endpoints auf und prüft,
dass mindestens eine Anomalie erkannt wird.

---

## Lizenz

MIT © 2025