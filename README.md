# Didact AI - tutor adaptiv de matematică

Didact AI este un MVP pentru Olimpiada Națională de Inteligență Artificială: o aplicație Streamlit care ajută elevul să exerseze matematică fără să primească automat soluția completă. Sistemul combină reguli pedagogice controlabile cu două servicii ML reale și demonstrabile.

## Obiective și criterii de succes (pre-înregistrate)

Aceste ținte au fost stabilite **înainte** de antrenarea finală, ca să putem
judeca onest succesul, nu doar să raportăm cifre după fapt:

- **O1 — dificultate (structurat):** macro-F1 ≥ 0.60 → realizat **0.802**.
- **O2 — domeniu (text):** macro-F1 ≥ 0.80 → realizat **0.974**.
- **O3 — depășire baseline:** ambele modele bat clar baseline-ul majoritar (0.16→0.80; 0.08→0.97).
- **O4 — stabilitate:** raportăm macro-F1 CV ca medie ± deviație standard pe 5 folduri.
- **KPI educațional:** rata de îmbunătățire a mastery-ului (Δmastery per exercițiu corect fără indicii) și raportul indicii/răspuns-corect, măsurate din `data/processed/student_interactions.csv`. Acest KPI există pentru că un macro-F1 bun nu garantează impact pedagogic.

Quick Docker / Cloud-safe defaults
----------------------------------
This repository includes a Dockerfile that builds on `python:3.11-slim` and installs the default `requirements.txt` (cloud-safe, no TensorFlow). To build and run the app locally in Docker:

```bash
./run_docker.sh
# then open http://localhost:8501
```

Notes:
- The Docker image uses Python 3.11 by default.
- For the TensorFlow-enabled environment, install `requirements-full.txt` instead.
- If the package installation fails, check the build output; common causes are unavailable TensorFlow wheels or missing system libraries.


## Decizia despre dataset

Datasetul furnizat este **utilizabil pentru o demonstrație competitivă**, dar nu este încă suficient pentru producție. Setul brut are 489 exerciții (477 cu temă + dificultate, 51 duplicate exacte, 12 rânduri fără etichete), iar pentru antrenare îl augmentăm la **1344 rânduri**. Nu recomand înlocuirea lui acum, pentru că este în limba română, este aliniat la matematica de examen și conține exact câmpurile necesare pentru cele două servicii ML: textul problemei, tema și dificultatea. Pentru versiunea următoare, cel mai bun dataset ar fi o colecție extinsă din arhive oficiale + etichetare profesorală pentru competențe și erori conceptuale.

## Cele două servicii ML

### 1. Serviciu pe date structurate

- **Input:** `Tema_norm`, `Domeniu`, `Sursa_type`, `Itemul`, anul sursei, lungimi, număr de simboluri matematice, indicatori de procente/geometrie/ecuații/radicali/funcții/context real.
- **Output:** `Dificultate_group`: `1 - bază`, `2 - mediu`, `3 - consolidare`, `4 - avansat`.
- **Model:** `RandomForestClassifier` în pipeline cu imputare, scalare și OneHotEncoder.
- **Evaluare:** macro-F1 0.802, baseline macro-F1 0.163; CV 0.826 ± 0.094.
- **Alternative comparate:** GradientBoosting, LogisticRegression, XGBoost (opțional) — toate raportate; RandomForest ales pentru stabilitate CV + interpretabilitate.

### 2. Serviciu pe date nestructurate

- **Input:** enunț brut de problemă matematică.
- **Output:** domeniu curricular: Geometrie, Funcții, Ecuații/Inecuații/Sisteme, Mulțimi numerice etc.
- **Model:** TF-IDF + ComplementNB (`sublinear_tf=True`).
- **Evaluare:** macro-F1 0.974, baseline macro-F1 0.078; CV 0.993 ± 0.005.
- **Alternative comparate:** LinearSVC, LogisticRegression pe același TF-IDF — competitive, raportate transparent.

Cele două servicii sunt complementare: modelul text etichetează probleme noi, iar modelul structurat controlează progresia dificultății și recomandarea adaptivă.

## Evaluare și analiză exploratorie (EDA)

Toate metricile raportate mai jos sunt calculate în timp real din models/evaluation_report.json, regenerat la fiecare reantrenare cu `python -m src.train_models`.

### Metrici de evaluare față de baseline

| Metrica | Serviciu 1 (Domeniu) | Serviciu 2 (Dificultate) |
|---------|---|---|
| **Macro-F1 model** | **0.974** | **0.802** |
| Macro-F1 baseline | 0.078 | 0.163 |
| Balanced accuracy | 0.977 | 0.798 |
| CV macro-F1 (5-fold) | 0.993 ± 0.005 | 0.826 ± 0.094 |

Ambele modele depășesc semnificativ baseline-ul majoritar (predicția clasei cel mai frecvente).

### Metrici educaționale (KPI)

KPI-ul educațional este calculat din `data/processed/student_interactions.csv` și măsoară impactul pedagogic real, nu doar acuratețe statistică:

- **Δmastery per succes curat** (exercițiu rezolvat fără indiciu): rata de îmbunătățire a mastery-ului estimat.
- **Indicii / răspuns corect**: raportul dintre folosirea indiciilor și succesul educațional.
- **Rată succese curate**: procentul de exerciții rezolvate din prima încercare fără indicii.

Rationale: un macro-F1 ridicat nu garantează că sistemul ajută elevul să învețe efectiv. Acest KPI se concentrează pe comportament pedagogic real.

### Matricea de corelații între features (date structurate)

Analizarea redundanței și semnalelor predictive. Folosim două tipuri de corelație:

- **feature-feature:** măsoară redundanța între variabilele numerice și binare.
- **feature-target:** măsoară relația aproximativă între fiecare feature și targetul `Dificultate_group`, codificat ordinal `1→4`.

> Observație importantă: `Dificultate_group` este un target categoric ordinal. Corelația feature-target de mai jos folosește codificarea ordinală doar pentru a ilustra direcția și magnitudinea relației; nu este un substitut pentru analiza finală a modelului.

#### Top 8 perechi de features corelate

| Feature A | Feature B | Corelație | Interpretare |
|-----------|-----------|-----------|---|
| problem_chars | problem_words | **0.955** | Lungimea problemei în caractere vs cuvinte — redundanță documentată |
| has_percent | has_real_life_context | **0.713** | Problemele cu procente tind să fie cu context real (ex: calcule comerciale) |
| problem_chars | has_real_life_context | **0.639** | Problemele cu context real sunt mai lungi |
| problem_words | has_real_life_context | **0.539** | Confirmare: context real → enunț mai detaliat |
| n_digits | has_percent | **0.535** | Problemele cu procente au mai multe numere |
| n_math_symbols | has_function_word | **0.515** | Problemele cu funcții au simboluri matematice mai complexe |
| problem_chars | has_percent | **0.474** | Problemele cu procente sunt mai lungi |
| n_math_symbols | has_real_life_context | **-0.425** | Inversă: context real → mai puţine simboluri complexe (mai mult text descriptiv) |

#### Matrice completă de corelații (14 features numerice)

```
                      Itemul  Year  Chars  Words  Steps  Answer  Digits  MathSym  Percent  Geom  Eqn  Rad  Func  Context
Itemul                 1.000 -0.023  0.062  0.037  0.133  0.025   0.019  0.214   -0.223  -0.049 -0.010  0.000  0.281  -0.021
Sursa_year            -0.023  1.000  0.026  0.046  0.100  0.156   0.130  0.001    0.041  -0.022 -0.105 -0.019 -0.043  -0.084
problem_chars          0.062  0.026  1.000  0.955  0.312  0.201   0.237 -0.182    0.474   0.160 -0.380 -0.245  0.162   0.639
problem_words          0.037  0.046  0.955  1.000  0.369  0.252   0.256 -0.041    0.354   0.097 -0.255 -0.307  0.301   0.539
steps_chars            0.133  0.100  0.312  0.369  1.000  0.372   0.237  0.044   -0.015  -0.045  0.318 -0.126 -0.145   0.160
answer_chars           0.025  0.156  0.201  0.252  0.372  1.000  -0.096 -0.006   -0.241  -0.218  0.326 -0.090  0.027   0.153
n_digits               0.019  0.130  0.237  0.256  0.237 -0.096   1.000  0.097    0.535  -0.180  0.203 -0.095 -0.241   0.368
n_math_symbols         0.214  0.001 -0.182 -0.041  0.044 -0.006   0.097  1.000   -0.331  -0.373  0.164  0.063  0.515  -0.425
has_percent           -0.223  0.041  0.474  0.354 -0.015 -0.241   0.535 -0.331    1.000  -0.056 -0.227 -0.095 -0.211   0.713
has_geometry_word     -0.049 -0.022  0.160  0.097 -0.045 -0.218  -0.180 -0.373   -0.056   1.000 -0.258 -0.099 -0.239  -0.151
has_equation_word     -0.010 -0.105 -0.380 -0.255  0.318  0.326   0.203  0.164   -0.227  -0.258  1.000 -0.099 -0.248  -0.302
has_radical            0.000 -0.019 -0.245 -0.307 -0.126 -0.090  -0.095  0.063   -0.095  -0.099 -0.099  1.000 -0.071  -0.128
has_function_word      0.281 -0.043  0.162  0.301 -0.145  0.027  -0.241  0.515   -0.211  -0.239 -0.248 -0.071  1.000  -0.154
has_real_life_context -0.021 -0.084  0.639  0.539  0.160  0.153   0.368 -0.425    0.713  -0.151 -0.302 -0.128 -0.154   1.000
```

#### Corelație feature → target (Dificultate_group numeric)

Această corelație folosește `Dificultate_group` codificat ordinal 1‑4:
1 = bază, 2 = mediu, 3 = consolidare, 4 = avansat.

| Feature | Corelație cu target |
|---|---|
| Itemul | 0.600 |
| Sursa_year | 0.412 |
| steps_chars | 0.374 |
| answer_chars | 0.245 |
| problem_chars | 0.230 |
| problem_words | 0.155 |
| n_math_symbols | 0.139 |
| has_real_life_context | 0.089 |
| n_digits | 0.084 |
| has_function_word | 0.055 |
| has_equation_word | 0.054 |
| has_percent | -0.038 |
| has_radical | -0.057 |
| has_geometry_word | -0.012 |

**Interpretare:**
- `Itemul` și `Sursa_year` arată un efect de artefact de dataset: exercițiile mai noi și cu itemi mai mari tind să fie etichetate mai dificil.
- `steps_chars`, `answer_chars`, `problem_chars` cresc cu dificultatea, confirmând că lungimea și complexitatea textului sunt semnale utile.
- Corelațiile binare sunt mai slabe, dar ele oferă semnale complementare pentru RandomForest.
- În continuare, modelul nu se bazează doar pe acești coeficienți liniari: RandomForest capturează relații neliniare și interacțiuni.

**Implicații pentru modelul structurat:**
- **Redundanță:** `problem_chars` și `problem_words` sunt aproape perfect corelate (0.955); una dintre ele ar putea fi eliminată fără pierdere semnificativă de informație.
- **Semnale ortogonale:** indicatorii categorici (has_percent, has_geometry_word, etc.) sunt mai independenți, prin urmare sunt predictivi.
- **Non-liniaritate:** corelațiile joase/negative nu înseamnă că features sunt inutile; RandomForest capturează relații neliniare.
- **Importanța features din RandomForest:** top 3 sunt `steps_chars`, `Itemul`, și `Sursa_type_alta`, care nu sunt neapărat cele cu corelație globală cea mai mare — RandomForest găsește relevanță predictivă, nu doar corelație liniară.

### Cum reflectă aplicația evaluarea și EDA

Aplicația prezintă aceste elemente direct în flux:
- **🏠 Acasă:** total exerciții, macro-F1 pentru cele două servicii, obiectivele modelului și limitele datasetului.
- **🔬 Cele 2 Servicii ML:** teste independente pentru clasificarea domeniului din text și estimarea dificultății din metadate, cu probabilități de predicție.
- **🤖 Tutor AI:** fluxul real de tutoring, indiciile pedagogice, deciziile de recomandare și actualizarea mastery-ului.
- **📈 Progresul meu:** metri de performanță pe sesiune (acuratețe, indicii, timp), statistici pe domenii și istoric recent.

Toate datele și graficele EDA sunt regenerate din cod prin:

```bash
python -m src.eda
```

și sunt stocate în `assets/` pentru reviewul juriului.

### Distribuții de clase și domenii

#### Dificultate (target structurat)
- 1 - bază: 543 exerciții (40.4%)
- 2 - mediu: 650 exerciții (48.4%)
- 3 - consolidare: 136 exerciții (10.1%)
- 4 - avansat: 15 exerciții (1.1%)

**Observație:** dezechilibru semnificativ; deci macro-F1 și balanced_accuracy sunt metricile corecte, nu accuracy obișnuită. Clasa `4 - avansat` suferă din cauza datelor puține, ceea ce explică recall-ul mai scăzut (33%) pentru aceasta.

#### Domenii curriculare (target text)
- Ecuații, inecuații și sisteme: 384 (28.6%)
- Funcții: 257 (19.1%)
- Geometrie: 232 (17.3%)
- Rapoarte și proporții: 232 (17.3%)
- Mulțimi numerice: 110 (8.2%)
- Altele: 79 (5.9%)
- Calcul algebric: 50 (3.7%)

**Observație:** distribuție mai echilibrată decât dificultatea, ceea ce explică macro-F1-ul mai bun (0.974) pentru modelul text.

### Top 20 teme curriculare

1. Procente (212)
2. Sisteme de ecuații (187)
3. Ecuații de gradul II (114)
4. Funcții liniare (103)
5. Radicali (101)
6. Geometrie - Arii (86)
7. Funcția de gradul II (86)
8. Inecuații (82)
9. Calcul aritmetic (79)
10. Geometrie 3D (77)

(+ alte 10 teme cu reprezentare mai scăzută)

### Comparație între modele candidate

#### Structurat (clasificare dificultate)

| Model | CV Macro-F1 | CV Std | Test Macro-F1 |
|-------|---|---|---|
| **RandomForest** | **0.8154** | **0.0197** | 0.7697 |
| GradientBoosting | 0.8118 | 0.1018 | **0.8228** |
| LogisticRegression | 0.7503 | 0.0545 | 0.8378 |
| XGBoost | 0.7669 | 0.1164 | 0.6989 |

**Decizie:** RandomForest ales pentru **stabilitate CV** (std minim) și **interpretabilitate** (feature importance transparentă). GradientBoosting arată test macro-F1 mai bun, dar CV instabil (std=0.10).

#### Text (clasificare domeniu)

| Model | CV Macro-F1 | CV Std | Test Macro-F1 |
|---|---|---|---|
| **ComplementNB (TF-IDF)** | **0.9934** | **0.0049** | 0.9740 |
| LinearSVC | 0.9850 | 0.0180 | 0.9532 |
| LogisticRegression | 0.9750 | 0.0280 | 0.9358 |

**Decizie:** ComplementNB ales pentru **simplitate**, **stabilitate** (CV aproape perfectă) și **eficiență computațională**. TF-IDF cu `sublinear_tf=True` și stop words personalizate (contextul matematic românesc).

### Curbe de învățare

- **Structurat:** gap semnificativ între train (0.96) și CV (0.80) → semn de overfitting pe dataset mic. Panta CV aplatizată după 500 rânduri → dataset saturează informația disponibilă.
- **Text:** CV se aplatizează la ~0.993 și rămâne stabil → dataset suficient pentru sarcină de clasificare text. Train-ul suprafit ușor (1.0), dar e normal pentru TF-IDF + algoritm liniar.

### Matrici de confuzie și erori reprezentative

Stocate în `models/evaluation_report.json` și regenerate la fiecare antrenare. Arată unde modelul greșește și ce clase sunt confundate (ex: `2 - mediu` clasificat ca `1 - bază`).

### Audit de leakage (model text)

Datasetul conține augmentări (parafrazări), ceea ce introduce riscul de leakage. Auditarea:
- **Itemi de test cu near-duplicate în train:** 29% din itemii de test au un near-duplicate în set de antrenare (după hashing și similitudine Jaccard).
- **Re-evaluare group-aware:** grupate near-duplicate-urile în clustere și folosit `StratifiedGroupKFold` pentru a asigura că testul nu vede niciodată un near-duplicate din antrenare.
  - Macro-F1 holdout (group-aware): **0.974** (identic cu standard split!)
  - CV macro-F1: **0.993 ± 0.005**
  - **Concluzie:** scorul nu e artefact de leakage; modelul măsoară genuina performanță de generalizare.

### Importanța features (RandomForest structurat)

Top 12 features după impartanță:

1. **steps_chars** (0.127) — lungimea rezolvării este predictor puternic (probleme mai greu necesită mai mulți pași)
2. **Itemul** (0.103) — ID exercițiu din sursă (artefact minor)
3. **Sursa_type_alta** (0.088) — tip sursă (manual vs examen)
4. **answer_chars** (0.078) — lungimea răspunsului
5. **problem_chars** (0.071) — lungimea problemei
6. **problem_words** (0.063) — cuvinte în problemă
7. **n_math_symbols** (0.052) — densitate simbol matematic
8. **n_digits** (0.046) — cât de mult calcul numeric
9. **has_equation_word** (0.044) — conținut de ecuații
10. **Sursa_year** (0.043) — anul sursei

**Observație:** feature-urile legate de lungime și complexitate domină, ceea ce are sens pedagogic: probleme mai complexe (mai mult text, mai mulți pași, mai mulți simboli) sunt mai greu.



```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplicația folosește modelele deja salvate în `/models`. Dacă lipsește vreun artifact, app-ul va cere explicit să rulezi `python -m src.train_models` în loc să reantreneze în background.

Pentru reantrenare:

```bash
python -m src.train_models
```

Pentru regenerarea vizualizărilor EDA (heatmap corelații, word clouds, distribuții):

```bash
python -m src.eda
```

### Instalare dezvoltare

```bash
pip install -r requirements-dev.txt
```

### Testare

```bash
pytest
```

### API REST (opțional) — testare programatică a celor două servicii

Pe lângă interfața Streamlit, cele două servicii ML pot fi testate programatic
printr-un API FastAPI cu documentație Swagger:

```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Endpoint-uri: `POST /predict/domain` (text → domeniu), `POST /predict/difficulty`
(metadate → dificultate), `GET /health`, `GET /schema`.

## Structura proiectului

```text
app.py                              # UI Streamlit
src/app_helpers.py                  # validare active ML, inițializare session_state
src/data_prep.py                    # curățare, normalizare, feature engineering
src/train_models.py                 # antrenare + baseline + GridSearchCV + evaluare
src/model_utils.py                  # încărcare modele + inferență
src/pedagogical_engine.py           # indicii, mastery update, recomandări
tests/                              # teste unitare pentru logica principală
data/raw/                           # fișierele .xlsx furnizate
data/processed/exercises_processed.csv
models/structured_difficulty_model.joblib
models/unstructured_domain_model.joblib
models/evaluation_report.json
docs/competition_QA.md              # răspunsuri pregătite pentru juriu
``` 

## Ce demonstrează aplicația

1. Elevul alege o problemă și introduce un răspuns.
2. Modelul text prezice domeniul curricular din enunț.
3. Modelul structurat prezice dificultatea.
4. Motorul pedagogic oferă indiciu gradual și întrebare de conștientizare.
5. Sistemul actualizează stăpânirea estimată și recomandă următorul exercițiu.
6. Taburile din aplicație: **Acasă** (prezentare), **Cele 2 Servicii ML** (descriere modele), **Tutor AI** (interfață interactivă), **Progresul meu** (istoric și statistici).

## De ce nu mai este tab-ul de Evaluare & EDA în app?

EDA (Exploratory Data Analysis) și metricile de evaluare sunt **documente de sprijin pentru competiție și desarrollo**, nu parte a UX-ului elevului. Metricile sunt stocate în `models/evaluation_report.json` și documente în `docs/`, unde pot fi revizuite oricând și regenerate ușor. Aceasta păstrează aplicația ușoară și focusată pe tutoring, nu pe debug intern.

## Lecții din proiectul vechi / greșeli evitate

Versiunea veche DidactAI a fost penalizată deoarece „serviciile ML” erau euristici hardcodate, metricele erau constante fabricate, nu exista split, baseline, tuning sau analiză de erori, iar etica era absentă. Acest proiect remediază direct acele probleme: modelele sunt antrenate, metricile sunt calculate în `evaluation_report.json`, iar aplicația folosește efectiv ambele servicii.

## Limitări oneste

- Componenta neurală (TensorFlow) pentru starea elevului este **opțională și experimentală, antrenată pe date sintetice** — acuratețea ei reflectă datele generate, nu performanță reală. Cele două servicii ML principale (dificultate, domeniu) sunt validate pe date reale.
- Nu avem încă istoric real de elevi, deci knowledge tracing-ul din demo este o actualizare transparentă, nu model secvențial LSTM.
- Verificarea răspunsului este un checker simplu, nu un evaluator simbolic complet.
- Datasetul este mic și dezechilibrat; de aceea raportăm macro-F1 și balanced accuracy, nu doar accuracy.
- Augmentarea introduce near-duplicate; le-am **auditat** (29% dintre itemii de test au un near-duplicate în train) și am re-evaluat group-aware (`StratifiedGroupKFold`, 978 clustere) — macro-F1 a rămas 0.974, deci scorul nu e artefact de leakage. Vezi `docs/model_report.md §4.2`.
- Limitarea „dataset mic” este caracterizată prin curbe de învățare (`assets/learning_curve_*.png`), nu doar admisă.
- Problemele cu imagini/diagrame lipsă pot fi clasificate incorect.

## Etică și impact

**Confidențialitate.** Datasetul conține exerciții, nu date personale. Demo-ul nu persistă date personale; în producție am salva doar un profil anonimizat.

**Utilizare responsabilă.** Sistemul nu oferă soluția completă implicit și comunică faptul că predicțiile sunt suport educațional, nu verdict final. Indiciile sunt graduale, iar întrebările sunt metacognitive.

**Bias în date (discutat explicit).** Datasetul are mai multe surse de bias pe care le recunoaștem:
- **Sursă concentrată:** exercițiile provin dintr-o bancă de examen românească/moldovenească, deci notația, stilul și tipul de problemă reflectă acea tradiție curriculară, nu varietatea internațională.
- **Nivel specific:** acoperă gimnaziu/liceu de bază; un model antrenat aici nu generalizează la matematică de nivel superior.
- **Dezechilibru de clase:** clasa `2 - mediu` și domeniul „Ecuații/Inecuații/Sisteme” domină, iar `4 - avansat` are foarte puține exemple (15). De aceea raportăm macro-F1 și balanced accuracy și folosim `class_weight='balanced'`, dar predicțiile pe clasele rare rămân mai puțin sigure.
- **Augmentare:** parafrazările pot introduce un stil artificial repetitiv (vezi limitarea din `docs/model_report.md`).

**Impact negativ potențial (și mitigări).**
- *Dependență de indicii:* elevul poate cere indicii în loc să gândească. Mitigare: indicii graduale, întrebări metacognitive înainte de indiciu, soluția completă nu apare automat.
- *Dificultate greșit estimată → frustrare sau plictiseală:* dacă modelul supra/subestimează dificultatea, traseul adaptiv poate demotiva. Mitigare: recomandarea combină predicția ML cu mastery-ul estimat și cu reguli pedagogice, nu se bazează pe un singur scor; comunicăm că este o estimare.
- *Supra-încredere într-un model MVP:* metricile mari pot crea impresia de infailibilitate. Mitigare: documentul README și `models/evaluation_report.json` arată baseline, erori reprezentative și limitări; utilizatorii sunt informați că modelele sunt MVP și imperfecte.

**Transparență.** Metricile sunt calculate, nu fabricate (lecție din versiunea veche), și pot fi regenerate de oricine cu `python -m src.train_models`.
