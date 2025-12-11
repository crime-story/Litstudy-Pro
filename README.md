# 📚 LitStudy Pro - Analiză Bibliometrică Avansată

**LitStudy Pro** este o aplicație web interactivă dezvoltată pentru a asista cercetătorii în procesul de analiză a literaturii științifice (Literature Review). Construită pe baza bibliotecii `litstudy`, aplicația extinde funcționalitățile acesteia oferind o interfață grafică prietenoasă, capabilități avansate de procesare a limbajului natural (NLP) și generare automată de rapoarte PDF.

Acest proiect a fost realizat ca parte a evaluării pentru cursul de Topici Speciale în Inginerie Software - Master Inginerie Software Anul 2, demonstrând reproducerea, îmbunătățirea și integrarea unui articol științific într-un produs software funcțional.

Articol Litstudy: https://www.sciencedirect.com/science/article/pii/S235271102200125X

## 👥 Membrii echipei
- Popescu Paullo-Robertto-Karlos 506
- Horceag Andrei 506
- Pasăre Roxana-Francisca​ 506

## Demo al aplicației
Puteți vizualiza live demo-ul aplicației aici: [Demo Litstudy Pro](https://youtu.be/fNvYmaUuIEA)

Sau dacă apasați pe imaginea de mai jos:

[![Video - Demo Litstudy Pro](https://github.com/user-attachments/assets/564ea341-bd07-469e-97dd-80ec9d668268)](https://youtu.be/fNvYmaUuIEA)

## 🚀 Funcționalități Cheie
### 1. Ingestie și Normalizare de Date
- **Căutare Live:** Integrare cu API-ul **DBLP** pentru căutarea în timp real a articolelor științifice.

- **Import Fișiere:** Suport pentru formatele standard academice: `.bib` (BibTeX), `.ris` și `.csv`.

- **Auto-Repair:** Modulul `normalize_documents` detectează și corectează automat metadatele lipsă sau formatate greșit (ex: sursa jurnalului).

### 2. Dashboard Interactiv
- **Filtrare Avansată:** Filtrare dinamică după ani, jurnale/conferințe, autori sau cuvinte cheie în titlu.

- **Vizualizare Statistici:** Histograme pentru evoluția publicațiilor, top autori și surse.

- **Word Cloud:** Generare vizuală a celor mai frecvenți termeni din titlurile articolelor.

### 3. NLP & Topic Modeling
- Implementare personalizată a algoritmului **NMF (Non-negative Matrix Factorization)** folosind `scikit-learn`.

- Extragerea automată a subiectelor (topics) din abstractele articolelor.

- Vizualizarea cuvintelor dominante pentru fiecare topic identificat.

### 4. Analiză de Rețea
- Generarea grafurilor de colaborare (co-author networks).

- Identificarea clusterelor de cercetători care lucrează împreună.

### 5. Raportare și Export
- **Export CSV:** Descărcarea datelor curățate și procesate.

- **Generator PDF Inteligent:** Crearea automată a unui raport profesional care include:

  - Rezumatul selecției.

  - Toate graficele generate în sesiune.

  - Analiza semantică a topicurilor.

## 🛠️ Arhitectura Tehnică
Proiectul este construit folosind ecosistemul Python Data Science:
- **Limbaj:** Python **3.12.10** (Versiune necesară). ❗

- **Frontend:** `Streamlit` (pentru interfață web rapidă și interactivă).

- **Backend Logic:** `litstudy` (procesare bibliometrică), `pandas` (manipulare date).

- **NLP:** `scikit-learn` (TF-IDF Vectorizer, NMF Model).

- **Persistență & Cache:** Sistemul de caching DBLP pentru interogări rapide și fișiere temporare CSV pentru procesarea upload-urilor.

- **Raportare:** `fpdf` (generare documente PDF programatic).

## 📥 Instalare și Configurare
### 1. Clonare Repository
```bash
git clone https://github.com/FranciscaPasare28/TSS.git
cd TSS
```

### 2. Instalare Dependențe
```bash
pip install streamlit litstudy matplotlib scikit-learn numpy seaborn pandas wordcloud fpdf networkx
```

### 3. Rulare Aplicație
```bash
streamlit run main.py
```

Aplicația se va deschide automat în browser la adresa `http://localhost:8501`.

## 📖 Ghid de Utilizare
### Pasul 1: Încărcarea Datelor (Sidebar)
În meniul din stânga, alege metoda de import:

- **Căutare Live (DBLP):** Introdu cuvinte cheie (ex: "Machine Learning") și numărul maxim de rezultate. Notă: _Rezultatele sunt salvate local în fișiere cache `.dblp` pentru viteză._

- **Fișier Local:** Poți folosi fișierul `papers.csv` inclus în proiect pentru un demo rapid, sau încărca propriile fișiere `.bib` / `.ris`.

### Pasul 2: Filtrarea
Folosește sliderele și meniurile dropdown din sidebar pentru a rafina setul de date. Graficele se vor actualiza în timp real.

- _Sfat:_ Folosește bara de progres pentru a vedea câte articole au rămas după filtrare.

### Pasul 3: Analiza (Tab-uri)
1. **Dashboard Statistici:** Analizează tendințele generale.

2. **Topic Modeling:** Alege numărul de topicuri (ex: 5) și apasă "Rulează Analiza" pentru a vedea ce subiecte latente există în abstracte.

3. **Rețele:** Vizualizează conexiunile dintre autori.

### Pasul 4: Export
Mergi în tab-ul "Export Date".

- Apasă **Generare Raport PDF** pentru a primi un document complet.

## 📂 Structura Proiectului
```plaintext
TSS/
├── lib/                   # Librării sau resurse adiționale ale proiectului
├── .dblp.bak/.dat/.dir    # Fișiere de cache generate automat de litstudy (pentru a stoca căutările DBLP)
├── citation.html          # (Output) Vizualizarea interactivă a rețelei de co-autori
├── debug.log              # (Output) Log-uri pentru debugging și monitorizare erori
├── main.py                # 🚀 CODUL SURSĂ PRINCIPAL (Aplicația Streamlit)
├── papers.csv             # Dataset de exemplu (poate fi folosit pentru demo)
├── README.md              # Documentația proiectului
└── temp_*.csv             # Fișiere temporare generate în timpul procesării upload-urilor
```
