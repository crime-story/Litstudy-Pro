import streamlit as st
import litstudy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import seaborn as sns
import pandas as pd
import streamlit.components.v1 as components
from wordcloud import WordCloud
from fpdf import FPDF
import tempfile
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),   # log in fisier
        logging.StreamHandler()             # log in consola Streamlit
    ]
)

logger = logging.getLogger(__name__)

# logger.debug("mesaj")
# logger.error("eroare")


# --- CONFIGURARE PAGINA ---
st.set_page_config(
    page_title="LitStudy Pro - Analiză Bibliometrică",
    layout="wide",
    page_icon="📚"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    h1 { color: #2c3e50; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("📚 LitStudy Pro: Analiză Bibliometrică Avansată")
st.markdown("### Instrument pentru analiza automată a literaturii științifice")

# --- SIDEBAR: DATA LOADING ---
st.sidebar.header("1. Sursă Date")
sursa_date = st.sidebar.radio("Metoda de import:", ("Căutare Live (DBLP)", "Fișier Local"))

# Initializam session_state pentru a nu pierde datele la refresh (filtrare)
if 'docs' not in st.session_state:
    st.session_state['docs'] = []

if 'figures' not in st.session_state:
    st.session_state['figures'] = {}
if 'nmf_topics' not in st.session_state:
    st.session_state['nmf_topics'] = None

# --- FUNCTIE DE NORMALIZARE (FIX PENTRU TOATE FORMATELE) ---
def normalize_documents(new_docs):
    """
    Aceasta functie repara datele lipsa din obiectele litstudy,
    indiferent daca vin din CSV, BIB sau RIS.
    """
    count_fixed = 0
    for new_doc in new_docs:
        # FIX SOURCE (Jurnal/Conferinta)
        # Daca 'source' lipseste, incercam sa il gasim in alte campuri standard BibTeX/RIS
        if not hasattr(new_doc, 'source') or not new_doc.source or str(new_doc.source).lower() == 'nan':
            new_source = None
            # Ordinea de prioritate pentru a gasi sursa:
            if hasattr(new_doc, 'journal') and new_doc.journal:
                new_source = new_doc.journal
            elif hasattr(new_doc, 'booktitle') and new_doc.booktitle:
                new_source = new_doc.booktitle
            elif hasattr(new_doc, 'publisher') and new_doc.publisher:
                new_source = new_doc.publisher
            
            # Aplicam sursa gasita
            if new_source:
                new_doc.source = str(new_source)
                count_fixed += 1
            else:
                new_doc.source = "Unknown" # Ca sa nu crape graficul
    return new_docs, count_fixed

def clean_text(text):
    """Curata textul de diacritice pentru PDF"""
    if not isinstance(text, str): return str(text)
    replacements = {'ă':'a', 'â':'a', 'î':'i', 'ș':'s', 'ț':'t', 'Ă':'A', 'Â':'A', 'Î':'I', 'Ș':'S', 'Ț':'T', '„':'"', '”':'"', '–':'-'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

# --- CLASA GENERARE PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'LitStudy Pro - Raport', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, clean_text(title), 0, 1, 'L', 1)
        self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, clean_text(body))
        self.ln()
# ==============================

# LOGICA DE INCARCARE
new_docs = []

if 'regenerate_word_cloud' not in st.session_state:
    st.session_state['regenerate_word_cloud'] = False
regenerate_word_cloud = st.session_state['regenerate_word_cloud'] 

if sursa_date == "Căutare Live (DBLP)":
    query = st.sidebar.text_input("Cuvinte cheie", value="Machine Learning")
    limit_docs = st.sidebar.slider("Nr. maxim articole", 100, 500, 100, step=100)
    
    if st.sidebar.button("🔍 Caută pe DBLP"):
        with st.spinner('Se descarcă datele de pe DBLP...'):
            try:
                new_docs = litstudy.search_dblp(query, limit=limit_docs)
                new_docs, _ = normalize_documents(new_docs)
                regenerate_word_cloud = True
                st.session_state['docs'] = new_docs
                st.success(f"Găsite: {len(new_docs)} articole.")
            except Exception as e:
                st.error(f"Eroare: {e}")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Încarcă fișier (BibTeX, RIS, CSV)", 
        type=["bib", "ris", "csv"])
    if uploaded_file:
        temp_name = f"temp_{uploaded_file.name}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner('Se procesează fișierul...'):
                if temp_name.endswith(".csv"):
                    # CITIM CU PANDAS PENTRU A REPARA DATELE
                    df_temp = pd.read_csv(temp_name)
                    
                    # Normalizam numele coloanelor la litere mici (Source -> source)
                    df_temp.columns = df_temp.columns.str.lower()

                    # Redenumim 'link' -> 'doi' pentru Litstudy
                    if 'link' in df_temp.columns and 'doi' not in df_temp.columns:
                        df_temp.rename(columns={'link': 'doi'}, inplace=True)
                    
                    # Salvam CSV-ul temporar corectat
                    df_temp.to_csv(temp_name, index=False)
                    
                    # Incarcam documentele de baza
                    docs = litstudy.load_csv(temp_name)
                    
                    # --- FIX CRITIC PENTRU SURSE ---
                    # Litstudy ignora coloana 'source' daca nu e standard. O injectam manual.
                    if 'source' in df_temp.columns:
                        for i, doc in enumerate(docs):
                            if i < len(df_temp):
                                val = df_temp.iloc[i]['source']
                                # Daca avem o valoare valida in CSV, o punem in obiectul doc
                                if not pd.isna(val) and str(val).lower() != 'nan':
                                    doc.source = str(val)
                                    # Putem completa si jurnalul pentru siguranta
                                    doc.journal = str(val)
                elif temp_name.endswith(".bib"):
                    docs = litstudy.load_bibtex(temp_name)
                elif temp_name.endswith(".ris"):
                    docs = litstudy.load_ris(temp_name)

                # --- APLICAM NORMALIZAREA PENTRU TOATE ---
                docs, fixed_count = normalize_documents(docs)
                st.session_state['docs'] = docs
                st.sidebar.success(f"Fișier procesat: {len(docs)} articole")

                if fixed_count > 0:
                    st.sidebar.info(f"🛠️ S-au normalizat sursele pentru {fixed_count} articole.")
        except Exception as e:
            st.sidebar.error(f"Eroare fișier: {e}")

# Preluam documentele din memorie
docs = st.session_state['docs']

# --- SISTEM DE FILTRARE ---
filtered_docs = docs
if docs:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Filtrare Rezultate")
    
    # A. Filtru Ani
    years = [d.publication_year for d in docs if d.publication_year is not None]
    if years:
        min_y, max_y = int(min(years)), int(max(years))
        def slider_change_callback():
            st.session_state['regenerate_word_cloud'] = True
        sel_years = st.sidebar.slider("📅 Interval Ani", min_y, max_y, (min_y, max_y), key='my_slider', on_change=slider_change_callback)
        filtered_docs = [d for d in filtered_docs if d.publication_year and sel_years[0] <= d.publication_year <= sel_years[1]]

    # B. Filtru Sursa (Jurnal)
    sources = list(set([d.source for d in docs if hasattr(d, 'source') and d.source]))
    if sources:
        sel_source = st.sidebar.multiselect("📖 Filtru Jurnal/Conferință", sources)
        if sel_source:
            filtered_docs = [d for d in filtered_docs if hasattr(d, 'source') and d.source in sel_source]

    # C. Filtru Autori
    # Colectam toti autorii unici din documentele ramase
    all_authors = set()
    for d in docs:
        if d.authors:
            for a in d.authors:
                all_authors.add(a.name)
    
    if all_authors:
        # Sortam alfabetic pentru a fi usor de gasit
        sel_authors = st.sidebar.multiselect("👤 Filtru Autori", sorted(list(all_authors)))
        if sel_authors:
            # Pastram documentul daca MACAR UNUL din autorii selectati a contribuit la el
            filtered_docs = [
                d for d in filtered_docs 
                if d.authors and any(a.name in sel_authors for a in d.authors)
            ]

    # D. Cautare Text in Titlu
    text_search = st.sidebar.text_input("🔍 Caută în Titlu (ex: learn)")
    if text_search:
        # Filtrare case-insensitive (nu conteaza majusculele)
        filtered_docs = [
            d for d in filtered_docs 
            if d.title and text_search.lower() in d.title.lower()
        ]

    st.sidebar.markdown("---")
    # Afisare contor cu bara de progres vizuala
    procent = len(filtered_docs) / len(docs) if len(docs) > 0 else 0
    st.sidebar.progress(procent)
    st.sidebar.info(f"Se analizează: **{len(filtered_docs)}** / {len(docs)} articole")

docs = st.session_state.get('docs', [])

def author_name(a):
    # Daca autorul e un obiect cu atribut 'name'
    try:
        return a.name
    except Exception:
        # Daca e un dict / str / alt tip
        if isinstance(a, dict) and 'name' in a:
            return a['name']
        return str(a)

# --- INTERFATA PRINCIPALA ---
if filtered_docs:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Statistici", 
        "🧠 Topic Modeling (NLP)", 
        "🕸️ Rețele", 
        "📥 Export Date"
    ])

    # === TAB 1: STATISTICI EXTINSE ===
    with tab1:
        st.subheader("Privire de ansamblu")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Publicații pe ani**")
            fig1 = plt.figure(figsize=(8, 4))
            litstudy.plot_year_histogram(filtered_docs)
            st.pyplot(fig1, use_container_width=True)
            st.session_state['figures']['years'] = fig1
            
        with col2:
            st.markdown("**Top Autori**")
            fig2 = plt.figure(figsize=(8, 4))
            litstudy.plot_author_histogram(filtered_docs, limit=10)
            st.pyplot(fig2, use_container_width=True)
            st.session_state['figures']['authors'] = fig2

        st.markdown("---")
        
        # Statistici Surse
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Top Surse de Publicare**") 
            
            sources_list = []
            for d in filtered_docs:
                # Verificam daca avem atributul source si daca nu e "Unknown"
                if hasattr(d, 'source') and d.source and str(d.source) != "nan":
                    sources_list.append(d.source)
                elif hasattr(d, 'publisher') and d.publisher:
                    sources_list.append(d.publisher)
            
            if len(sources_list) > 0:
                s_counts = pd.Series(sources_list).value_counts().head(10)
                
                fig3, ax = plt.subplots(figsize=(8, 4))
                s_counts.plot(kind='bar', ax=ax, color='#4682B4') 
                
                ax.set_ylabel("Nr. Articole")
                ax.set_xlabel("") # Scoatem eticheta de jos ca sa fie mai curat
                
                # Rotim etichetele de jos pentru a se citi usor
                plt.xticks(rotation=45, ha='right')
                
                # Ajustam marginile ca sa nu taie textul
                plt.tight_layout()
                
                st.pyplot(fig3, use_container_width=True)
                st.session_state['figures']['sources'] = fig3
            else:
                st.warning("Nu au fost găsite informații despre Jurnal/Conferință în date.")

        with col4:
            st.markdown("**Word Cloud (Din Titluri)**")
            # Generare WordCloud
            text = " ".join([d.title for d in filtered_docs if d.title])
            if text:
                if 'wc' not in st.session_state:
                    st.session_state['wc'] = None
                wc = st.session_state['wc']
                if regenerate_word_cloud or wc is None:
                    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                    st.session_state['wc'] = wc
                    st.session_state['regenerate_word_cloud'] = False
                fig_wc = plt.figure(figsize=(8, 4))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(fig_wc, use_container_width=True)
                st.session_state['figures']['wordcloud'] = fig_wc

    # === TAB 2: TOPIC MODELING (Implementare "Low-Level" Scikit-Learn) ===
    with tab2:
        st.subheader("Detectare Automată a Subiectelor (NMF)")
        st.markdown("""
        > Această secțiune implementează algoritmul **NMF (Non-negative Matrix Factorization)** folosind direct 
        > biblioteca *Scikit-Learn* pentru o precizie maximă și control total asupra datelor.
        """)

        if len(filtered_docs) < 10:
            st.warning("⚠️ Ai nevoie de cel puțin 10 articole pentru a genera topicuri relevante.")
        else:
            col_settings, col_viz = st.columns([1, 3])
            
            with col_settings:
                st.markdown("**Setări Model**")
                num_topics = st.slider("Număr de topicuri", 3, 10, 5)
                run_nlp = st.button("🚀 Rulează Analiza")

            with col_viz:
                if run_nlp:
                    with st.spinner("Se procesează textul și se antrenează modelul NMF..."):
                        try:
                            # 1. PREGATIRE DATE (Extragem textul din obiectele LitStudy)
                            # Combinam titlul cu abstractul (daca exista) pentru fiecare articol
                            text_data = []
                            for doc in filtered_docs:
                                content = doc.title
                                if hasattr(doc, 'abstract') and doc.abstract:
                                    content += " " + doc.abstract
                                text_data.append(content)

                            # 2. VECTORIZARE (TF-IDF)
                            # Transformam textul in numere, eliminand cuvintele comune (stop words)
                            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                            tfidf = tfidf_vectorizer.fit_transform(text_data)
                            feature_names = tfidf_vectorizer.get_feature_names_out()

                            # 3. ANTRENARE MODEL NMF
                            nmf_model = NMF(n_components=num_topics, random_state=42, init='nndsvd')
                            nmf_model.fit(tfidf)
                            topics_data = []
                            for topic_idx, topic in enumerate(nmf_model.components_):
                                top_indices = topic.argsort()[:-11:-1]
                                top_w = [feature_names[i] for i in top_indices]
                                topics_data.append(f"Topic {topic_idx + 1}: {', '.join(top_w)}")
                            st.session_state['nmf_topics'] = topics_data
                            
                            st.success("Analiză finalizată cu succes (Scikit-Learn Backend)!")
                            st.markdown("### 🧩 Rezultate Identificate:")

                            # 4. EXTRAGERE SI AFISARE TOPICURI
                            for topic_idx, topic in enumerate(nmf_model.components_):
                                # Luam top 10 cuvinte cu cea mai mare greutate in topic
                                top_indices = topic.argsort()[:-11:-1]
                                top_words = [feature_names[i] for i in top_indices]
                                
                                with st.expander(f"Topic {topic_idx + 1}: {top_words[0].upper()}..."):
                                    st.write(f"**Cuvinte cheie:** {', '.join(top_words)}")
                                    # Afisam un mini grafic de bare pentru importanta cuvintelor (Bonus vizual)
                                    topic_df = pd.DataFrame({
                                        'Cuvânt': top_words,
                                        'Importanță': topic[top_indices]
                                    })
                                    st.bar_chart(topic_df.set_index('Cuvânt'))

                        except Exception as e:
                            st.error(f"A apărut o eroare la procesare: {e}")
                            st.info("Verifică dacă articolele selectate au abstracte disponibile.")

    # === TAB 3: RETELE ===
    with tab3:
        st.subheader("Rețea de Co-autori")
        st.info("Această vizualizare arată grupurile de cercetători care colaborează frecvent.")
        
        try:
            net = litstudy.build_coauthor_network(filtered_docs)
            if net and len(net.nodes) > 0:
                html_file = "network.html"
                litstudy.plot_network(net, height="600px")
                
                # 'Hack' pentru a citi fisierul generat de litstudy
                # De obicei il salveaza ca 'citation.html' sau deschide temp
                if os.path.exists("citation.html"):
                    with open("citation.html", 'r', encoding='utf-8') as f:
                        html_src = f.read()
                    components.html(html_src, height=620, scrolling=True)
                else:
                    st.warning("Rețeaua a fost generată în fundal.")
            else:
                st.warning("Nu există suficiente conexiuni pentru o rețea.")
        except Exception as e:
            st.error(f"Eroare rețea: {e}")


    # === TAB 4: EXPORT DATE ===
    with tab4:
        st.subheader("Generare Raport și Export Date")
        
        # Pregatire DataFrame (partea ta veche, pastrata)
        data_export = []
        for d in filtered_docs:
            data_export.append({
                "Titlu": d.title,
                "An": d.publication_year,
                "Autori": ", ".join([author_name(a) for a in d.authors]) if d.authors else "",
                "Sursă": d.source if hasattr(d, 'source') else ""
            })
        df = pd.DataFrame(data_export)

        col_pdf, col_csv = st.columns(2)

        # 1. EXPORT CSV
        with col_csv:
            st.markdown("#### 1. Date Brute (CSV)")
            st.dataframe(df.head(5), use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Descarcă CSV", data=csv, file_name="litstudy_data.csv", mime="text/csv")

        # 2. EXPORT PDF
        with col_pdf:
            st.markdown("#### 2. Raport Oficial (PDF)")
            st.info("Generează raport cu grafice și concluzii.")
            
            if st.button("🖨️ Generează Raport PDF"):
                # Verificam daca utilizatorul a vizitat Tab-ul 1 pentru a genera graficele
                if not st.session_state['figures']:
                    st.error("⚠️ Mergi mai întâi în 'Dashboard Statistici' pentru a se genera graficele!")
                else:
                    with st.spinner("Se generează PDF-ul..."):
                        try:
                            pdf = PDFReport()
                            pdf.add_page()
                            
                            # A. HEADER INFO
                            pdf.chapter_title(f"Rezumat: {len(filtered_docs)} articole analizate")
                            if years:
                                pdf.chapter_body(f"Interval ani: {min(years)} - {max(years)}")
                            
                            # B. TOP AUTORI (TEXT)
                            if filtered_docs:
                                authors_flat = [a.name for d in filtered_docs for a in d.authors or []]
                                top_auth = pd.Series(authors_flat).value_counts().head(5)
                                txt = "\n".join([f"- {n} ({c})" for n, c in top_auth.items()])
                                pdf.chapter_title("Top 5 Autori")
                                pdf.chapter_body(txt)

                            # C. INSERARE GRAFICE SALVATE
                            pdf.chapter_title("Vizualizare Grafica")
                            
                            # Definim descrieri profesionale pentru fiecare tip de grafic
                            info_grafice = {
                                'years': {
                                    'titlu': "1. Evolutia Temporala a Publicatiilor",
                                    'desc': "Acest grafic ilustreaza distributia articolelor pe ani. Se poate observa tendinta de crestere sau scadere a interesului academic pentru acest subiect in perioada selectata."
                                },
                                'authors': {
                                    'titlu': "2. Top 10 Cei Mai Productivi Autori",
                                    'desc': "Analiza autorilor evidentiaza cercetatorii cu cel mai mare numar de contributii in setul de date. Acestia sunt liderii de opinie in domeniu."
                                },
                                'sources': {
                                    'titlu': "3. Principalele Surse de Publicare",
                                    'desc': "Graficul prezinta jurnalele si conferintele unde au aparut cele mai multe articole, indicand unde se poarta discutiile relevante."
                                },
                                'wordcloud': {
                                    'titlu': "4. Analiza Vizuala a Cuvintelor Cheie",
                                    'desc': "Norul de cuvinte (Word Cloud) evidentiaza termenii cei mai frecventi din titlurile articolelor, oferind o imagine rapida a terminologiei dominante."
                                }
                            }

                            # Iteram prin graficele salvate
                            temp_files = []
                            # Iteram prin graficele salvate
                            for name, fig_obj in st.session_state['figures'].items():
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                    # Salvam imaginea temporar
                                    fig_obj.savefig(tmp.name, bbox_inches='tight', dpi=100)
                                    temp_files.append(tmp.name)
                                    
                                    # 1. LOGICA INTELIGENTĂ DE PAGINARE
                                    # O pagina A4 are ~297mm inaltime. 
                                    # Daca cursorul e mai jos de 200mm, nu mai avem loc de grafic + text.
                                    # Fortam pagina noua CA SA TINEM TITLUL LANGA GRAFIC.
                                    if pdf.get_y() > 200: 
                                        pdf.add_page()
                                    
                                    # Verificam daca avem descrieri "premium"
                                    if name in info_grafice:
                                        titlu_afisat = info_grafice[name]['titlu']
                                        descriere_afisata = info_grafice[name]['desc']
                                    else:
                                        titlu_afisat = f"Figura: {name.upper()}"
                                        descriere_afisata = ""

                                    # 2. Titlul Graficului
                                    pdf.set_font('Arial', 'B', 11)
                                    pdf.cell(0, 8, clean_text(titlu_afisat), 0, 1)
                                    
                                    # 3. Descrierea
                                    if descriere_afisata:
                                        pdf.set_font('Arial', 'I', 9)
                                        pdf.multi_cell(0, 5, clean_text(descriere_afisata))
                                        pdf.ln(2)

                                    # 4. Imaginea
                                    # O punem centrata, latime 160mm
                                    try:
                                        pdf.image(tmp.name, w=160, x=25)
                                    except:
                                        pdf.cell(0, 10, "Eroare la incarcarea imaginii", 1, 1)
                                        
                                    pdf.ln(10) # Spatiu dupa grafic pentru a respira

                            # D. TOPIC MODELING
                            if st.session_state['nmf_topics']:
                                pdf.add_page()
                                pdf.chapter_title("Analiza Semantica (Topic Modeling)")
                                
                                # Setam culoarea de fundal pentru un aspect modern (gri foarte deschis)
                                pdf.set_fill_color(245, 245, 245)
                                
                                for t in st.session_state['nmf_topics']:
                                    # Spargem textul in: "Topic X" si "cuvinte"
                                    # Exemplu t: "Topic 1: mere, pere, prune"
                                    parts = t.split(":", 1) 
                                    if len(parts) == 2:
                                        title = parts[0].strip() # "Topic 1"
                                        content = parts[1].strip() # "mere, pere..."
                                        
                                        # 1. Scriem Titlul Topic-ului cu BOLD
                                        pdf.set_font('Arial', 'B', 11)
                                        pdf.cell(30, 10, title + ":", 0, 0) # Celula nu face line break
                                        
                                        # 2. Scriem cuvintele Normal
                                        pdf.set_font('Arial', '', 11)
                                        pdf.multi_cell(0, 10, clean_text(content))
                                        
                                        # 3. Adaugam o linie mica de spatiu pentru aerisire
                                        pdf.ln(2)
                                    else:
                                        # Fallback daca formatul e ciudat
                                        pdf.chapter_body(t)     

                            # E. DESCARCARE
                            pdf_byte = pdf.output(dest='S').encode('latin-1')
                            st.download_button(
                                "📥 Descarcă PDF Final", 
                                data=pdf_byte, 
                                file_name="raport_litstudy.pdf", 
                                mime="application/pdf"
                            )
                            
                            # Cleaning
                            for tf in temp_files: os.remove(tf)

                        except Exception as e:
                            st.error(f"Eroare PDF: {e}")

elif not docs:
    st.info("👈 Începe prin a căuta un termen sau a încărca un fișier în meniul din stânga.")