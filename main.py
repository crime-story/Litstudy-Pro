import streamlit as st
import litstudy
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os

# Configurare pagină
st.set_page_config(page_title="LitStudy Dashboard", layout="wide")

st.title("📚 Analiză bibliometrică cu LitStudy")

# --- FUNCȚIE PENTRU TEXTUL DE PREZENTARE (LANDING PAGE) ---
def show_landing_page():
    st.markdown(""" Acest instrument este conceput pentru a ajuta cercetătorii să navigheze rapid prin literatura de specialitate.""")

# --- 1. SIDEBAR: SELECTARE SURSĂ ---
st.sidebar.header("Metoda de preluare a articolelor")
sursa_date = st.sidebar.radio(
    "Alege sursa datelor:",
    ("Căutare online (DBLP)", "Încărcare fișier local")
)

docs = [] # Lista care va ține articolele

# --- LOGICA PENTRU CĂUTARE ONLINE ---
if sursa_date == "Căutare online (DBLP)":
    st.sidebar.subheader("Parametri de căutare")
    query = st.sidebar.text_input("Cuvinte cheie", value="Machine Learning")
    limit_docs = st.sidebar.slider("Numărul maxim de articole", 100, 500, 100, step=100)
    
    if st.sidebar.button("Caută"):
        with st.spinner('Se descarcă datele...'):
            try:
                docs = litstudy.search_dblp(query, limit=limit_docs)
            except Exception as e:
                st.error(f"Eroare la căutare: {e}")

# --- LOGICA PENTRU ÎNCĂRCARE FIȘIER ---
else:
    st.sidebar.subheader("Upload fișier")
    uploaded_file = st.sidebar.file_uploader(
        "Trage un fișier aici (BibTeX, RIS, CSV)", 
        type=["bib", "ris", "csv"]
    )
    
    if uploaded_file is not None:
        # Salvăm temporar fișierul pentru ca litstudy să îl poată citi
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner('Se procesează fișierul...'):
                if temp_filename.endswith(".bib"):
                    docs = litstudy.load_bibtex(temp_filename)
                elif temp_filename.endswith(".ris"):
                    docs = litstudy.load_ris(temp_filename)
                elif temp_filename.endswith(".csv"):
                    docs = litstudy.load_csv(temp_filename)
                
                st.sidebar.success(f"Fișier încărcat: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Eroare la citirea fișierului: {e}")
            st.warning("Verifică dacă formatul este valid.")

# --- 2. ZONA PRINCIPALĂ DE VIZUALIZARE ---
if docs and len(docs) > 0:
    st.success(f"Au fost găsite {len(docs)} de articole!")

    # --- TAB-URI ---
    tab1, tab2, tab3 = st.tabs(["📈 Statistici generale", "🕸️ Rețea co-autori", "📄 Listă articole"])

    # TAB 1: GRAFICE (Statistici centrate)
    with tab1:
        # --- GRAFIC 1: ANI ---
        st.subheader("Publicații pe ani")
        left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
        with center_col:
            fig1 = plt.figure(figsize=(8, 4)) 
            litstudy.plot_year_histogram(docs)
            st.pyplot(fig1, use_container_width=True)

        st.markdown("---") 

        # --- GRAFIC 2: AUTORI ---
        st.subheader("Top autori")
        left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
        with center_col:
            fig2 = plt.figure(figsize=(8, 5)) 
            litstudy.plot_author_histogram(docs, limit=10)
            st.pyplot(fig2, use_container_width=True)

    # TAB 2: REȚEA CO-AUTORI
    with tab2:
        st.subheader("Rețeaua de co-autori")
        st.info("Poți da zoom și trage de noduri.")
        
        try:
            net_authors = litstudy.build_coauthor_network(docs)
            html_file = "citation.html"
            litstudy.plot_network(net_authors)
            
            if os.path.exists(html_file):
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_string = f.read()
                components.html(html_string, height=750, scrolling=True)

        except Exception as e:
            st.warning("Nu s-a putut genera rețeaua. Posibil prea puține date sau lipsesc autorii.")

    # TAB 3: TABEL
    with tab3:
        st.subheader("Date brute")
        data_list = []
        for d in docs:
            # Verificăm dacă există anul, altfel punem 'N/A'
            year = d.publication_year if hasattr(d, 'publication_year') else 'N/A'
            authors = d.authors if d.authors else []
            
            data_list.append({
                "Titlu": d.title,
                "An": year, 
                "Autori": len(authors)
            })
        st.dataframe(
            data_list, 
            height=700,
            use_container_width=True
        )

elif sursa_date == "Încărcare fișier local" and not uploaded_file:
    st.info("👈 Încarcă un fișier în meniul lateral.")
    show_landing_page()

elif sursa_date == "Căutare online (DBLP)" and not docs:
    st.info("👈 Introdu termenii în meniul lateral.")
    show_landing_page()