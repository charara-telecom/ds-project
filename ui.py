import streamlit as st
from dotenv import load_dotenv
import os
from gemini_helper import GeminiHelper
from rag_service import answer_question

load_dotenv()

GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS").split("|||||")

@st.cache_resource
def load_gemini():
    return GeminiHelper(api_keys=GEMINI_API_KEYS)

gemini = load_gemini()

# ================================
# 🌟 PAGE CONFIG
# ================================
st.set_page_config(
    page_title="🎬 Real-Time Movie Review RAG",
    page_icon="🎥",
    layout="centered"
)

# ================================
# 🌈 CUSTOM CSS (Nettoyage des marges et ajustements du bouton)
# ================================
st.markdown("""
    <style>
        /* GLOBAL FONT */
        * {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        /* BACKGROUND */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e3f2fd 100%);
            color: #111;
        }

        /* Conteneur principal - NETTOYAGE DES MARGES INUTILES */
        .main .block-container {
            max-width: 750px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            /* Important : Streamlit ajoute des marges par défaut, nous les gérons */
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* CARD STYLE - Moderne, basé sur votre couleur blanche */
        .card {
            padding: 25px;
            border-radius: 20px;
            background-color: #ffffff;
            border: 1px solid #d1d5db; 
            box-shadow: 0 10px 20px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.03); 
            margin-bottom: 25px;
        }

        /* BUTTON STYLE - Augmentation de la taille du bouton et ajustement du padding */
        .stButton>button {
            width: 120%;
            background-color: #3b82f6;
            color: white;
            font-weight: 600;
            font-size: 1.15rem; /* Légèrement plus grand */
            padding: 0.85rem 0; /* Un peu plus de padding vertical */
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 10px rgba(59, 130, 246, 0.4);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #60a5fa;
            transform: scale(1.02);
            box-shadow: 0 6px 15px rgba(59, 130, 246, 0.5);
        }

        /* TEXT INPUT STYLE - Rendu moins haut */
        div.stTextInput > div[data-baseweb="input"] > div {
            background-color: #f1f5f9;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            padding: 0.3rem 1rem; /* Padding réduit */
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        }
        div.stTextInput > div[data-baseweb="input"] > div:focus-within {
             border-color: #3b82f6 !important;
        }


        /* HEADINGS */
        h1, h2, h3 {
            color: #1e3a8a;
            font-weight: 700;
        }
        
        /* Nettoyer l'espace après le titre H3 dans les cartes */
        h3 {
            margin-top: 0;
            margin-bottom: 15px; 
        }

        /* INFOBAR / PROJECT CARD */
        .info-card {
            background: #eef2ff;
            border-left: 5px solid #3b82f6;
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }
        
        .info-card ul {
            padding-left: 20px;
            margin-top: 8px;
            margin-bottom: 0;
            list-style-type: none;
        }
        
        .info-card ul li:before {
            content: "•"; 
            color: #3b82f6; 
            font-weight: bold; 
            display: inline-block; 
            width: 1em;
            margin-left: -1em;
        }
        
        /* NETTOYAGE FINAL : Supprimer les marges Streamlit par défaut sous le bouton */
        div.stButton {
            margin-top: 15px; /* Ajoutez un peu de marge en haut pour séparer de l'input */
            margin-bottom: 0px; /* Supprimez la marge en bas */
        }
        
        /* S'assurer qu'il n'y a pas d'espace blanc sous les cartes */
        .stAlert {
            margin-bottom: 0;
        }
    </style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown("<h1 style='text-align:center;'>🎬 Real-Time Movie Review RAG</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; opacity:0.8; font-size:1.05rem; margin-bottom: 30px;'>Live movie review analysis using Kafka streaming + vector database + RAG summarization.</p>",
    unsafe_allow_html=True
)

# ================================
# PROJECT INFO CARDS (Pas de conteneurs vides ici)
# ================================
# Carte 1: Overview
st.markdown("<div class='info-card'><b>📌 Project Overview:</b> "
            "This project ingests real-time movie reviews "
            "via Apache Kafka, stores them in a vector database, and generates context-aware answers "
            "using Retrieval-Augmented Generation (RAG).</div>", unsafe_allow_html=True)

# Carte 2: Key Components 
st.markdown("<div class='info-card'><b>⚙️ Key Components:</b> "
            "<ul>"
            "<li>Streaming ingestion with Apache Kafka</li>"
            "<li>Vector DB (Chroma) for fast embeddings</li>"
            "<li>Temporal-aware retrieval for freshest reviews</li>"
            "<li>RAG model to generate summaries and insights</li>"
            "</ul></div>", unsafe_allow_html=True)


# ================================
# USER INPUT CARD (Carte principale sans espace vide à l'intérieur)
# ================================

# st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### 🔍 Ask a question about any movie:")

# L'input
query = st.text_input(
    "Ask a question about any movie:",
    label_visibility="collapsed",
    placeholder="Ex: What do people think about the new action movie 'Stellar Siege'?"
)

# Centrage du bouton dans une colonne plus large pour qu'il soit bien centré
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col2:
    ask = st.button("🚀 Ask the system")

st.markdown("</div>", unsafe_allow_html=True) # Fermeture immédiate de la carte

# ================================
# OUTPUT AREA (Pas de conteneurs vides ici non plus)
# ================================
if ask:
    if query.strip() == "":
        st.warning("⚠️ Please type a question to get real-time insights.")
    else:
        with st.spinner("🔎 Analyzing real-time reviews..."):


            
            result = answer_question(gemini, query)

            st.subheader("💡 AI Answer")
            st.info(result["answer"])

            with st.expander("Debug (filters / retrieved / used_filter)"):
                st.json(result)

            st.markdown("</div>", unsafe_allow_html=True)