# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from src.indexer import build_or_load_index, reset_index
from src.prompts import QA_SYSTEM, SUMMARY_SYSTEM


load_dotenv()

st.set_page_config(page_title="RAG Code de la route", page_icon="🚗", layout="wide")
st.title(" Assistant RAG – Code de la route (PDF)")

# --- vérification env ---
needed = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_API_VERSION",
]
missing = [k for k in needed if not os.getenv(k)]
if missing:
    st.error("Variables manquantes dans .env : " + ", ".join(missing))
    st.stop()

# --- cache index ---
@st.cache_resource
def get_index():
    return build_or_load_index()

with st.sidebar:
    st.subheader(" Index")
    if st.button(" Ré-indexer (reset Chroma)"):
        st.cache_resource.clear()
        reset_index()
        st.success("Index supprimé. Relance l’app ou clique sur une action.")
        st.stop()

    top_k = st.slider("Nombre de passages récupérés (k)", 2, 10, 5)
    st.caption("Mets ton PDF dans `data/` puis pose des questions.")

tab1, tab2 = st.tabs([" Questions / Réponses", " Résumé"])

# =========================
# TAB 1 : Q&A RAG
# =========================
with tab1:
    st.subheader(" Pose une question sur le Code de la route")

    query = st.text_input(
        "Question",
        placeholder="Ex: Que signifie un panneau triangulaire ?",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        ask_btn = st.button("Répondre", use_container_width=True)
    with col2:
        show_sources = st.toggle("Afficher les sources détaillées", value=True)

    if ask_btn:
        if not query.strip():
            st.warning("Écris une question d’abord.")
            st.stop()

        index = get_index()

        # Retriever + Query Engine
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.2)],
        )

        # On ajoute le "system" en préfixe au prompt via Settings.llm
        # (LlamaIndex: on peut injecter dans le query)
        full_query = f"{QA_SYSTEM}\n\nQuestion: {query}"

        with st.spinner("Recherche dans le PDF + génération..."):
            response = query_engine.query(full_query)

        st.markdown("###  Réponse")
        st.write(str(response))

        # Sources (si disponibles)
        if show_sources:
            st.markdown("###  Sources")
            try:
                nodes = response.source_nodes or []
            except Exception:
                nodes = []

            if not nodes:
                st.info("Aucune source affichable (selon la réponse).")
            else:
                for i, n in enumerate(nodes, 1):
                    meta = n.node.metadata or {}
                    file_name = meta.get("file_name") or meta.get("filename") or meta.get("source") or "PDF"
                    page = meta.get("page_label") or meta.get("page") or "?"
                    score = getattr(n, "score", None)

                    st.write(f"**[{i}]** `{file_name}` — page **{page}**" + (f" — score {score:.3f}" if score else ""))
                    excerpt = n.node.get_text()[:700].replace("\n", " ")
                    st.caption(excerpt + ("..." if len(n.node.get_text()) > 700 else ""))

# =========================
# TAB 2 : SUMMARY
# =========================
with tab2:
    st.subheader(" Résumé (global / points clés)")

    mode = st.selectbox(
        "Type de résumé",
        ["Résumé court", "Points clés (bullet points)", "Fiche de révision simple"],
        index=1,
    )

    run_sum = st.button("Générer le résumé", use_container_width=True)

    if run_sum:
        index = get_index()

        # Pour un résumé global, on récupère des passages "représentatifs"
        # Astuce simple: on fait plusieurs requêtes internes puis on résume.
        # (Simple, académique, et marche bien.)
        seed_queries = [
            "résume les règles importantes",
            "signalisation panneaux marquage",
            "priorités intersections",
            "vitesse distance sécurité",
            "sanctions alcool téléphone",
        ]

        retriever = VectorIndexRetriever(index=index, similarity_top_k=6)
        collected = []
        for q in seed_queries:
            nodes = retriever.retrieve(q)
            for n in nodes:
                txt = n.node.get_text()
                if txt and txt not in collected:
                    collected.append(txt)

        # Limiter la taille envoyée au modèle
        joined = "\n\n".join(collected[:25])

        if mode == "Résumé court":
            instr = "Fais un résumé court (8 à 12 lignes) en français."
        elif mode == "Fiche de révision simple":
            instr = (
                "Crée une fiche de révision simple: "
                "1) Définitions, 2) Règles clés, 3) Panneaux/Signalisation (si présent), "
                "4) Sanctions (si présent)."
            )
        else:
            instr = "Donne uniquement les points clés sous forme de puces (15 à 25 puces max)."

        prompt = f"""{SUMMARY_SYSTEM}

Instruction: {instr}

Texte à résumer (extraits du PDF):
{joined}

Sortie attendue:
- en français
- clair et structuré
"""

        with st.spinner("Génération du résumé..."):
            # Utiliser directement le LLM configuré dans Settings (déjà set dans indexer)
            llm = Settings.llm
            resp = llm.complete(prompt)

        st.markdown("###  Résumé")
        st.write(resp.text)
