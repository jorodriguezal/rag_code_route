# app.py
import os
from dotenv import load_dotenv
import streamlit as st

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from src.indexer import build_or_load_index, reset_index
from src.prompts import QA_SYSTEM

load_dotenv()

st.set_page_config(page_title="RAG Code de la route", page_icon="🚗", layout="wide")
st.title("🚗 Assistant RAG – Code de la route (PDF)")

# ---------------------------
# 1) Vérifier les variables .env
# ---------------------------
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

# ---------------------------
# 2) Cache index (important)
# ---------------------------
@st.cache_resource
def get_index():
    return build_or_load_index()

# ---------------------------
# 3) Mémoire conversation (comme ChatGPT)
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # liste de {"role": "user"/"assistant", "content": "...", "sources": [...]}

def build_history_text(history, max_turns=6) -> str:
    """
    Construit un texte d'historique court pour aider le LLM
    (sans tout envoyer pour éviter trop de tokens).
    """
    last = history[-2 * max_turns :]
    lines = []
    for m in last:
        role = "Étudiant" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.subheader(" Paramètres")
    top_k = st.slider("Nombre de passages récupérés (k)", 2, 12, 5)
    cutoff = st.slider("Seuil similarité (cutoff)", 0.0, 0.6, 0.2, 0.05)

    st.divider()

    if st.button(" Ré-indexer (reset Chroma)", use_container_width=True):
        st.cache_resource.clear()
        reset_index()
        st.success("Index supprimé  Relance une question (l'index sera reconstruit).")
        st.stop()

    if st.button(" Effacer la conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(" Mets tes PDF dans `data/` (texte sélectionnable).")

# ---------------------------
# Afficher l'historique
# ---------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Afficher sources si présentes (uniquement pour assistant)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(" Sources (PDF / page / score)"):
                for s in msg["sources"]:
                    st.write(f"- `{s['file']}` — page **{s['page']}** — score **{s['score']}**")
                    if s.get("excerpt"):
                        st.caption(s["excerpt"])

# ---------------------------
# Input style ChatGPT
# ---------------------------
query = st.chat_input("Pose ta question sur le Code de la route...")

if query:
    # afficher message user
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Charger l'index
    index = get_index()

    # Construire le query engine (retriever + filtre)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=float(cutoff))],
    )

    # Injecter historique (aide pour questions du type "et après ?" / "explique mieux")
    history_text = build_history_text(st.session_state.chat_history, max_turns=6)

    # Important : autoriser la reformulation à partir du contexte (pas d'invention)
    full_query = f"""{QA_SYSTEM}

Règle supplémentaire:
- Si la réponse n'est pas mot pour mot dans le contexte mais que le contexte contient des éléments pertinents,
  alors reformule en t'appuyant STRICTEMENT sur ces éléments, sans ajouter d'information externe.

Historique récent (pour contexte conversationnel) :
{history_text}

Question actuelle :
{query}
"""

    with st.chat_message("assistant"):
        with st.spinner(" Recherche dans le PDF + génération..."):
            response = query_engine.query(full_query)

        answer = str(response)
        st.write(answer)

        # Récupérer sources
        sources = []
        try:
            nodes = response.source_nodes or []
        except Exception:
            nodes = []

        for n in nodes:
            meta = n.node.metadata or {}
            file_name = (
                meta.get("file_name")
                or meta.get("filename")
                or meta.get("source")
                or meta.get("document")
                or "PDF"
            )
            page = meta.get("page_label") or meta.get("page") or "?"
            score = getattr(n, "score", None)
            score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "?"

            excerpt = n.node.get_text().replace("\n", " ")
            if len(excerpt) > 280:
                excerpt = excerpt[:280] + "..."

            sources.append(
                {"file": file_name, "page": page, "score": score_txt, "excerpt": excerpt}
            )

        if sources:
            with st.expander(" Sources (PDF / page / score)"):
                for s in sources:
                    st.write(f"- `{s['file']}` — page **{s['page']}** — score **{s['score']}**")
                    st.caption(s["excerpt"])

    # Sauvegarder dans l'historique
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
