# app.py
# ------------------------------------------------------------

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from src.indexer import build_or_load_index, reset_index, get_data_dir, list_pdfs
from src.prompts import QA_SYSTEM, SUMMARY_SYSTEM

# Charge les variables d'environnement du fichier .env
load_dotenv()

# ------------------------------------------------------------
# 1) Configuration de la page Streamlit (titre, layout, icône)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Assistant RAG – Code de la route",
    page_icon="",
    layout="wide",
)

# ------------------------------------------------------------
# 2) Vérification des variables nécessaires (.env)
#    -> Evite les erreurs 401/404 Azure
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 3) Session State : mémoire de conversation (comme ChatGPT)
# ------------------------------------------------------------
if "chat_history" not in st.session_state:
    # Chaque message sera un dict :
    # {"role": "user"/"assistant", "content": "...", "sources": [...]}
    st.session_state.chat_history = []

# Un petit flag pour savoir qu'on doit reconstruire l'index
if "need_rebuild" not in st.session_state:
    st.session_state.need_rebuild = False

# Un message “toast” à afficher après rerun 
if "toast_msg" not in st.session_state:
    st.session_state.toast_msg = ""

def build_history_text(history, max_turns=6) -> str:
    """
    On transforme les derniers échanges en texte pour donner au modèle
    un contexte conversationnel (ex: l'utilisateur dit "et après ?").
    """
    last = history[-2 * max_turns :]
    lines = []
    for m in last:
        role = "Étudiant" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

# ------------------------------------------------------------
# 4) Cache : index vectoriel (important)
#    Streamlit relance souvent le script : on ne veut pas réindexer
#    le PDF à chaque refresh. Donc cache_resource.
# ------------------------------------------------------------
@st.cache_resource
def get_index():
    return build_or_load_index()

def rebuild_index_now():
    """
    Reconstruit l'index immédiatement (avec spinner) pour éviter
    le scénario : "reset -> page blanche -> index sera reconstruit plus tard".
    """
    with st.spinner(" Reconstruction de l’index... (lecture PDF + embeddings)"):
        # Cette ligne reconstruit l’index si Chroma est vide
        _ = get_index()

# ------------------------------------------------------------
# 5) UI Header
# ------------------------------------------------------------
st.markdown("##  Assistant RAG – Code de la route (PDF)")
st.caption("Pose des questions sur tes PDF. Réponses basées sur les passages retrouvés (RAG) + sources.")

# Affiche un toast si on en a un, puis on le vide
if st.session_state.toast_msg:
    st.toast(st.session_state.toast_msg)
    st.session_state.toast_msg = ""

# ------------------------------------------------------------
# 6) Sidebar : paramètres + actions
# ------------------------------------------------------------
with st.sidebar:
    st.subheader(" Paramètres RAG")

    # k = nombre de chunks récupérés
    top_k = st.slider("Nombre de passages récupérés (k)", 2, 12, 5)

    # cutoff = filtre de pertinence (score minimum)
    cutoff = st.slider("Seuil similarité (cutoff)", 0.0, 0.6, 0.20, 0.05)

    st.divider()
    st.subheader(" Documents")

    pdfs = list_pdfs()
    st.write(f"PDF détectés : **{len(pdfs)}**")

    if pdfs:
        with st.expander("Voir la liste des PDF"):
            for p in pdfs:
                st.write(f"- {p.name}")

    st.divider()

    # --- Ré-indexer proprement (SANS stop qui casse l'UI)
    if st.button(" Ré-indexer (reset Chroma)", use_container_width=True):
        # 1) On supprime Chroma
        reset_index()

        # 2) On vide le cache de l’index Streamlit
        st.cache_resource.clear()

        # 3) On active un flag pour reconstruire automatiquement après rerun
        st.session_state.need_rebuild = True

        # 4) On met un petit message toast à afficher après rerun
        st.session_state.toast_msg = "Index supprimé  Rebuild automatique en cours..."

        # 5) On rerun : la page reste “normale” 
        st.rerun()

    # --- Effacer conversation = comme “new chat”
    if st.button("🧹 Effacer la conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.toast_msg = "Conversation effacée "
        st.rerun()

    st.caption(" Après ajout d’un PDF : ré-indexer pour l’inclure dans la recherche.")

# ------------------------------------------------------------
# 7) Si on doit reconstruire l’index (après reset / upload)
# ------------------------------------------------------------
if st.session_state.need_rebuild:
    # IMPORTANT : on rebuild ici sans faire stop()
    rebuild_index_now()
    st.session_state.need_rebuild = False
    st.session_state.toast_msg = "Index reconstruit  Tu peux poser des questions."
    st.rerun()

# ------------------------------------------------------------
# 8) Tabs : Chat / Résumé / Upload
# ------------------------------------------------------------
tab_chat, tab_resume, tab_docs = st.tabs([" Chat (Q&A)", " Résumé", " Upload PDF"])

# ============================================================
# TAB 1 : Chat (Q&A) style ChatGPT
# ============================================================
with tab_chat:
    st.subheader(" Questions / Réponses (RAG)")

    # 1) On affiche l'historique comme ChatGPT
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # On affiche les sources uniquement pour les réponses assistant
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(" Sources (PDF / page / score)"):
                    for s in msg["sources"]:
                        st.write(f"- `{s['file']}` — page **{s['page']}** — score **{s['score']}**")
                        if s.get("excerpt"):
                            st.caption(s["excerpt"])

    # 2) Input en bas (comme ChatGPT)
    query = st.chat_input("Pose ta question sur le Code de la route…")

    if query:
        # On ajoute la question dans l'historique + on l'affiche
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # On charge l'index (s’il existe déjà c'est instantané)
        index = get_index()

        # Le retriever récupère les chunks les plus proches de la question
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

        # Le query_engine : retrieval + filtre similarité
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=float(cutoff))],
        )

        # Historique conversationnel (améliore "et après ?" / "explique mieux")
        history_text = build_history_text(st.session_state.chat_history, max_turns=6)

        # Prompt final : règles + historique + question
        full_query = f"""{QA_SYSTEM}

Règle supplémentaire:
- Si la réponse n'est pas mot pour mot dans le contexte mais que le contexte contient des éléments pertinents,
  alors reformule en t'appuyant STRICTEMENT sur ces éléments, sans ajouter d'information externe.

Historique récent :
{history_text}

Question actuelle :
{query}
"""

        # On génère la réponse et on affiche côté assistant
        with st.chat_message("assistant"):
            with st.spinner(" Recherche dans le PDF + génération..."):
                response = query_engine.query(full_query)

            answer = str(response)
            st.write(answer)

            # On récupère les sources renvoyées par LlamaIndex
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

                sources.append({"file": file_name, "page": page, "score": score_txt, "excerpt": excerpt})

            if sources:
                with st.expander(" Sources (PDF / page / score)"):
                    for s in sources:
                        st.write(f"- `{s['file']}` — page **{s['page']}** — score **{s['score']}**")
                        st.caption(s["excerpt"])

        # Enfin, on stocke la réponse dans l'historique (comme ChatGPT)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

# ============================================================
# TAB 2 : Résumé
# ============================================================
with tab_resume:
    st.subheader(" Résumé à partir des PDF")

    mode = st.selectbox(
        "Type de sortie",
        ["Résumé court", "Points clés", "Fiche de révision"],
        index=1
    )

    theme = st.text_input(
        "Thème (optionnel)",
        value="priorité, vitesse, alcool, téléphone, stationnement",
        help="Si tu veux un résumé ciblé sur un thème."
    )

    if st.button("Générer le résumé", use_container_width=True):
        index = get_index()
        retriever = VectorIndexRetriever(index=index, similarity_top_k=6)

        # On récupère des passages représentatifs pour résumer
        seed_queries = [t.strip() for t in theme.split(",") if t.strip()]
        if not seed_queries:
            seed_queries = ["règles importantes", "priorité", "vitesse", "sanctions"]

        collected = []
        for q in seed_queries[:10]:
            nodes = retriever.retrieve(q)
            for n in nodes:
                txt = n.node.get_text()
                if txt and txt not in collected:
                    collected.append(txt)

        joined = "\n\n".join(collected[:30])

        if mode == "Résumé court":
            instr = "Fais un résumé court (8 à 12 lignes) en français."
        elif mode == "Fiche de révision":
            instr = (
                "Crée une fiche de révision simple : "
                "1) définitions, 2) règles clés, 3) exceptions, 4) sanctions si présentes."
            )
        else:
            instr = "Donne uniquement les points clés sous forme de puces (15 à 25 puces max)."

        prompt = f"""{SUMMARY_SYSTEM}

Instruction : {instr}

Texte à résumer (extraits des PDF) :
{joined}

Contraintes :
- en français
- clair et structuré
- ne pas inventer
"""

        with st.spinner(" Génération du résumé..."):
            llm = Settings.llm
            resp = llm.complete(prompt)

        st.markdown("###  Résultat")
        st.write(resp.text)

# ============================================================
# TAB 3 : Upload PDF
# ============================================================
with tab_docs:
    st.subheader(" Upload de PDF")

    st.info(
        "Dépose un ou plusieurs PDF ici. Ils seront copiés dans `data/`.\n"
        "Ensuite, clique sur **Ré-indexer** (ou active le rebuild automatique)."
    )

    uploaded = st.file_uploader(
        "Dépose tes PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(" Enregistrer les PDF", use_container_width=True):
            if not uploaded:
                st.warning("Aucun PDF uploadé.")
            else:
                saved = 0
                for uf in uploaded:
                    # On sécurise le nom de fichier (évite les chemins bizarres)
                    safe_name = Path(uf.name).name.replace("..", "").replace("/", "_").replace("\\", "_")
                    out_path = data_dir / safe_name
                    with open(out_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved += 1

                st.success(f" {saved} PDF enregistrés dans {data_dir}")

                # On déclenche automatiquement une reconstruction d'index
                # (comme ça l'utilisateur n'a pas besoin de cliquer ailleurs)
                reset_index()
                st.cache_resource.clear()
                st.session_state.need_rebuild = True
                st.session_state.toast_msg = "PDF ajoutés  Rebuild automatique en cours..."
                st.rerun()

    with col2:
        if st.button(" Supprimer tous les PDF (data/)", use_container_width=True):
            for p in list_pdfs():
                try:
                    p.unlink()
                except Exception:
                    pass

            # On reset l'index car les docs ont disparu
            reset_index()
            st.cache_resource.clear()

            st.session_state.toast_msg = "Tous les PDF supprimés "
            st.rerun()

    st.divider()
    st.caption("Conseil : utilise des PDF avec texte sélectionnable (pas scanné).")
