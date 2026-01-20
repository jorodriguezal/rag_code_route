# src/indexer.py
from __future__ import annotations

from pathlib import Path
import os

import chromadb
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "code_route"


def _configure_llamaindex():
    """Configure LlamaIndex global Settings with Azure LLM + Azure Embeddings."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Chat (GPT-4.1)
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    # Embeddings (text-embedding-3-large)
    emb_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    emb_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

    llm = AzureOpenAI(
        model="gpt-4.1",  
        deployment_name=chat_deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=chat_api_version,
        temperature=0.2,
    )

    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name=emb_deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=emb_api_version,
    )

    # Réglages globaux LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 900
    Settings.chunk_overlap = 150


def build_or_load_index() -> VectorStoreIndex:
    """
    - Lit les PDF depuis data/
    - Crée (ou recharge) un index vectoriel persisté dans Chroma
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier data/ introuvable: {DATA_DIR}")

    _configure_llamaindex()

    # Chroma persistent client
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Si la collection contient déjà des embeddings, on recharge directement
    # (Heuristique simple: count)
    try:
        existing = collection.count()
    except Exception:
        existing = 0

    if existing and existing > 0:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        return index

    # Sinon, on indexe les PDF
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=False,
        required_exts=[".pdf"],
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def reset_index():
    """Supprime la base Chroma locale pour forcer une ré-indexation propre."""
    if CHROMA_DIR.exists():
        # supprime tout le dossier
        for p in CHROMA_DIR.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(CHROMA_DIR.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        CHROMA_DIR.rmdir()
