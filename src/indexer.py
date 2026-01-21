# src/indexer.py
from __future__ import annotations

from pathlib import Path
import os
import shutil

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

def get_data_dir() -> Path:
    return DATA_DIR

def list_pdfs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"))

def _configure_llamaindex():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

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

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 900
    Settings.chunk_overlap = 150

def build_or_load_index() -> VectorStoreIndex:
    DATA_DIR.mkdir(exist_ok=True)

    # Vérifier qu'on a au moins un PDF
    pdfs = list_pdfs()
    if not pdfs:
        raise FileNotFoundError(
            f"Aucun PDF dans {DATA_DIR}. Ajoute un PDF (onglet Upload) puis ré-indexe."
        )

    _configure_llamaindex()

    # Chroma persistent client
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Si collection déjà remplie -> on recharge
    try:
        existing = collection.count()
    except Exception:
        existing = 0

    if existing and existing > 0:
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    # Sinon on indexe les PDF présents dans data/
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
    """Supprime complètement la base Chroma locale (force rebuild)."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
