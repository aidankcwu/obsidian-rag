"""Build or load the vector store index"""
from pathlib import Path
from llama_index.readers.obsidian import ObsidianReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding


def load_documents(vault_path: Path):
    """Load all documents from the Obsidian vault."""
    reader = ObsidianReader(input_dir=str(vault_path))
    docs = reader.load_data()
    print(f"Loaded {len(docs)} notes from vault")
    return docs


def build_or_load_index(
    docs,
    persist_dir: Path,
    embedding_model: str,
) -> VectorStoreIndex:
    """
    Build a new index from documents, or load from disk if it already exists
    """
    embed_model = OpenAIEmbedding(model=embedding_model)

    if persist_dir.exists():
        print("Loading existing index")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        print("Building index, using openai")
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print(f"Index saved to {persist_dir}")

    return index
