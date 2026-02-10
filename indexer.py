"""Build or load the vector store index."""
from pathlib import Path
from llama_index.readers.obsidian import ObsidianReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
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
    Build a new index from documents, or load from disk if it already exists.
    Uses chunking with 50 token overlap for better context.
    """
    embed_model = OpenAIEmbedding(model=embedding_model)

    if persist_dir.exists():
        print("Loading existing index")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        print("Building index, using openai")

        # Custom chunking with overlap for better context
        parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embed_model,
            transformations=[parser],
        )

        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print(f"Index saved to {persist_dir}")

    return index
