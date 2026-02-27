"""Build or load the vector store index."""
import json
import re
from pathlib import Path
from llama_index.readers.obsidian import ObsidianReader
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_path(persist_dir: Path) -> Path:
    """Manifest lives alongside the index dir: .obsrag/manifest.json"""
    return persist_dir.parent / "manifest.json"


def _load_manifest(persist_dir: Path) -> dict:
    """Load manifest from disk, returning {} if it doesn't exist."""
    path = _manifest_path(persist_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict, persist_dir: Path) -> None:
    """Persist manifest to disk."""
    path = _manifest_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _build_manifest_from_docs(docs: list, vault_path: Path) -> dict:
    """
    Build an initial manifest from an already-loaded docs list.

    Groups documents by their source file, records current mtime and doc_ids.
    Used on first run when no manifest exists yet.
    """
    manifest = {}
    for doc in docs:
        folder = doc.metadata.get("folder_path", "")
        fname = doc.metadata.get("file_name", "")
        if not folder or not fname:
            continue
        try:
            rel = str((Path(folder) / fname).relative_to(vault_path))
        except ValueError:
            continue
        file_path = vault_path / rel
        if not file_path.exists():
            continue
        entry = manifest.setdefault(rel, {
            "last_modified": file_path.stat().st_mtime,
            "doc_ids": [],
        })
        entry["doc_ids"].append(doc.id_)
    return manifest


# ---------------------------------------------------------------------------
# Shared insert helper
# ---------------------------------------------------------------------------

def _insert_vault_doc(
    index: VectorStoreIndex,
    file_path: Path,
    vault_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Document:
    """
    Read a vault .md file, build a Document with ObsidianReader-compatible
    metadata, chunk it, insert into the index, and return the Document.
    """
    text = file_path.read_text(encoding="utf-8")
    wikilinks = re.findall(r'\[\[([^\]]+)\]\]', text)

    doc = Document(
        text=text,
        metadata={
            "file_name": file_path.name,
            "folder_path": str(file_path.parent),
            "folder_name": file_path.parent.name,
            "note_name": file_path.stem,
            "wikilinks": wikilinks,
            "backlinks": [],
        },
    )

    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents([doc])
    index.insert_nodes(nodes)
    return doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> VectorStoreIndex:
    """
    Build a new index from documents, or load from disk if it already exists.
    Uses chunking with configurable overlap for better context.
    """
    embed_model = OpenAIEmbedding(model=embedding_model)

    if persist_dir.exists():
        print("Loading existing index")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        print("Building index, using openai")

        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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


def sync_index(
    index: VectorStoreIndex,
    docs: list,
    vault_path: Path,
    persist_dir: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Sync the vector index against the current state of the vault.

    Compares each vault .md file's mtime against a manifest. Re-indexes files
    that changed, inserts new files, and removes deleted files — without a full
    rebuild. On first run (no manifest), bootstraps the manifest from the
    already-loaded docs and returns without modifying the index.

    Call at startup after build_or_load_index so every session starts with an
    up-to-date index that reflects manual edits made in Obsidian.
    """
    if not _manifest_path(persist_dir).exists():
        manifest = _build_manifest_from_docs(docs, vault_path)
        _save_manifest(manifest, persist_dir)
        print(f"Manifest initialised: {len(manifest)} files tracked")
        return  # nothing to sync on first run — index matches current vault

    manifest = _load_manifest(persist_dir)
    vault_files = {
        str(p.relative_to(vault_path)): p
        for p in vault_path.rglob("*.md")
    }

    changed = new = deleted = 0

    # Check for new or modified files
    for rel_path, file_path in vault_files.items():
        mtime = file_path.stat().st_mtime
        if rel_path not in manifest:
            doc = _insert_vault_doc(index, file_path, vault_path, chunk_size, chunk_overlap)
            manifest[rel_path] = {"last_modified": mtime, "doc_ids": [doc.id_]}
            new += 1
        elif abs(mtime - manifest[rel_path]["last_modified"]) > 0.001:
            for doc_id in manifest[rel_path]["doc_ids"]:
                index.delete_ref_doc(doc_id, delete_from_docstore=True)
            doc = _insert_vault_doc(index, file_path, vault_path, chunk_size, chunk_overlap)
            manifest[rel_path] = {"last_modified": mtime, "doc_ids": [doc.id_]}
            changed += 1

    # Check for deleted files
    for rel_path in list(manifest):
        if rel_path not in vault_files:
            for doc_id in manifest[rel_path]["doc_ids"]:
                index.delete_ref_doc(doc_id, delete_from_docstore=True)
            del manifest[rel_path]
            deleted += 1

    if changed or new or deleted:
        index.storage_context.persist(persist_dir=str(persist_dir))
        _save_manifest(manifest, persist_dir)
        print(f"Sync: {changed} updated, {new} new, {deleted} deleted")
    else:
        print("Sync: index is up to date")


def add_note_to_index(
    index: VectorStoreIndex,
    note_path: Path,
    vault_path: Path,
    persist_dir: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Document:
    """
    Add a single newly-written note to the existing index without a full rebuild.

    Inserts the note, re-persists the index, and updates the manifest so
    sync_index can track the note correctly on future startups.

    Returns the Document created so callers can append it to their docs list.
    """
    doc = _insert_vault_doc(index, note_path, vault_path, chunk_size, chunk_overlap)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"Index updated: added '{note_path.stem}' ({len(doc.text.splitlines())} lines)")

    # Update manifest using the same vault-relative key that sync_index uses
    manifest = _load_manifest(persist_dir)
    try:
        rel_path = str(note_path.relative_to(vault_path))
    except ValueError:
        rel_path = str(note_path)  # fallback if note is outside vault
    manifest[rel_path] = {
        "last_modified": note_path.stat().st_mtime,
        "doc_ids": [doc.id_],
    }
    _save_manifest(manifest, persist_dir)

    return doc
