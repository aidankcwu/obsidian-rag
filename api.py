"""FastAPI REST API for the Obsidian RAG pipeline.

Usage:
    python api.py              Start the API server on port 8000
    uvicorn api:app --reload   Start with auto-reload for development
"""
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from obsrag.config import get_config
from obsrag.rag.indexer import load_documents, build_or_load_index
from obsrag.rag.tags import load_tag_set, build_tag_context
from obsrag.rag.suggest import suggest_links_and_tags, suggest_tags_via_llm
from obsrag.ocr import ocr_pdf_with_llm
from obsrag.writer import write_note
from llama_index.core.postprocessor import SentenceTransformerRerank

app = FastAPI(title="Obsidian RAG API", version="1.0.0")

# Shared resources initialized at startup
cfg = None
docs = None
index = None
tag_set = None
tag_context = None
reranker = None


class SuggestRequest(BaseModel):
    text: str
    top_k: int = 10


class SuggestResponse(BaseModel):
    suggested_links: list[dict]
    suggested_tags: list[dict]
    llm_tags: dict | None = None


class ProcessResponse(BaseModel):
    title: str
    ocr_text: str
    suggested_links: list[dict]
    suggested_tags: list[dict]
    llm_tags: dict | None = None
    note_path: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_tags: int
    vault_path: str


@app.on_event("startup")
def startup():
    """Initialize all shared resources once at startup."""
    global cfg, docs, index, tag_set, tag_context, reranker
    cfg = get_config()
    print("Initializing Obsidian RAG pipeline...")
    docs = load_documents(cfg.vault_path)
    index = build_or_load_index(
        docs, cfg.persist_dir, cfg.embedding.model,
        chunk_size=cfg.embedding.chunk_size,
        chunk_overlap=cfg.embedding.chunk_overlap,
    )
    tag_set = load_tag_set(
        cfg.vault_path,
        style=cfg.tags.style,
        tags_folder_name=cfg.folders.tags,
    )
    tag_context = build_tag_context(docs, tag_set)
    reranker = SentenceTransformerRerank(
        model=cfg.rag.reranker_model,
        top_n=cfg.rag.reranker_top_n,
    )
    print(f"Ready. {len(docs)} docs, {len(tag_set)} tags loaded, {len(tag_context)} tags with context.")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if index is not None else "not_ready",
        index_loaded=index is not None,
        num_tags=len(tag_set) if tag_set else 0,
        vault_path=str(cfg.vault_path) if cfg else "",
    )


@app.get("/tags")
def get_tags() -> list[str]:
    """Return all available tags from the vault."""
    if tag_set is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    return sorted(tag_set)


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    """Suggest wikilinks and tags for the given text (no OCR, no file write)."""
    if index is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    result = suggest_links_and_tags(
        req.text, index, tag_set, docs,
        reranker=reranker, top_k=req.top_k,
    )
    retrieval_tags = [t["title"] for t in result["suggested_tags"]]

    top_score = result["suggested_links"][0]["score"] if result["suggested_links"] else 0

    llm_tags = None
    if len(retrieval_tags) < cfg.rag.min_tags_threshold or top_score < cfg.rag.min_confidence_threshold:
        llm_tags = suggest_tags_via_llm(
            note_text=req.text,
            all_tags=sorted(tag_set),
            retrieval_tags=retrieval_tags,
            tag_context=tag_context,
        )

    return SuggestResponse(
        suggested_links=result["suggested_links"],
        suggested_tags=result["suggested_tags"],
        llm_tags=llm_tags,
    )


@app.post("/process", response_model=ProcessResponse)
async def process(file: UploadFile = File(...)):
    """Upload a PDF, run the full OCR-to-Obsidian pipeline."""
    if index is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file to a temp directory
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # OCR
        input_text, page_images, page_offsets = ocr_pdf_with_llm(tmp_path, model=cfg.ocr.model)

        # Layer 1: Retrieval-based suggestions
        result = suggest_links_and_tags(
            input_text, index, tag_set, docs,
            reranker=reranker, top_k=cfg.rag.top_k,
        )
        retrieval_tags = [t["title"] for t in result["suggested_tags"]]
        top_score = result["suggested_links"][0]["score"] if result["suggested_links"] else 0

        # Layer 2: LLM fallback
        llm_tags = None
        if len(retrieval_tags) < cfg.rag.min_tags_threshold or top_score < cfg.rag.min_confidence_threshold:
            llm_tags = suggest_tags_via_llm(
                note_text=input_text,
                all_tags=sorted(tag_set),
                retrieval_tags=retrieval_tags,
                filename=file.filename,
                tag_context=tag_context,
            )

        # Determine final tags
        if llm_tags:
            final_tags = llm_tags.get("existing_tags", []) + llm_tags.get("new_tags", [])
        else:
            final_tags = retrieval_tags

        references = [link["title"] for link in result["suggested_links"] if link.get("source") == "retrieval"]
        title = tmp_path.stem.replace("_", " ").replace("-", " ").title()

        # Write to Obsidian
        note_path = write_note(
            title=title,
            content=input_text,
            tags=final_tags,
            references=references,
            inbox_path=cfg.inbox_path,
            tag_style=cfg.tags.style,
            template=cfg.note_template,
            page_images=page_images,
            page_offsets=page_offsets,
            attachments_path=cfg.attachments_path,
        )

        return ProcessResponse(
            title=title,
            ocr_text=input_text,
            suggested_links=result["suggested_links"],
            suggested_tags=result["suggested_tags"],
            llm_tags=llm_tags,
            note_path=str(note_path),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
