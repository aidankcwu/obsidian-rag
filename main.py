"""Entry point for the Obsidian RAG system.

Usage:
    python main.py <pdf_path>    Process a single PDF
    python main.py --watch       Watch folder for new PDFs
"""
import sys
from pathlib import Path
from config import VAULT_PATH, PERSIST_DIR, EMBEDDING_MODEL
from indexer import load_documents, build_or_load_index
from tags import load_tag_set, build_tag_context
from suggest import suggest_links_and_tags, suggest_tags_via_llm
from ocr import ocr_pdf_with_llm
from write_to_obsidian import write_note
from watcher import watch_loop
from llama_index.core.postprocessor import SentenceTransformerRerank

# Thresholds for LLM fallback
MIN_TAGS_THRESHOLD = 3
MIN_CONFIDENCE_THRESHOLD = 0.4


def setup():
    """Initialize all shared resources once."""
    docs = load_documents(VAULT_PATH)
    index = build_or_load_index(docs, PERSIST_DIR, EMBEDDING_MODEL)
    tag_set = load_tag_set(VAULT_PATH)
    tag_context = build_tag_context(docs, tag_set)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
    )
    return docs, index, tag_set, tag_context, reranker


def process_pdf(pdf_path: Path, docs, index, tag_set, tag_context, reranker):
    """Run the full pipeline on a single PDF."""
    # OCR
    print(f"Processing PDF: {pdf_path}")
    input_text = ocr_pdf_with_llm(pdf_path)
    print(f"\n--- OCR Output ---\n{input_text[:500]}...\n")

    # Layer 1: Retrieval-based suggestions (retrieve 10, rerank to 5)
    result = suggest_links_and_tags(
        input_text,
        index,
        tag_set,
        docs,
        reranker=reranker,
        top_k=10,
    )
    retrieval_tags = [t["title"] for t in result["suggested_tags"]]

    # Check confidence
    top_score = result["suggested_links"][0]["score"] if result["suggested_links"] else 0

    # Layer 2: LLM fallback if not enough tags or low retrieval confidence
    if len(retrieval_tags) < MIN_TAGS_THRESHOLD or top_score < MIN_CONFIDENCE_THRESHOLD:
        print(f"[LLM fallback triggered: {len(retrieval_tags)} tags, top_score={top_score:.2f}]")
        llm_result = suggest_tags_via_llm(
            note_text=input_text,
            all_tags=sorted(tag_set),
            retrieval_tags=retrieval_tags,
            filename=pdf_path.name,
            tag_context=tag_context,
        )
        result["llm_tags"] = llm_result

    # Display results
    print(f"\nSuggested wikilinks:")
    for link in result["suggested_links"]:
        score = link.get("score", "n/a")
        source = link.get("source", "retrieval")
        print(f"  [[{link['title']}]] (score: {score}, source: {source})")

    print(f"\nSuggested tags (retrieval):")
    for tag in result["suggested_tags"]:
        score = tag.get("score", "n/a")
        source = tag.get("source", "retrieval")
        print(f"  [[{tag['title']}]] (score: {score}, source: {source})")

    if "llm_tags" in result:
        print(f"\nSuggested tags (LLM):")
        print(f"  Existing: {result['llm_tags'].get('existing_tags', [])}")
        print(f"  New: {result['llm_tags'].get('new_tags', [])}")
        print(f"  Reasoning: {result['llm_tags'].get('reasoning', '')}")

    # Write to Obsidian
    if "llm_tags" in result:
        final_tags = result["llm_tags"].get("existing_tags", []) + result["llm_tags"].get("new_tags", [])
    else:
        final_tags = retrieval_tags

    references = [link["title"] for link in result["suggested_links"] if link.get("source") == "retrieval"]
    title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    note_path = write_note(
        title=title,
        content=input_text,
        tags=final_tags,
        references=references,
    )
    print(f"\nNote saved to: {note_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <pdf_path>    Process a single PDF")
        print("  python main.py --watch       Watch folder for new PDFs")
        sys.exit(1)

    # Initialize once
    docs, index, tag_set, tag_context, reranker = setup()

    if sys.argv[1] == "--watch":
        # Watch mode: poll for new PDFs
        watch_loop(lambda pdf: process_pdf(pdf, docs, index, tag_set, tag_context, reranker))
    else:
        # Single PDF mode
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"Error: {pdf_path} not found")
            sys.exit(1)
        process_pdf(pdf_path, docs, index, tag_set, tag_context, reranker)


if __name__ == "__main__":
    main()
