"""Entry point for the Obsidian RAG system."""
from pathlib import Path
from config import VAULT_PATH, PERSIST_DIR, EMBEDDING_MODEL
from indexer import load_documents, build_or_load_index
from tags import load_tag_set
from suggest import suggest_links_and_tags, suggest_tags_via_llm
from ocr import ocr_pdf_structured
from formatter import regions_to_raw_markdown, format_with_llm
from write_to_obsidian import write_note
from llama_index.core.postprocessor import SentenceTransformerRerank

# thresholds for LLM fallback
MIN_TAGS_THRESHOLD = 3
MIN_CONFIDENCE_THRESHOLD = 0.4


def main():
    # setup
    docs = load_documents(VAULT_PATH)
    index = build_or_load_index(docs, PERSIST_DIR, EMBEDDING_MODEL)
    tag_set = load_tag_set(VAULT_PATH)

    # initialize reranker
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
    )

    # option 1 - structured ocr from pdf (with math/LaTeX detection)
    pdf_path = Path("test_note.pdf")
    if pdf_path.exists():
        print(f"Processing PDF: {pdf_path}")
        all_pages = ocr_pdf_structured(pdf_path)

        # Flatten all page regions and convert to raw markdown
        all_regions = [region for page in all_pages for region in page]
        raw_markdown = regions_to_raw_markdown(all_regions)
        print(f"\n--- Raw OCR ---\n{raw_markdown[:500]}...\n")

        # Clean up with LLM
        print("Formatting with LLM...")
        input_text = format_with_llm(raw_markdown)
        print(f"\n--- Formatted ---\n{input_text[:500]}...\n")

        input_source = f"OCR'd note: {pdf_path.name}"
    else:
        # Option 2: Test query (fallback if no PDF)
        input_text = "convolutional neural networks"
        input_source = f"Query: '{input_text}'"

    # Retrieval-based suggestions (retrieve 10, rerank to 5)
    result = suggest_links_and_tags(
        input_text,
        index,
        tag_set,
        docs,
        reranker=reranker,
        top_k=10,
    )
    retrieval_tags = [t["title"] for t in result["suggested_tags"]]

    # check confidence: best retrieval score
    top_score = result["suggested_links"][0]["score"] if result["suggested_links"] else 0

    # LLM fallback if not enough tags or low retrieval confidence
    if len(retrieval_tags) < MIN_TAGS_THRESHOLD or top_score < MIN_CONFIDENCE_THRESHOLD:
        print(f"[LLM fallback triggered: {len(retrieval_tags)} tags, top_score={top_score:.2f}]")
        llm_result = suggest_tags_via_llm(
            note_text=input_text,
            all_tags=sorted(tag_set),
            retrieval_tags=retrieval_tags,
        )
        result["llm_tags"] = llm_result

    # display results
    print(f"\n{'='*50}")
    print(f"Suggestions for: {input_source}")
    print(f"{'='*50}")

    print("\nSuggested wikilinks:")
    for link in result["suggested_links"]:
        score = link.get("score", "n/a")
        source = link.get("source", "retrieval")
        print(f"  [[{link['title']}]] (score: {score}, source: {source})")

    print("\nSuggested tags (retrieval):")
    for tag in result["suggested_tags"]:
        score = tag.get("score", "n/a")
        source = tag.get("source", "retrieval")
        print(f"  [[{tag['title']}]] (score: {score}, source: {source})")

    if "llm_tags" in result:
        print("\nSuggested tags (LLM):")
        print(f"  Existing: {result['llm_tags'].get('existing_tags', [])}")
        print(f"  New: {result['llm_tags'].get('new_tags', [])}")
        print(f"  Reasoning: {result['llm_tags'].get('reasoning', '')}")

    # --- Write to Obsidian ---
    # Collect final tags (prefer LLM if available, otherwise retrieval)
    if "llm_tags" in result:
        final_tags = result["llm_tags"].get("existing_tags", []) + result["llm_tags"].get("new_tags", [])
    else:
        final_tags = retrieval_tags

    # Collect wikilinks as references
    references = [link["title"] for link in result["suggested_links"] if link.get("source") == "retrieval"]

    # Derive title from PDF filename
    title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    note_path = write_note(
        title=title,
        content=input_text,
        tags=final_tags,
        references=references,
    )
    print(f"\nNote saved to: {note_path}")


if __name__ == "__main__":
    main()
