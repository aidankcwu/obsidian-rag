"""Entry point for the Obsidian RAG system."""
from config import VAULT_PATH, PERSIST_DIR, TOP_K, EMBEDDING_MODEL
from indexer import load_documents, build_or_load_index
from tags import load_tag_set
from suggest import suggest_links_and_tags, suggest_tags_via_llm

# Thresholds for LLM fallback
MIN_TAGS_THRESHOLD = 3
MIN_CONFIDENCE_THRESHOLD = 0.4


def main():
    # Setup
    docs = load_documents(VAULT_PATH)
    index = build_or_load_index(docs, PERSIST_DIR, EMBEDDING_MODEL)
    tag_set = load_tag_set(VAULT_PATH)

    # Test query
    test_query = "convolutional neural networks"

    # Layer 1: Retrieval-based suggestions
    result = suggest_links_and_tags(test_query, index, tag_set, docs, top_k=TOP_K)
    retrieval_tags = [t["title"] for t in result["suggested_tags"]]

    # Check confidence: best retrieval score
    top_score = result["suggested_links"][0]["score"] if result["suggested_links"] else 0

    # Layer 2: LLM fallback if not enough tags OR low retrieval confidence
    if len(retrieval_tags) < MIN_TAGS_THRESHOLD or top_score < MIN_CONFIDENCE_THRESHOLD:
        print(f"[LLM fallback triggered: {len(retrieval_tags)} tags, top_score={top_score:.2f}]")
        llm_result = suggest_tags_via_llm(
            note_text=test_query,
            all_tags=sorted(tag_set),
            retrieval_tags=retrieval_tags,
        )
        result["llm_tags"] = llm_result

    # Display results
    print(f"\n{'='*50}")
    print(f"Query: '{test_query}'")
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


if __name__ == "__main__":
    main()
