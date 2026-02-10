"""Entry point for the Obsidian RAG system."""
from config import VAULT_PATH, PERSIST_DIR, TOP_K, EMBEDDING_MODEL
from indexer import load_documents, build_or_load_index
from tags import load_tag_set
from suggest import suggest_links_and_tags


def main():
    # setup
    docs = load_documents(VAULT_PATH)
    index = build_or_load_index(docs, PERSIST_DIR, EMBEDDING_MODEL)
    tag_set = load_tag_set(VAULT_PATH)

    # test query, see if it works
    test_query = "convolutional neural networks"
    result = suggest_links_and_tags(test_query, index, tag_set, docs, top_k=TOP_K)

    print(f"\n{'='*50}")
    print(f"Query: '{test_query}'")
    print(f"{'='*50}")

    print("\n Suggested wikilinks:")
    for link in result["suggested_links"]:
        score = link.get("score", "n/a")
        source = link.get("source", "retrieval")
        print(f"  [[{link['title']}]] (score: {score}, source: {source})")

    print("\n Suggested tags:")
    for tag in result["suggested_tags"]:
        score = tag.get("score", "n/a")
        source = tag.get("source", "retrieval")
        print(f"  [[{tag['title']}]] (score: {score}, source: {source})")


if __name__ == "__main__":
    main()