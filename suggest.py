"""RAG-based suggestion engine for wikilinks and tags."""


def suggest_links_and_tags(
    text: str,
    index,
    tag_set: set[str],
    top_k: int = 10,
) -> dict:
    """
    Given input text, retrieve similar notes and return
    suggested wikilinks and tags, separated by type.

    Tags are identified by checking if a note name exists
    in the vault's '3 - Tags' folder
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(text)

    # Debug: check what metadata the retrieved nodes have
    print("\n[DEBUG] Retrieved node metadata:")
    for i, node in enumerate(results[:3]):
        print(f"  Node {i}: {node.metadata}")

    # Deduplicate by note name, keep best score
    seen = {}
    for node in results:
        name = node.metadata.get("note_name", "")
        if name and (name not in seen or node.score > seen[name]["score"]):
            seen[name] = {
                "title": name,
                "score": round(node.score, 4),
                "folder": node.metadata.get("folder_name", ""),
                "wikilinks": node.metadata.get("wikilinks", []),
                "backlinks": node.metadata.get("backlinks", []),
            }

    #Collect secondary links from graph
    secondary_names = set()
    for info in seen.values():
        for wl in info["wikilinks"]:
            secondary_names.add(wl)
        for bl in info["backlinks"]:
            secondary_names.add(bl)
    secondary_names -= set(seen.keys())

    # Split into tags vs notes
    suggested_tags = []
    suggested_links = []

    for info in sorted(seen.values(), key=lambda x: x["score"], reverse=True):
        if info["title"] in tag_set:
            suggested_tags.append({"title": info["title"], "score": info["score"], "source": "retrieval"})
        else:
            suggested_links.append(info)

    for name in secondary_names:
        entry = {"title": name, "source": "graph"}
        if name in tag_set:
            suggested_tags.append(entry)
        else:
            suggested_links.append(entry)

    return {
        "suggested_links": suggested_links,
        "suggested_tags": suggested_tags,
    }
