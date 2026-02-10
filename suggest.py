"""
Suggestion engine for wikilinks and tags

Provides two layers of tag suggestion:
1. Retrieval-based: Fast vector similarity to find related notes and extract
   tags from their metadata and graph connections.
2. LLM fallback: When retrieval returns too few tags or low confidence scores,
   uses GPT to intelligently select from all available tags.
"""
import json
import openai


def suggest_links_and_tags(
    text: str,
    index,
    tag_set: set[str],
    docs: list,
    reranker=None,
    top_k: int = 10,
) -> dict:
    """
    Suggest wikilinks and tags using vector retrieval (Layer 1).

    Retrieves similar notes from the index, optionally reranks them,
    then extracts suggestions from:
    - Direct retrieval: Notes semantically similar to the input text
    - Graph expansion: Wikilinks and backlinks from retrieved notes

    Args:
        text: The input text to find suggestions for.
        index: The VectorStoreIndex to query.
        tag_set: Set of valid tag names from the vault's tags folder.
        docs: Original documents from ObsidianReader (for complete metadata).
        reranker: Optional SentenceTransformerRerank for better ranking.
        top_k: Number of candidates to retrieve (reranker reduces this further).

    Returns:
        Dict with 'suggested_links' and 'suggested_tags' lists.
    """
    # Build lookup from note_name -> full doc metadata
    # Merge wikilinks/backlinks from all chunks of the same note
    doc_metadata = {}
    for doc in docs:
        name = doc.metadata.get("note_name", "")
        if name:
            if name not in doc_metadata:
                doc_metadata[name] = {"wikilinks": set(), "backlinks": set()}
            doc_metadata[name]["wikilinks"].update(doc.metadata.get("wikilinks", []))
            doc_metadata[name]["backlinks"].update(doc.metadata.get("backlinks", []))

    # Convert sets back to lists
    for name in doc_metadata:
        doc_metadata[name]["wikilinks"] = list(doc_metadata[name]["wikilinks"])
        doc_metadata[name]["backlinks"] = list(doc_metadata[name]["backlinks"])

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(text)

    # Rerank if available (cross-encoder for better semantic matching)
    if reranker:
        results = reranker.postprocess_nodes(results, query_str=text)

    # Deduplicate by note name, keep best score
    # Use ORIGINAL doc metadata for wikilinks/backlinks
    seen = {}
    for node in results:
        name = node.metadata.get("note_name", "")
        if name and (name not in seen or node.score > seen[name]["score"]):
            orig = doc_metadata.get(name, {})
            seen[name] = {
                "title": name,
                "score": round(node.score, 4),
                "folder": node.metadata.get("folder_name", ""),
                "wikilinks": orig.get("wikilinks", []),
                "backlinks": orig.get("backlinks", []),
            }

    # Collect secondary links from graph (wikilinks and backlinks of retrieved notes)
    secondary_names = set()
    for info in seen.values():
        for wl in info["wikilinks"]:
            secondary_names.add(wl)
        for bl in info["backlinks"]:
            secondary_names.add(bl)
    secondary_names -= set(seen.keys())

    # Split into tags vs notes based on whether the name is in tag_set
    suggested_tags = []
    suggested_links = []

    for info in sorted(seen.values(), key=lambda x: x["score"], reverse=True):
        if info["title"] in tag_set:
            suggested_tags.append({
                "title": info["title"],
                "score": info["score"],
                "source": "retrieval",
            })
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


def suggest_tags_via_llm(
    note_text: str,
    all_tags: list[str],
    retrieval_tags: list[str],
    min_tags: int = 3,
    max_tags: int = 6,
) -> dict:
    """
    Use an LLM to select the best tags (Layer 2 fallback)

    Called when retrieval returns too few tags or low confidence scores.
    Uses GPT to intelligently select from all available tags and can
    propose new tags if no existing tag covers a key concept.

    Args:
        note_text: The content of the note to tag.
        all_tags: List of all available tags in the vault.
        retrieval_tags: Tags already suggested by retrieval (for context).
        min_tags: Minimum number of tags to suggest.
        max_tags: Maximum number of tags to suggest.

    Returns:
        Dict with 'existing_tags', 'new_tags', and 'reasoning' keys.
    """
    prompt = f"""You are a tagging system for a personal knowledge base.

Given the following note content and a list of all available tags,
select the most relevant tags for this note.

Note content:
{note_text[:3000]}

Available tags:
{json.dumps(sorted(all_tags))}

Tags already suggested by retrieval (may or may not be correct):
{json.dumps(retrieval_tags)}

Rules:
- Select {min_tags} to {max_tags} tags total.
- Choose from the available tags list whenever possible.
- Only propose a NEW tag if no existing tag covers the concept.
- New tags must use lowercase-with-hyphens format.
- Return valid JSON only, no other text.

Return format:
{{
    "existing_tags": ["tag1", "tag2"],
    "new_tags": ["tag3"],
    "reasoning": "brief explanation of choices"
}}"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)
