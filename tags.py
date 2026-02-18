"""Load tag set from the Obsidian vault"""
from pathlib import Path


def load_tag_set(vault_path: Path) -> set[str]:
    """
    Scan the tags folder and return all tag names.
    Tags in this vault are empty .md files in the '3 - Tags' folder.
    """
    tags_folder = vault_path / "3 - Tags"
    if not tags_folder.exists():
        print(f"Warning: Tags folder not found at {tags_folder}")
        return set()

    tags = {f.stem for f in tags_folder.glob("*.md")}
    print(f"Loaded {len(tags)} tags from vault")
    return tags


def build_tag_context(docs: list, tag_set: set[str]) -> dict[str, list[str]]:
    """
    Build a mapping of each tag to the note titles that reference it.

    Scans all documents' wikilinks and backlinks to find which notes
    use each tag, giving the LLM richer context for tag suggestion.

    Returns:
        Dict mapping tag name -> list of note titles that use it.
    """
    tag_to_notes: dict[str, set[str]] = {tag: set() for tag in tag_set}

    for doc in docs:
        note_name = doc.metadata.get("note_name", "")
        if not note_name:
            continue
        links = set(doc.metadata.get("wikilinks", [])) | set(doc.metadata.get("backlinks", []))
        for link in links:
            if link in tag_to_notes:
                tag_to_notes[link].add(note_name)

    # Convert sets to sorted lists
    return {tag: sorted(notes) for tag, notes in tag_to_notes.items() if notes}
