"""Load tag set from the Obsidian vault."""
import re
from pathlib import Path


def load_tag_set(vault_path: Path, style: str = "wikilink", tags_folder_name: str = "3 - Tags") -> set[str]:
    """
    Load all tags from the vault.

    Args:
        vault_path: Path to the Obsidian vault root.
        style: "wikilink" scans the tags folder for .md files,
               "hashtag" scans all notes for #tag patterns.
        tags_folder_name: Name of the tags folder (wikilink style only).

    Returns:
        Set of tag name strings.
    """
    if style == "hashtag":
        tags = _scan_hashtags(vault_path)
    else:
        tags = _scan_wikilink_tags(vault_path, tags_folder_name)

    print(f"Loaded {len(tags)} tags from vault ({style} style)")
    return tags


def _scan_wikilink_tags(vault_path: Path, tags_folder_name: str) -> set[str]:
    """Scan the tags folder for .md files — each filename is a tag."""
    tags_folder = vault_path / tags_folder_name
    if not tags_folder.exists():
        print(f"Warning: Tags folder not found at {tags_folder}")
        return set()
    return {f.stem for f in tags_folder.glob("*.md")}


def _scan_hashtags(vault_path: Path) -> set[str]:
    """Scan all vault notes for #hashtag patterns."""
    # Regex: # preceded by whitespace or line start, followed by a letter,
    # then letters/digits/hyphens/underscores. Matches Obsidian's tag format.
    tag_pattern = re.compile(r'(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]*)')
    # Pattern to strip fenced code blocks before scanning
    code_fence_pattern = re.compile(r'```.*?```', re.DOTALL)

    tags = set()
    for md_file in vault_path.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # Strip code blocks so we don't pick up # in code
        text = code_fence_pattern.sub("", text)
        tags.update(tag_pattern.findall(text))

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
