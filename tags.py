"""Load tag set from the Obsidian vault."""
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
