"""Write formatted notes to the Obsidian vault inbox."""
from pathlib import Path
from datetime import datetime
from config import VAULT_PATH

INBOX_FOLDER = VAULT_PATH / "1 - Inbox"


def write_note(
    title: str,
    content: str,
    tags: list[str] = None,
    references: list[str] = None,
) -> Path:
    """
    Write a formatted note to the Obsidian vault's inbox folder.

    Args:
        title: The note title (also used as the filename).
        content: The main body text of the note.
        tags: List of tag names to add as [[tag]] wikilinks.
        references: List of reference note names to add as [[note]] wikilinks.

    Returns:
        Path to the created note file.
    """
    tags = tags or []
    references = references or []

    # Format timestamp
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Format tags as wikilinks
    tags_str = ", ".join(f"[[{tag}]]" for tag in tags) if tags else ""

    # Format references as wikilinks
    refs_str = "\n".join(f"- [[{ref}]]" for ref in references) if references else ""

    # Build the note
    note = f"""{date_str} {time_str}

Status: #review

Tags: {tags_str}

# {title}

{content}

## References
{refs_str}
"""

    # Write to inbox
    INBOX_FOLDER.mkdir(parents=True, exist_ok=True)
    file_path = INBOX_FOLDER / f"{title}.md"
    file_path.write_text(note)
    print(f"Note written to {file_path}")

    return file_path
