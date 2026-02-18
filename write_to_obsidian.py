"""Write formatted notes to the Obsidian vault inbox."""
from pathlib import Path
from datetime import datetime


def write_note(
    title: str,
    content: str,
    tags: list[str] = None,
    references: list[str] = None,
    inbox_path: Path = None,
    tag_style: str = "wikilink",
    template: str = None,
) -> Path:
    """
    Write a formatted note to the Obsidian vault's inbox folder.

    Args:
        title: The note title (also used as the filename).
        content: The main body text of the note.
        tags: List of tag names.
        references: List of reference note names.
        inbox_path: Path to the inbox folder.
        tag_style: "wikilink" formats as [[tag]], "hashtag" formats as #tag.
        template: Note template string with {date}, {time}, {title}, {content},
                  {tags}, {references} placeholders.

    Returns:
        Path to the created note file.
    """
    tags = tags or []
    references = references or []

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Format tags based on style
    if tag_style == "hashtag":
        tags_str = ", ".join(f"#{tag}" for tag in tags) if tags else ""
    else:
        tags_str = ", ".join(f"[[{tag}]]" for tag in tags) if tags else ""

    # Format references as wikilinks
    refs_str = "\n".join(f"- [[{ref}]]" for ref in references) if references else ""

    # Use default template if none provided
    if template is None:
        from config import get_config, DEFAULT_NOTE_TEMPLATE
        try:
            cfg = get_config()
            template = cfg.note_template
        except (FileNotFoundError, ValueError):
            template = DEFAULT_NOTE_TEMPLATE

    note = template.format(
        date=date_str,
        time=time_str,
        title=title,
        content=content,
        tags=tags_str,
        references=refs_str,
    )

    # Determine inbox path
    if inbox_path is None:
        from config import get_config
        cfg = get_config()
        inbox_path = cfg.inbox_path

    inbox_path.mkdir(parents=True, exist_ok=True)
    file_path = inbox_path / f"{title}.md"
    file_path.write_text(note)
    print(f"Note written to {file_path}")

    return file_path
