"""Write formatted notes to the Obsidian vault inbox."""
import re
from pathlib import Path
from datetime import datetime

from PIL import Image


def write_note(
    title: str,
    content: str,
    tags: list[str] = None,
    references: list[str] = None,
    inbox_path: Path = None,
    tag_style: str = "wikilink",
    template: str = None,
    page_images: list[Image.Image] = None,
    page_offsets: list[tuple[int, int]] = None,
    attachments_path: Path = None,
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
        page_images: List of PIL images, one per PDF page.
        page_offsets: List of (start_char, end_char) tuples mapping each page
                      to its position in the content string.
        attachments_path: Path to the vault's attachments folder for diagram images.

    Returns:
        Path to the created note file.
    """
    tags = tags or []
    references = references or []

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Embed diagram images before templating
    if page_images and page_offsets and attachments_path:
        content = _embed_diagrams(content, title, page_images, page_offsets, attachments_path)

    # Format tags based on style
    if tag_style == "hashtag":
        tags_str = ", ".join(f"#{tag}" for tag in tags) if tags else ""
    else:
        tags_str = ", ".join(f"[[{tag}]]" for tag in tags) if tags else ""

    # Format references as wikilinks
    refs_str = "\n".join(f"- [[{ref}]]" for ref in references) if references else ""

    # Use default template if none provided
    if template is None:
        from obsrag.config import get_config, DEFAULT_NOTE_TEMPLATE
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
        from obsrag.config import get_config
        cfg = get_config()
        inbox_path = cfg.inbox_path

    inbox_path.mkdir(parents=True, exist_ok=True)
    file_path = inbox_path / f"{title}.md"
    file_path.write_text(note)
    print(f"Note written to {file_path}")

    return file_path


def _find_page_for_position(pos: int, page_offsets: list[tuple[int, int]]) -> int:
    """Map a character position to a page index using pre-computed offsets."""
    for i, (start, end) in enumerate(page_offsets):
        if start <= pos <= end:
            return i
    return len(page_offsets) - 1


def _embed_diagrams(
    content: str,
    title: str,
    page_images: list[Image.Image],
    page_offsets: list[tuple[int, int]],
    attachments_path: Path,
) -> str:
    """Replace [Diagram: ...] markers with collapsed Obsidian callouts containing page images."""
    attachments_path.mkdir(parents=True, exist_ok=True)

    # Sanitize title for use in filenames
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

    # Track which pages have already been saved to avoid duplicates
    saved_pages: dict[int, str] = {}  # page_idx -> img_name

    pattern = re.compile(r'\[Diagram:\s*([^\]]*)\]')

    def replace_match(match):
        description = match.group(1).strip() or "diagram"
        page_idx = _find_page_for_position(match.start(), page_offsets)

        # Save page image once per page
        if page_idx not in saved_pages:
            img_name = f"{safe_title}_page_{page_idx + 1}.png"
            img_path = attachments_path / img_name
            page_images[page_idx].save(img_path, format="PNG")
            saved_pages[page_idx] = img_name
            print(f"  Saved page image: {img_path}")

        img_name = saved_pages[page_idx]
        # Both lines need > prefix for Obsidian to treat the image as part of the callout
        return f"> [!info]- Original page (diagram: {description})\n> ![[{img_name}]]"

    return pattern.sub(replace_match, content)
