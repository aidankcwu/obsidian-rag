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


def _crop_diagram(page_image: Image.Image, x_min: int, y_min: int, x_max: int, y_max: int, padding: int = 15) -> Image.Image:
    """Crop a diagram region from a page image with padding, clamped to image bounds."""
    w, h = page_image.size
    left = max(0, x_min - padding)
    top = max(0, y_min - padding)
    right = min(w, x_max + padding)
    bottom = min(h, y_max + padding)
    return page_image.crop((left, top, right, bottom))


def _embed_diagrams(
    content: str,
    title: str,
    page_images: list[Image.Image],
    page_offsets: list[tuple[int, int]],
    attachments_path: Path,
) -> str:
    """Replace diagram markers with embedded images saved to the attachments folder."""
    attachments_path.mkdir(parents=True, exist_ok=True)

    # Sanitize title for use in filenames
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

    diagram_count = 0

    # Pattern for coordinate-based markers: [DIAGRAM x_min y_min x_max y_max]
    coord_pattern = re.compile(r'\[DIAGRAM\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]')
    # Fallback pattern for description-based markers: [Diagram: description]
    fallback_pattern = re.compile(r'\[Diagram:\s*([^\]]*)\]')

    def replace_coord_match(match):
        nonlocal diagram_count
        diagram_count += 1
        x_min, y_min, x_max, y_max = (int(v) for v in match.groups())
        page_idx = _find_page_for_position(match.start(), page_offsets)
        page_img = page_images[page_idx]

        cropped = _crop_diagram(page_img, x_min, y_min, x_max, y_max)
        img_name = f"{safe_title}_p{page_idx + 1}_d{diagram_count}.png"
        img_path = attachments_path / img_name
        cropped.save(img_path, format="PNG")
        print(f"  Saved diagram: {img_path}")
        return f"![[{img_name}]]"

    def replace_fallback_match(match):
        nonlocal diagram_count
        diagram_count += 1
        page_idx = _find_page_for_position(match.start(), page_offsets)
        page_img = page_images[page_idx]

        # Save full page image as fallback
        img_name = f"{safe_title}_p{page_idx + 1}_d{diagram_count}.png"
        img_path = attachments_path / img_name
        page_img.save(img_path, format="PNG")
        print(f"  Saved diagram (full page fallback): {img_path}")
        return f"![[{img_name}]]"

    # First pass: replace coordinate-based markers
    content = coord_pattern.sub(replace_coord_match, content)
    # Second pass: replace any remaining description-based markers
    content = fallback_pattern.sub(replace_fallback_match, content)

    return content
