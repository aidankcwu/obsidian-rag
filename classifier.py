"""Classify OCR regions into text, math, and diagram."""
import re
from google.cloud import vision


# Characters that suggest math content
MATH_INDICATORS = set("=±√∫∑∏∂∆≈≠≤≥αβγδεθλμπσφψω")
MATH_PATTERNS = re.compile(r"[a-z]\s*[=<>]\s*[a-z0-9]|d[xy]|/d[xy]|\^[0-9]", re.IGNORECASE)


def classify_block(block, page_height, page_width):
    """
    Classify a single block from Vision API as text, math, or diagram.

    Returns: {"type": "text"|"math"|"diagram", "text": str, "bounds": {...}}
    """
    paragraphs_text = []
    total_confidence = 0
    symbol_count = 0
    char_count = 0

    for paragraph in block.paragraphs:
        for word in paragraph.words:
            word_text = "".join(s.text for s in word.symbols)
            word_confidence = word.confidence if hasattr(word, "confidence") else 1.0
            paragraphs_text.append(word_text)
            total_confidence += word_confidence
            char_count += len(word_text)
            symbol_count += sum(1 for c in word_text if c in MATH_INDICATORS)

    full_text = " ".join(paragraphs_text)
    avg_confidence = total_confidence / max(len(paragraphs_text), 1)

    # Get bounding box
    vertices = block.bounding_box.vertices
    bounds = {
        "x_min": min(v.x for v in vertices),
        "y_min": min(v.y for v in vertices),
        "x_max": max(v.x for v in vertices),
        "y_max": max(v.y for v in vertices),
    }

    # Classification heuristics
    has_math_symbols = symbol_count > 0
    has_math_pattern = bool(MATH_PATTERNS.search(full_text))
    low_confidence = avg_confidence < 0.75
    short_garbled = len(full_text) < 10 and low_confidence

    if has_math_symbols or (has_math_pattern and low_confidence) or short_garbled:
        return {"type": "math", "text": full_text, "bounds": bounds, "confidence": avg_confidence}

    return {"type": "text", "text": full_text, "bounds": bounds, "confidence": avg_confidence}


def classify_page(page_annotation, page_height, page_width):
    """
    Classify all blocks on a page.
    Returns list of classified regions sorted by vertical position.
    """
    regions = []
    for block in page_annotation.blocks:
        if block.block_type == vision.Block.BlockType.TEXT:
            region = classify_block(block, page_height, page_width)
            regions.append(region)

    # Sort top to bottom
    regions.sort(key=lambda r: r["bounds"]["y_min"])
    return regions
