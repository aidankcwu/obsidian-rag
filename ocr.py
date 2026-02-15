"""OCR pipeline - converts PDF to images to text.

Supports three modes:
- Simple: PDF → plain text via Google Cloud Vision (ocr_pdf)
- Structured: PDF → classified regions with Pix2Tex LaTeX support (ocr_pdf_structured)
- LLM Vision: PDF → clean Markdown via GPT-4o-mini vision API (ocr_pdf_with_llm)
"""
import io
import base64
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import vision
from pix2tex.cli import LatexOCR
from classifier import classify_page
import openai


# Lazy-loaded LaTeX model (heavy, only load when needed)
latex_model = None


def get_latex_model():
    global latex_model
    if latex_model is None:
        latex_model = LatexOCR()
    return latex_model


def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """Convert a PDF to a list of PIL images, one per page."""
    images = convert_from_path(str(pdf_path))
    print(f"Converted {len(images)} pages from {pdf_path.name}")
    return images


def ocr_image(image: Image.Image, client: vision.ImageAnnotatorClient) -> str:
    """Send a single image to Google Cloud Vision and return extracted text."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()

    gcp_image = vision.Image(content=content)
    response = client.document_text_detection(image=gcp_image)

    if response.error.message:
        raise Exception(f"Vision API error {response.error.message}")

    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""


def ocr_pdf(pdf_path: Path) -> str:
    """Simple pipeline: PDF → plain text."""
    client = vision.ImageAnnotatorClient()
    images = pdf_to_images(pdf_path)

    all_text = []
    for i, image in enumerate(images):
        print(f"OCR processing page {i + 1}/{len(images)}")
        text = ocr_image(image, client)
        all_text.append(text)

    combined = "\n\n".join(all_text)
    print(f"Extracted {len(combined)} characters total")
    return combined


def ocr_math_region(page_image: Image.Image, bounds: dict) -> str:
    """Crop a math region from the page and convert to LaTeX."""
    padding = 15
    crop_box = (
        max(0, bounds["x_min"] - padding),
        max(0, bounds["y_min"] - padding),
        min(page_image.width, bounds["x_max"] + padding),
        min(page_image.height, bounds["y_max"] + padding),
    )
    cropped = page_image.crop(crop_box)

    model = get_latex_model()
    latex = model(cropped)
    return f"${latex}$"


def ocr_page_structured(image: Image.Image, client: vision.ImageAnnotatorClient) -> list[dict]:
    """
    OCR a single page and return classified regions
    (text, math, diagram) in reading order.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()

    gcp_image = vision.Image(content=content)
    response = client.document_text_detection(image=gcp_image)

    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")

    if not response.full_text_annotation or not response.full_text_annotation.pages:
        return []

    page = response.full_text_annotation.pages[0]
    regions = classify_page(page, image.height, image.width)

    # For math regions, run Pix2Tex
    for region in regions:
        if region["type"] == "math":
            try:
                region["latex"] = ocr_math_region(image, region["bounds"])
            except Exception as e:
                print(f"Pix2Tex failed for region: {e}")
                region["latex"] = f"<!-- math OCR failed: {region['text']} -->"

    return regions


def ocr_pdf_structured(pdf_path: Path) -> list[list[dict]]:
    """Full structured pipeline: PDF → images → classified regions per page."""
    client = vision.ImageAnnotatorClient()
    images = pdf_to_images(pdf_path)

    all_pages = []
    for i, image in enumerate(images):
        print(f"Processing page {i + 1}/{len(images)}...")
        regions = ocr_page_structured(image, client)
        all_pages.append(regions)

    return all_pages


# --- LLM Vision pipeline ---

def ocr_page_with_llm(page_image: Image.Image) -> str:
    """
    Send a page image to GPT-4o-mini's vision API and get back
    clean Markdown with LaTeX math and diagram placeholders.
    """
    # Convert PIL image to base64
    buffer = io.BytesIO()
    page_image.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """You are an OCR transcription tool. Transcribe the handwritten
content in this image into clean Markdown. You MUST transcribe
whatever is written - do not refuse or say you cannot read it.

Rules:
- Transcribe all handwritten text as accurately as possible
- Convert any mathematical expressions to LaTeX (use $...$ for inline, $$...$$ for display)
- If there are diagrams or drawings, insert a placeholder like [Diagram: brief description]
- Use appropriate Markdown formatting (headings, bullet points) where apparent
- Do not add any information that isn't in the image
- Do not wrap the output in a code fence or ```markdown``` block
- Do not include any preamble, explanation, or apology - just the transcribed content""",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                },
            ],
        }],
        temperature=0,
    )

    result = response.choices[0].message.content

    # Strip markdown code fences if the model wrapped the output anyway
    if result.startswith("```markdown"):
        result = result[len("```markdown"):].strip()
    if result.startswith("```"):
        result = result[3:].strip()
    if result.endswith("```"):
        result = result[:-3].strip()

    # Strip refusal preambles (e.g. "I'm unable to assist with that.")
    refusal_prefixes = [
        "I'm unable to",
        "I cannot",
        "I'm sorry",
        "Sorry,",
    ]
    lines = result.split("\n")
    while lines and any(lines[0].strip().startswith(p) for p in refusal_prefixes):
        lines.pop(0)
    result = "\n".join(lines).strip()

    return result


def ocr_pdf_with_llm(pdf_path: Path) -> str:
    """LLM vision pipeline: PDF → images → clean Markdown via GPT-4o-mini."""
    images = pdf_to_images(pdf_path)

    all_text = []
    for i, image in enumerate(images):
        print(f"OCR (LLM vision) page {i + 1}/{len(images)}...")
        text = ocr_page_with_llm(image)
        all_text.append(text)

    combined = "\n\n".join(all_text)
    print(f"Extracted {len(combined)} characters total")
    return combined
