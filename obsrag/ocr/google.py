"""Google Cloud Vision OCR — simple and structured pipelines."""
import io
from pathlib import Path
from PIL import Image
from google.cloud import vision
from pix2tex.cli import LatexOCR
from .classifier import classify_page
from .vision import pdf_to_images


# Lazy-loaded LaTeX model (heavy, only load when needed)
latex_model = None


def get_latex_model():
    global latex_model
    if latex_model is None:
        latex_model = LatexOCR()
    return latex_model


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
