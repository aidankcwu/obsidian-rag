"""OCR pipeline - converts PDF to images to text using Google Cloud Vision API"""
import io
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import vision


def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """Convert a PDF to a list of PIL images, one per page"""
    images = convert_from_path(str(pdf_path))
    print(f"Converted {len(images)} pages from {pdf_path.name}")
    return images


def ocr_image(image: Image.Image, client: vision.ImageAnnotatorClient) -> str:
    """Send a single image to Google Cloud Vision and return extracted text"""
    # Convert PIL image to bytes
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
    """Full pipeline"""
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