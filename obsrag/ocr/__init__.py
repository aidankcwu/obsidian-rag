"""OCR pipeline — converts PDF to images to text.

Re-exports the main OCR functions for convenient imports:
    from obsrag.ocr import ocr_pdf_with_llm
"""
from .vision import ocr_pdf_with_llm, pdf_to_images
from .google import ocr_pdf, ocr_pdf_structured
