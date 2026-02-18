"""LLM vision OCR — converts PDF pages to clean Markdown via GPT-4o-mini."""
import io
import base64
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
import openai


def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """Convert a PDF to a list of PIL images, one per page."""
    images = convert_from_path(str(pdf_path))
    print(f"Converted {len(images)} pages from {pdf_path.name}")
    return images


def ocr_page_with_llm(page_image: Image.Image, model: str = "gpt-4o-mini") -> str:
    """
    Send a page image to an LLM vision API and get back
    clean Markdown with LaTeX math and diagram placeholders.
    """
    # Convert PIL image to base64
    buffer = io.BytesIO()
    page_image.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = openai.chat.completions.create(
        model=model,
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


def ocr_pdf_with_llm(pdf_path: Path, model: str = "gpt-4o-mini") -> str:
    """LLM vision pipeline: PDF → images → clean Markdown via LLM vision API."""
    images = pdf_to_images(pdf_path)

    all_text = []
    for i, image in enumerate(images):
        print(f"OCR (LLM vision) page {i + 1}/{len(images)}...")
        text = ocr_page_with_llm(image, model=model)
        all_text.append(text)

    combined = "\n\n".join(all_text)
    print(f"Extracted {len(combined)} characters total")
    return combined
