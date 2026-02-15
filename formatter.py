"""Format classified OCR regions into clean Markdown."""
import openai


def regions_to_raw_markdown(regions: list[dict]) -> str:
    """Convert classified regions into rough Markdown."""
    parts = []
    for region in regions:
        if region["type"] == "text":
            parts.append(region["text"])
        elif region["type"] == "math":
            latex = region.get("latex", region["text"])
            parts.append(latex)
        elif region["type"] == "diagram":
            parts.append("<!-- diagram omitted -->")
    return "\n\n".join(parts)


def format_with_llm(raw_markdown: str) -> str:
    """Use LLM to clean up and structure the raw Markdown."""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Clean up and structure the following raw OCR output into
well-formatted Markdown for a personal knowledge base.

Rules:
- Fix obvious OCR errors and typos
- Add appropriate headings where topic shifts are apparent
- Preserve all LaTeX expressions exactly as they are (anything in $...$ or $$...$$)
- Replace <!-- diagram omitted --> with a placeholder like [Diagram: description if possible]
- Use clean Markdown formatting (headings, bullet points where appropriate)
- Do not add any information that isn't in the original text
- Do not remove any information

Raw OCR output:
{raw_markdown}

Return only the cleaned Markdown, no explanation."""
        }],
        temperature=0,
    )
    return response.choices[0].message.content
