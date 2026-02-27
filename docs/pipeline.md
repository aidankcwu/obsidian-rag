# Pipeline Details

## Overview

```
PDF ──► OCR ──► Vector Retrieval + Reranking ──► Tag/Link Suggestions ──► Obsidian Note
         │              │                                │                       │
    GPT-4o-mini    LlamaIndex +                   LLM fallback if         Incremental
    vision API     cross-encoder                  retrieval is low        index update
                   reranker                       confidence
```

---

## OCR Stage

The default OCR mode (`openai_vision`) sends each page of the PDF as a base64-encoded image to GPT-4o-mini's vision API with a prompt that:

- Transcribes handwritten text as accurately as possible
- Converts math to LaTeX (`$...$` inline, `$$...$$` display)
- Inserts `[Diagram: brief description]` placeholders for diagrams, drawings, or graphs
- Applies Markdown formatting (headings, bullets) where apparent

Post-processing strips any code fences or refusal preambles the model may add.

After OCR, `[Diagram: ...]` markers are resolved: the full page image is saved as a PNG to the vault's attachments folder, and each marker is replaced with a collapsed Obsidian callout:

```
> [!info]- Original page (diagram: description)
> ![[Note_Title_page_N.png]]
```

If a page contains multiple diagram markers, the page image is saved once and all markers on that page point to the same file.

Two alternative OCR modes are available in `obsrag/ocr/google.py` for reference:
- **`ocr_pdf`** — Simple plain text via Google Cloud Vision
- **`ocr_pdf_structured`** — Google Cloud Vision + region classification + Pix2Tex for LaTeX math

---

## Layer 1: Retrieval-Based Suggestions

1. **Vector search** — Retrieves the top-k (default 10) most similar notes from the index.
2. **Reranking** — A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks and filters to the top 5. Results with score <= 0 are discarded.
3. **Metadata merging** — Wikilinks and backlinks from all chunks of the same note are merged (ObsidianReader pre-chunks documents).
4. **Graph expansion** — Follows wikilinks and backlinks from retrieved notes to discover secondary suggestions.
5. **Tag separation** — Names that match entries in the tags folder become tag suggestions; all others become link suggestions.

---

## Layer 2: LLM Fallback

Triggered when retrieval returns fewer than 3 tags or the top retrieval score is below 0.4. Sends GPT-4o-mini:

- The note content (first 3000 characters)
- The source PDF filename (helps identify course context)
- All available tags with context showing which notes use each tag
- Tags already suggested by retrieval

The prompt instructs the LLM to select 3–6 tags, prefer existing tags over new ones, and assign course-code tags (e.g. `comp182`, `math212`) based on filename context rather than content similarity — typically only one course tag per note.

---

## Note Writing

Notes are written to the vault's inbox folder using a customizable template. Diagram images are saved to the attachments folder before the note is written, so `[Diagram: ...]` markers in the content are already resolved to callout blocks by the time the file is created.

The default template:

```
{date} {time}

Status: #review

Tags: {tags}

# {title}

{content}

## References
{references}
```

Tags are formatted as `[[tag]]` (wikilink style) or `#tag` (hashtag style) based on your config. References are rendered as bulleted wikilinks (`- [[Note Name]]`).

---

## Incremental Index Updates

The index is kept current through two complementary mechanisms that share a manifest file at `.obsrag/manifest.json`.

### Startup sync (`sync_index`)

Called automatically at every startup (via `setup()` in the CLI/watcher, or FastAPI `startup`). Scans the vault and diffs each `.md` file's modification time against the manifest:

- **Changed file** — deletes all old index chunks for that file (`delete_ref_doc`), re-embeds the updated content, updates the manifest entry
- **New file** — embeds and inserts it, adds a manifest entry
- **Deleted file** — removes all index chunks, removes the manifest entry
- **Unchanged file** — skipped entirely

On the very first run (no manifest yet), it bootstraps the manifest from the current vault state and returns without modifying the index, assuming the index is already accurate.

### Per-PDF update (`add_note_to_index`)

After each note is written by `process_pdf` or the `/process` API endpoint:

1. The new note is chunked with the same `SentenceSplitter` settings and inserted into the index.
2. The manifest is updated with the note's current mtime and doc_id.
3. The in-memory `docs` list and tag set are refreshed so subsequent PDFs in the same session can retrieve this note.

### Interaction between the two

Because both mechanisms write to the same manifest, they stay consistent:
- Notes written by `add_note_to_index` are already in the manifest when `sync_index` next runs
- If the user edits a pipeline-created note in Obsidian before the next startup, `sync_index` detects the mtime change and re-indexes it

Run `python cli.py build` to delete the index and manifest and start completely fresh.
