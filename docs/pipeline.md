# Pipeline Details

## Overview

```
PDF -> OpenAI vision OCR -> Retrieval + reranking -> Tag/link suggestions -> Obsidian note
```

## OCR

Each PDF page is rendered to an image with Poppler, sent to the configured OpenAI vision model, and transcribed into Markdown. The prompt asks the model to:

- transcribe handwriting as faithfully as possible,
- preserve math as LaTeX,
- insert `[Diagram: ...]` placeholders for drawings or graphs,
- avoid extra commentary.

If diagram placeholders are present, the original page image is saved into the vault attachments folder and embedded in a collapsed callout.

## Retrieval and Suggestions

The vault is indexed with LlamaIndex using OpenAI embeddings. For each OCR result:

1. Similar notes are retrieved from the vector index.
2. A cross-encoder reranks the candidates.
3. Retrieved notes are split into links vs tags based on the current vault taxonomy.
4. If retrieval confidence is weak, an LLM fallback chooses tags from the available tag set.

## Note Writing

The note is rendered through the configured template and written to the inbox folder. Existing filenames are not overwritten; if a collision occurs, a timestamp suffix is appended.

## Incremental State

- `obsrag build` creates a fresh index.
- `obsrag sync` reindexes only changed, new, or deleted notes based on a manifest.
- `obsrag watch` tracks PDFs by file fingerprint, so modified files can be reprocessed.
