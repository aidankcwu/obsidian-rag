# API Reference

The API is optional. Install it with:

```bash
pip install .[api]
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API uses the same config and validation rules as the CLI. If setup is incomplete, startup or request handling will fail with an actionable error.

## `GET /health`

Returns readiness and basic config state.

```json
{
  "status": "ok",
  "index_loaded": true,
  "num_tags": 42,
  "vault_path": "/path/to/vault"
}
```

## `GET /tags`

Returns all available tags as a sorted list.

```json
["algorithms", "calculus", "linear-algebra"]
```

## `POST /suggest`

Suggest links and tags for plain text without OCR or file writes.

```json
{
  "text": "The gradient descent algorithm minimizes the loss function...",
  "top_k": 10
}
```

## `POST /process`

Upload a PDF and run the full OCR-to-note pipeline.

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@lecture_notes.pdf"
```

Example response:

```json
{
  "title": "Lecture Notes",
  "ocr_text": "# Gradient Descent\n\nThe update rule is...",
  "suggested_links": [
    {"title": "Gradient Descent", "score": 0.8231, "source": "retrieval"}
  ],
  "suggested_tags": [
    {"title": "machine-learning", "score": 0.7102, "source": "retrieval"}
  ],
  "llm_tags": null,
  "note_path": "/path/to/vault/1 - Inbox/Lecture Notes.md"
}
```
