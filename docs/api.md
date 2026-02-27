# API Reference

The FastAPI server exposes four endpoints. Start it with:

```bash
python api.py
# or with auto-reload:
uvicorn api:app --reload
```

Interactive docs (Swagger UI) are available at `http://localhost:8000/docs` once the server is running.

---

## `GET /health`

Health check. Returns server readiness and basic stats.

**Response:**
```json
{
  "status": "ok",
  "index_loaded": true,
  "num_tags": 42,
  "vault_path": "/path/to/vault"
}
```

---

## `GET /tags`

Returns all available tags as a sorted list of strings.

**Response:**
```json
["algorithms", "calculus", "comp182", "linear-algebra", "math212"]
```

---

## `POST /suggest`

Suggest tags and wikilinks for a block of text. No OCR, no file writes.

**Request:**
```json
{
  "text": "The gradient descent algorithm minimizes the loss function...",
  "top_k": 10
}
```

**Response:**
```json
{
  "suggested_links": [
    {"title": "Gradient Descent", "score": 0.8231, "source": "retrieval"},
    {"title": "Loss Functions", "source": "graph"}
  ],
  "suggested_tags": [
    {"title": "machine-learning", "score": 0.7102, "source": "retrieval"}
  ],
  "llm_tags": null
}
```

`llm_tags` is non-null only when the LLM fallback was triggered (low retrieval confidence or fewer than 3 tags found).

---

## `POST /process`

Upload a PDF file. Runs the full pipeline: OCR → tag/link suggestion → note writing.

**Request:** multipart form with a `file` field (PDF only).

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@lecture_notes.pdf"
```

**Response:**
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
  "llm_tags": {
    "existing_tags": ["machine-learning"],
    "new_tags": [],
    "reasoning": "Content is clearly about ML optimization."
  },
  "note_path": "/path/to/vault/1 - Inbox/Lecture Notes.md"
}
```
