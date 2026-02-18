# Obsidian RAG

An automated pipeline that OCRs handwritten PDF notes (e.g. from GoodNotes), suggests relevant tags and wikilinks using retrieval-augmented generation, and writes formatted Markdown notes directly into your Obsidian vault.

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [API Keys](#api-keys)
  - [Building the Index](#building-the-index)
- [Usage](#usage)
  - [CLI](#cli)
  - [REST API](#rest-api)
  - [File Watcher](#file-watcher)
- [Pipeline Details](#pipeline-details)
  - [OCR Stage](#ocr-stage)
  - [Layer 1: Retrieval-Based Suggestions](#layer-1-retrieval-based-suggestions)
  - [Layer 2: LLM Fallback](#layer-2-llm-fallback)
  - [Note Writing](#note-writing)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)

---

## How It Works

```
PDF ──► OCR ──► Vector Retrieval + Reranking ──► Tag/Link Suggestions ──► Obsidian Note
         │              │                                │
    GPT-4o-mini    LlamaIndex +                   LLM fallback if
    vision API     cross-encoder                  retrieval is low
                   reranker                       confidence
```

1. **OCR** — Converts each page of a handwritten PDF to clean Markdown with LaTeX math support using GPT-4o-mini's vision API.
2. **Retrieval** — Queries a vector index of your entire Obsidian vault to find semantically similar notes. A cross-encoder reranker refines the results.
3. **Tag Suggestion** — Extracts tags and wikilinks from retrieved notes and their graph neighbors. Falls back to an LLM when retrieval confidence is low.
4. **Write** — Formats the OCR text with suggested tags and references into a note and saves it to your vault's inbox folder.

---

## Project Structure

```
obsidian-rag/
├── cli.py                       # Click CLI — init, build, process, watch
├── api.py                       # FastAPI REST API server
├── obsrag/                      # Core package
│   ├── config.py                # YAML config loader with lazy singleton
│   ├── pipeline.py              # setup() and process_pdf() orchestration
│   ├── writer.py                # Templated note writer
│   ├── watcher.py               # Folder polling for automatic processing
│   ├── ocr/                     # OCR subpackage
│   │   ├── vision.py            # LLM vision OCR (GPT-4o-mini) — primary pipeline
│   │   ├── google.py            # Google Cloud Vision + Pix2Tex structured OCR
│   │   ├── classifier.py        # Region classification (text/math/diagram)
│   │   └── formatter.py         # Markdown formatting with LLM cleanup
│   └── rag/                     # RAG subpackage
│       ├── indexer.py            # Vector index build/load with LlamaIndex
│       ├── tags.py               # Tag loading (wikilink or hashtag style)
│       └── suggest.py            # Two-layer suggestion engine (retrieval + LLM)
├── .obsrag.yaml.example         # Example configuration file
├── .env                         # API keys (not tracked)
├── .obsrag.yaml                 # Your config (not tracked)
└── .obsrag/                     # Runtime artifacts (not tracked)
    ├── index/                   # Persisted vector store
    └── processed.json           # Watcher's processed file log
```

---

## Setup

### Prerequisites

- Python 3.10+
- [poppler](https://poppler.freedesktop.org/) (for `pdf2image`)
  ```bash
  # macOS
  brew install poppler

  # Ubuntu/Debian
  sudo apt-get install poppler-utils
  ```

### Installation

```bash
git clone https://github.com/yourusername/obsidian-rag.git
cd obsidian-rag
python -m venv .venv
source .venv/bin/activate

pip install llama-index llama-index-readers-obsidian llama-index-embeddings-openai
pip install openai python-dotenv
pip install pdf2image Pillow
pip install sentence-transformers
pip install fastapi uvicorn python-multipart
pip install click pyyaml
pip install google-cloud-vision pix2tex  # optional, for alternative OCR modes
```

### Configuration

Run the interactive setup:

```bash
python cli.py init
```

This prompts you for your vault path, watch folder, tag style, and OCR provider, then generates `.obsrag.yaml`. You can also copy the example and edit manually:

```bash
cp .obsrag.yaml.example .obsrag.yaml
```

The config file is searched for in the current working directory first, then `~/.obsrag.yaml`.

### API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-proj-...
```

If using Google Cloud Vision OCR (optional):

```env
GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json
```

### Building the Index

Before processing any PDFs, build the vector index from your vault:

```bash
python cli.py build
```

This reads every note in your vault, chunks them into 512-token segments with 50-token overlap, embeds them with OpenAI's `text-embedding-3-small`, and persists the index to `.obsrag/index/`. Rebuild after significant vault changes with the same command.

---

## Usage

### CLI

```bash
python cli.py <command>
```

| Command | Description |
|---------|-------------|
| `init` | Interactive setup — generates `.obsrag.yaml` |
| `build` | Build or rebuild the vector index (deletes existing index first) |
| `process <pdf>` | Process a single PDF through the full pipeline |
| `watch` | Poll the watch folder for new PDFs and process them automatically |

Examples:

```bash
# First-time setup
python cli.py init
python cli.py build

# Process a single PDF
python cli.py process ~/Downloads/lecture_notes.pdf

# Auto-process new PDFs from a folder
python cli.py watch
```

### REST API

Start the server:

```bash
python api.py
# or with auto-reload for development:
uvicorn api:app --reload
```

The server initializes the index, tag set, and reranker once at startup, then exposes endpoints on `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

See [API Reference](#api-reference) below for endpoint details.

### File Watcher

The watcher polls a configured folder (e.g. a GoodNotes export directory) for new PDFs:

```bash
python cli.py watch
```

- Polls every 30 seconds (configurable via `watcher.poll_interval`)
- Tracks processed files in `.obsrag/processed.json` to avoid reprocessing
- Runs until interrupted with `Ctrl+C`

---

## Pipeline Details

### OCR Stage

The default OCR mode (`openai_vision`) sends each page of the PDF as a base64-encoded image to GPT-4o-mini's vision API with a prompt that:

- Transcribes handwritten text as accurately as possible
- Converts math to LaTeX (`$...$` inline, `$$...$$` display)
- Inserts `[Diagram: description]` placeholders for drawings
- Applies Markdown formatting (headings, bullets) where apparent

Post-processing strips any code fences or refusal preambles the model may add.

Two alternative OCR modes are available in `obsrag/ocr/google.py` for reference:
- **`ocr_pdf`** — Simple plain text via Google Cloud Vision
- **`ocr_pdf_structured`** — Google Cloud Vision + region classification + Pix2Tex for LaTeX math

### Layer 1: Retrieval-Based Suggestions

1. **Vector search** — Retrieves the top-k (default 10) most similar notes from the index.
2. **Reranking** — A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks and filters to the top 5. Results with score <= 0 are discarded.
3. **Metadata merging** — Wikilinks and backlinks from all chunks of the same note are merged (ObsidianReader pre-chunks documents).
4. **Graph expansion** — Follows wikilinks and backlinks from retrieved notes to discover secondary suggestions.
5. **Tag separation** — Names that match entries in the tags folder become tag suggestions; all others become link suggestions.

### Layer 2: LLM Fallback

Triggered when retrieval returns fewer than 3 tags or the top retrieval score is below 0.4. Sends GPT-4o-mini:

- The note content (first 3000 characters)
- The source PDF filename (helps identify course context)
- All available tags with context showing which notes use each tag
- Tags already suggested by retrieval

The prompt instructs the LLM to select 3–6 tags, prefer existing tags over new ones, and assign course-code tags (e.g. `comp182`, `math212`) based on filename context rather than content similarity — typically only one course tag per note.

### Note Writing

Notes are written to the vault's inbox folder using a customizable template. The default template:

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

## Configuration Reference

All settings are in `.obsrag.yaml`. Only `vault_path` is required — everything else has defaults.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `vault_path` | path | **required** | Path to your Obsidian vault |
| `watch_folder` | path | — | Folder to poll for new PDFs |
| `folders.inbox` | string | `"1 - Inbox"` | Vault subfolder for new notes |
| `folders.tags` | string | `"3 - Tags"` | Vault subfolder containing tag files |
| `tags.style` | string | `"wikilink"` | `"wikilink"` (`.md` files in tags folder) or `"hashtag"` (scan notes for `#tag`) |
| `note_template` | string | see above | Note template with `{date}`, `{time}`, `{title}`, `{content}`, `{tags}`, `{references}` |
| `ocr.provider` | string | `"openai_vision"` | `"openai_vision"` or `"google_vision"` |
| `ocr.model` | string | `"gpt-4o-mini"` | LLM model for vision OCR |
| `embedding.model` | string | `"text-embedding-3-small"` | OpenAI embedding model |
| `embedding.chunk_size` | int | `512` | Token chunk size for indexing |
| `embedding.chunk_overlap` | int | `50` | Token overlap between chunks |
| `rag.top_k` | int | `10` | Candidates to retrieve before reranking |
| `rag.reranker_model` | string | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder reranker model |
| `rag.reranker_top_n` | int | `5` | Results to keep after reranking |
| `rag.min_tags_threshold` | int | `3` | LLM fallback triggers if fewer tags than this |
| `rag.min_confidence_threshold` | float | `0.4` | LLM fallback triggers if top score below this |
| `watcher.poll_interval` | int | `30` | Seconds between watch folder polls |

---

## API Reference

### `GET /health`

Health check. Returns server readiness and basic stats.

```json
{
  "status": "ok",
  "index_loaded": true,
  "num_tags": 42,
  "vault_path": "/path/to/vault"
}
```

### `GET /tags`

Returns all available tags as a sorted list of strings.

```json
["algorithms", "calculus", "comp182", "linear-algebra", "math212"]
```

### `POST /suggest`

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

### `POST /process`

Upload a PDF file. Runs the full pipeline: OCR, suggestion, note writing.

**Request:** multipart form with `file` field (PDF).

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@lecture_notes.pdf"
```

**Response:**
```json
{
  "title": "Lecture Notes",
  "ocr_text": "# Gradient Descent\n\nThe update rule is...",
  "suggested_links": [...],
  "suggested_tags": [...],
  "llm_tags": {"existing_tags": ["machine-learning"], "new_tags": [], "reasoning": "..."},
  "note_path": "/path/to/vault/1 - Inbox/Lecture Notes.md"
}
```
