# Configuration Reference

All settings live in `.obsrag.yaml` in your project directory (or `~/.obsrag.yaml` as a fallback). Only `vault_path` is required — everything else has sensible defaults.

Run `python cli.py init` for an interactive setup that generates this file for you, or copy `.obsrag.yaml.example` and edit it manually.

---

## Full Example

```yaml
vault_path: ~/Documents/MyVault
watch_folder: ~/Downloads/GoodNotes

folders:
  inbox: 1 - Inbox
  tags: 3 - Tags
  attachments: attachments

tags:
  style: wikilink  # or "hashtag"

ocr:
  provider: openai_vision
  model: gpt-4o-mini

embedding:
  model: text-embedding-3-small
  chunk_size: 512
  chunk_overlap: 50

rag:
  top_k: 10
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  reranker_top_n: 5
  min_tags_threshold: 3
  min_confidence_threshold: 0.4

watcher:
  poll_interval: 30
```

---

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `vault_path` | path | **required** | Path to your Obsidian vault |
| `watch_folder` | path | — | Folder to poll for new PDFs |
| `folders.inbox` | string | `"1 - Inbox"` | Vault subfolder where new notes are written |
| `folders.tags` | string | `"3 - Tags"` | Vault subfolder containing tag files (wikilink style) |
| `folders.attachments` | string | `"attachments"` | Vault subfolder where diagram page images are saved |
| `tags.style` | string | `"wikilink"` | `"wikilink"` — tags are `.md` files in the tags folder; `"hashtag"` — tags are scanned from `#tag` usage across notes |
| `note_template` | string | see above | Note template string with `{date}`, `{time}`, `{title}`, `{content}`, `{tags}`, `{references}` placeholders |
| `ocr.provider` | string | `"openai_vision"` | `"openai_vision"` or `"google_vision"` |
| `ocr.model` | string | `"gpt-4o-mini"` | Model used for vision OCR |
| `embedding.model` | string | `"text-embedding-3-small"` | OpenAI embedding model for indexing |
| `embedding.chunk_size` | int | `512` | Token chunk size when indexing vault notes |
| `embedding.chunk_overlap` | int | `50` | Token overlap between consecutive chunks |
| `rag.top_k` | int | `10` | Number of candidates retrieved before reranking |
| `rag.reranker_model` | string | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model for reranking |
| `rag.reranker_top_n` | int | `5` | Results to keep after reranking |
| `rag.min_tags_threshold` | int | `3` | LLM fallback triggers if retrieval returns fewer tags than this |
| `rag.min_confidence_threshold` | float | `0.4` | LLM fallback triggers if top retrieval score is below this |
| `watcher.poll_interval` | int | `30` | Seconds between watch folder polls |
