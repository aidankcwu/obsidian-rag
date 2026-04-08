# Configuration Reference

Obsidian RAG reads config from `./.obsrag.yaml` first, then `~/.obsrag.yaml`. Most users should keep a single config at `~/.obsrag.yaml`.

Runtime state defaults to:
- `~/.obsrag/index`
- `~/.obsrag/manifest.json`
- `~/.obsrag/processed.json`

## Example

```yaml
vault_path: ~/Documents/MyVault
watch_folder: ~/Downloads/GoodNotes

folders:
  inbox: 1 - Inbox
  tags: 3 - Tags
  attachments: attachments

tags:
  style: wikilink

ocr:
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

# Optional advanced overrides
# state_dir: ~/.obsrag
# persist_dir: ~/.obsrag/index
```

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `vault_path` | path | required | Path to your Obsidian vault |
| `watch_folder` | path | unset | Folder to poll for PDFs with `obsrag watch` |
| `state_dir` | path | `~/.obsrag` | Stable location for processed log and other runtime state |
| `persist_dir` | path | `~/.obsrag/index` | Vector index storage directory |
| `folders.inbox` | string | `"1 - Inbox"` | Vault subfolder where new notes are written |
| `folders.tags` | string | `"3 - Tags"` | Vault subfolder containing tag files in `wikilink` mode |
| `folders.attachments` | string | `"attachments"` | Vault subfolder where diagram page images are saved |
| `tags.style` | string | `"wikilink"` | `wikilink` uses files in the tags folder; `hashtag` scans existing notes for hashtags |
| `note_template` | string | built-in template | Template with `{date}`, `{time}`, `{title}`, `{content}`, `{tags}`, `{references}` |
| `ocr.model` | string | `"gpt-4o-mini"` | OpenAI vision model used for OCR |
| `embedding.model` | string | `"text-embedding-3-small"` | Embedding model used for indexing |
| `embedding.chunk_size` | int | `512` | Chunk size for indexed notes |
| `embedding.chunk_overlap` | int | `50` | Overlap between chunks |
| `rag.top_k` | int | `10` | Candidates retrieved before reranking |
| `rag.reranker_model` | string | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Sentence-transformer cross-encoder used for reranking |
| `rag.reranker_top_n` | int | `5` | Results kept after reranking |
| `rag.min_tags_threshold` | int | `3` | LLM fallback triggers if retrieval finds fewer tags |
| `rag.min_confidence_threshold` | float | `0.4` | LLM fallback triggers if the top retrieval score is low |
| `watcher.poll_interval` | int | `30` | Seconds between watch folder polls |

## Notes

- This public release supports only OpenAI vision OCR.
- If you have an older config with `ocr.provider`, set it to `openai_vision` or remove it.
- Relative `state_dir` and `persist_dir` values are resolved relative to the config file location.
