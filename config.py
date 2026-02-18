"""Configuration loader for Obsidian RAG.

Reads settings from .obsrag.yaml (CWD first, then ~/.obsrag.yaml).
API keys are loaded from .env as before.
Config is loaded lazily via get_config() so that cli init works without a config file.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# --- Default values ---

DEFAULT_NOTE_TEMPLATE = """{date} {time}

Status: #review

Tags: {tags}

# {title}

{content}

## References
{references}
"""


@dataclass
class FoldersConfig:
    inbox: str = "1 - Inbox"
    tags: str = "3 - Tags"


@dataclass
class TagsConfig:
    style: str = "wikilink"  # "wikilink" or "hashtag"


@dataclass
class OcrConfig:
    provider: str = "openai_vision"  # "openai_vision" or "google_vision"
    model: str = "gpt-4o-mini"


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class RagConfig:
    top_k: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 5
    min_tags_threshold: int = 3
    min_confidence_threshold: float = 0.4


@dataclass
class WatcherConfig:
    poll_interval: int = 30


@dataclass
class Config:
    vault_path: Path = None
    watch_folder: Path = None
    persist_dir: Path = field(default_factory=lambda: Path(".obsrag/index"))
    folders: FoldersConfig = field(default_factory=FoldersConfig)
    tags: TagsConfig = field(default_factory=TagsConfig)
    note_template: str = DEFAULT_NOTE_TEMPLATE
    ocr: OcrConfig = field(default_factory=OcrConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)

    @property
    def inbox_path(self) -> Path:
        return self.vault_path / self.folders.inbox

    @property
    def tags_folder(self) -> Path:
        return self.vault_path / self.folders.tags


def _find_config_file() -> Path | None:
    """Look for .obsrag.yaml in CWD, then home directory."""
    candidates = [
        Path.cwd() / ".obsrag.yaml",
        Path.home() / ".obsrag.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_config() -> Config:
    """Load config from YAML file, applying defaults for missing values."""
    path = _find_config_file()
    if path is None:
        raise FileNotFoundError(
            "No .obsrag.yaml found. Run 'python cli.py init' to create one."
        )

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    vault_path = raw.get("vault_path")
    if not vault_path:
        raise ValueError("vault_path is required in .obsrag.yaml")

    cfg = Config(
        vault_path=Path(vault_path).expanduser(),
        persist_dir=Path(raw.get("persist_dir", ".obsrag/index")),
        note_template=raw.get("note_template", DEFAULT_NOTE_TEMPLATE),
    )

    if raw.get("watch_folder"):
        cfg.watch_folder = Path(raw["watch_folder"]).expanduser()

    # Nested configs
    if "folders" in raw:
        cfg.folders = FoldersConfig(**{k: v for k, v in raw["folders"].items()})

    if "tags" in raw:
        cfg.tags = TagsConfig(**{k: v for k, v in raw["tags"].items()})

    if "ocr" in raw:
        cfg.ocr = OcrConfig(**{k: v for k, v in raw["ocr"].items()})

    if "embedding" in raw:
        cfg.embedding = EmbeddingConfig(**{k: v for k, v in raw["embedding"].items()})

    if "rag" in raw:
        cfg.rag = RagConfig(**{k: v for k, v in raw["rag"].items()})

    if "watcher" in raw:
        cfg.watcher = WatcherConfig(**{k: v for k, v in raw["watcher"].items()})

    return cfg


# Lazy singleton
_cfg: Config | None = None


def get_config() -> Config:
    """Get the loaded config, parsing .obsrag.yaml on first call."""
    global _cfg
    if _cfg is None:
        _cfg = _load_config()
    return _cfg
