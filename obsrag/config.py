"""Configuration loader for Obsidian RAG."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

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
    attachments: str = "attachments"


@dataclass
class TagsConfig:
    style: str = "wikilink"  # "wikilink" or "hashtag"


@dataclass
class OcrConfig:
    provider: str = "openai_vision"
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
    config_path: Path = None
    state_dir: Path = field(default_factory=lambda: Path.home() / ".obsrag")
    persist_dir: Path = field(default_factory=lambda: Path.home() / ".obsrag" / "index")
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

    @property
    def attachments_path(self) -> Path:
        return self.vault_path / self.folders.attachments

    @property
    def manifest_path(self) -> Path:
        return self.persist_dir.parent / "manifest.json"

    @property
    def processed_log_path(self) -> Path:
        return self.state_dir / "processed.json"


def default_config_path() -> Path:
    """Return the default config path for new installs."""
    return Path.home() / ".obsrag.yaml"


def _find_config_file() -> Path | None:
    """Look for .obsrag.yaml in CWD, then home directory."""
    candidates = [
        Path.cwd() / ".obsrag.yaml",
        default_config_path(),
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
            f"No .obsrag.yaml found. Run 'obsrag init' to create one at {default_config_path()}."
        )

    load_dotenv(path.parent / ".env")
    load_dotenv()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    vault_path = raw.get("vault_path")
    if not vault_path:
        raise ValueError("vault_path is required in .obsrag.yaml")

    config_dir = path.parent
    state_dir_raw = raw.get("state_dir")
    if state_dir_raw:
        state_dir = Path(state_dir_raw).expanduser()
        if not state_dir.is_absolute():
            state_dir = (config_dir / state_dir).resolve()
    else:
        state_dir = Path.home() / ".obsrag"

    persist_dir_raw = raw.get("persist_dir")
    if persist_dir_raw:
        persist_dir = Path(persist_dir_raw).expanduser()
        if not persist_dir.is_absolute():
            persist_dir = (config_dir / persist_dir).resolve()
    else:
        persist_dir = state_dir / "index"

    vault_path = Path(vault_path).expanduser()
    if not vault_path.is_absolute():
        vault_path = (config_dir / vault_path).resolve()

    cfg = Config(
        config_path=path.resolve(),
        vault_path=vault_path,
        state_dir=state_dir,
        persist_dir=persist_dir,
        note_template=raw.get("note_template", DEFAULT_NOTE_TEMPLATE),
    )

    if raw.get("watch_folder"):
        cfg.watch_folder = Path(raw["watch_folder"]).expanduser()
        if not cfg.watch_folder.is_absolute():
            cfg.watch_folder = (config_dir / cfg.watch_folder).resolve()

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
