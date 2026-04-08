"""Runtime validation helpers for setup and user-facing errors."""
import os
import shutil


def validate_environment(
    cfg,
    *,
    require_openai: bool = True,
    require_poppler: bool = False,
    require_watch_folder: bool = False,
) -> None:
    """Raise a user-facing error when required setup is missing."""
    if cfg.ocr.provider != "openai_vision":
        raise ValueError(
            "Only 'openai_vision' is supported in this open-source release. "
            "Remove 'ocr.provider' from your config or set it to 'openai_vision'."
        )

    if cfg.vault_path is None or not cfg.vault_path.exists():
        raise ValueError(
            f"Vault path does not exist: {cfg.vault_path}. Update 'vault_path' in {cfg.config_path}."
        )

    if require_watch_folder and (cfg.watch_folder is None or not cfg.watch_folder.exists()):
        raise ValueError(
            f"Watch folder does not exist: {cfg.watch_folder}. Set 'watch_folder' in {cfg.config_path}."
        )

    if cfg.tags.style == "wikilink" and not cfg.tags_folder.exists():
        raise ValueError(
            f"Tags folder does not exist: {cfg.tags_folder}. "
            "Create it or switch 'tags.style' to 'hashtag'."
        )

    if require_openai and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your shell environment or to a .env file "
            f"next to {cfg.config_path}."
        )

    if require_poppler:
        has_poppler = shutil.which("pdftoppm") or shutil.which("pdfinfo")
        if not has_poppler:
            raise ValueError(
                "Poppler is required for PDF rendering but was not found on PATH. "
                "Install it with 'brew install poppler' on macOS or "
                "'sudo apt install poppler-utils' on Debian/Ubuntu."
            )
