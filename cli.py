"""CLI for Obsidian RAG.

Usage:
    python cli.py init              Interactive setup — generates .obsrag.yaml
    python cli.py build             Build or rebuild the vector index
    python cli.py process <pdf>     Process a single PDF through the pipeline
    python cli.py watch             Watch folder for new PDFs
"""
import shutil
from pathlib import Path

import click
import yaml


@click.group()
def cli():
    """Obsidian RAG — OCR handwritten notes into your Obsidian vault."""
    pass


@cli.command()
def init():
    """Interactive setup — generates .obsrag.yaml config file."""
    click.echo("Obsidian RAG Setup\n")

    vault_path = click.prompt(
        "Path to your Obsidian vault",
        type=click.Path(),
    )
    vault_path = str(Path(vault_path).expanduser())

    watch_folder = click.prompt(
        "Watch folder for new PDFs (leave blank to skip)",
        default="",
        show_default=False,
    )

    tag_style = click.prompt(
        "Tag style",
        type=click.Choice(["wikilink", "hashtag"]),
        default="wikilink",
    )

    inbox = click.prompt("Inbox folder name", default="1 - Inbox")
    tags_folder = click.prompt("Tags folder name", default="3 - Tags")

    ocr_provider = click.prompt(
        "OCR provider",
        type=click.Choice(["openai_vision", "google_vision"]),
        default="openai_vision",
    )

    config = {
        "vault_path": vault_path,
        "folders": {
            "inbox": inbox,
            "tags": tags_folder,
        },
        "tags": {
            "style": tag_style,
        },
        "ocr": {
            "provider": ocr_provider,
            "model": "gpt-4o-mini",
        },
        "embedding": {
            "model": "text-embedding-3-small",
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
        "rag": {
            "top_k": 10,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_top_n": 5,
            "min_tags_threshold": 3,
            "min_confidence_threshold": 0.4,
        },
        "watcher": {
            "poll_interval": 30,
        },
    }

    if watch_folder:
        config["watch_folder"] = str(Path(watch_folder).expanduser())

    out_path = Path(".obsrag.yaml")
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"\nConfig written to {out_path}")
    click.echo("Edit .obsrag.yaml to customize further, then run: python cli.py build")


@cli.command()
def build():
    """Build or rebuild the vector index."""
    from config import get_config
    from indexer import load_documents, build_or_load_index

    cfg = get_config()

    # Remove existing index to force rebuild
    if cfg.persist_dir.exists():
        shutil.rmtree(cfg.persist_dir)
        click.echo("Removed existing index.")

    docs = load_documents(cfg.vault_path)
    build_or_load_index(
        docs, cfg.persist_dir, cfg.embedding.model,
        chunk_size=cfg.embedding.chunk_size,
        chunk_overlap=cfg.embedding.chunk_overlap,
    )
    click.echo("Index built successfully.")


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
def process(pdf: Path):
    """Process a single PDF through the full pipeline."""
    from config import get_config
    from main import setup, process_pdf

    cfg = get_config()
    docs, index, tag_set, tag_context, reranker = setup(cfg)
    process_pdf(pdf, docs, index, tag_set, tag_context, reranker, cfg)


@cli.command()
def watch():
    """Watch folder for new PDFs and process them automatically."""
    from config import get_config
    from main import setup, process_pdf
    from watcher import watch_loop

    cfg = get_config()
    if not cfg.watch_folder:
        click.echo("Error: watch_folder not set in .obsrag.yaml")
        raise SystemExit(1)

    docs, index, tag_set, tag_context, reranker = setup(cfg)
    watch_loop(
        process_fn=lambda pdf: process_pdf(pdf, docs, index, tag_set, tag_context, reranker, cfg),
        watch_folder=cfg.watch_folder,
        poll_interval=cfg.watcher.poll_interval,
    )


if __name__ == "__main__":
    cli()
