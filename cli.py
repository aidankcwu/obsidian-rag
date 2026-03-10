"""CLI for Obsidian RAG.

Usage:
    python cli.py init              Interactive setup — generates .obsrag.yaml
    python cli.py build             Build or rebuild the vector index
    python cli.py process <pdf>     Process a single PDF through the pipeline
    python cli.py suggest <note>    Suggest additional tags for an existing note
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
    from obsrag.config import get_config
    from obsrag.rag.indexer import load_documents, build_or_load_index, _manifest_path

    cfg = get_config()

    # Remove existing index and manifest to force a clean rebuild
    if cfg.persist_dir.exists():
        shutil.rmtree(cfg.persist_dir)
        click.echo("Removed existing index.")
    manifest = _manifest_path(cfg.persist_dir)
    if manifest.exists():
        manifest.unlink()
        click.echo("Removed manifest.")

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
    from obsrag.config import get_config
    from obsrag.pipeline import setup, process_pdf

    cfg = get_config()
    docs, index, tag_set, tag_context, reranker = setup(cfg)
    process_pdf(pdf, docs, index, tag_set, tag_context, reranker, cfg)


@cli.command()
@click.argument("note", type=click.Path(exists=True, path_type=Path))
def suggest(note: Path):
    """Suggest additional tags for an existing Markdown note and insert them inline."""
    import re
    import subprocess
    from obsrag.config import get_config
    from obsrag.pipeline import setup
    from obsrag.rag.suggest import suggest_links_and_tags, suggest_tags_via_llm

    def notify(title: str, body: str):
        script = f'display notification "{body}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], check=False)

    try:
        cfg = get_config()
        docs, index, tag_set, tag_context, reranker = setup(cfg)

        text = note.read_text(encoding="utf-8")

        # Parse existing tags from the Tags: line
        tags_line_match = re.search(r"^Tags:(.*)$", text, re.MULTILINE)
        existing_tags: set[str] = set()
        tag_format = cfg.tags.style  # default format from config

        if tags_line_match:
            tags_line_content = tags_line_match.group(1)
            wikilink_tags = re.findall(r"\[\[([^\]]+)\]\]", tags_line_content)
            hashtag_tags = re.findall(r"#([\w/-]+)", tags_line_content)
            if wikilink_tags:
                existing_tags.update(wikilink_tags)
                tag_format = "wikilink"
            if hashtag_tags:
                existing_tags.update(hashtag_tags)
                if not wikilink_tags:
                    tag_format = "hashtag"

        # Run suggestion pipeline
        result = suggest_links_and_tags(
            text,
            index,
            tag_set,
            docs,
            reranker=reranker,
            top_k=cfg.rag.top_k,
        )
        retrieval_tags = [t["title"] for t in result["suggested_tags"]]
        top_score = result["suggested_links"][0]["score"] if result["suggested_links"] else 0

        if len(retrieval_tags) < cfg.rag.min_tags_threshold or top_score < cfg.rag.min_confidence_threshold:
            llm_result = suggest_tags_via_llm(
                note_text=text,
                all_tags=sorted(tag_set),
                retrieval_tags=retrieval_tags,
                filename=note.name,
                tag_context=tag_context,
            )
            all_suggested = llm_result.get("existing_tags", []) + llm_result.get("new_tags", [])
        else:
            all_suggested = retrieval_tags

        # Filter out already-assigned tags and pick top 2-3
        new_tags = [t for t in all_suggested if t not in existing_tags][:3]

        if not new_tags:
            notify("ObsRAG", "No additional tags to suggest")
            click.echo("No additional tags to suggest.")
            return

        # Format new tags to match existing style
        if tag_format == "wikilink":
            formatted = " ".join(f"[[{t}]]" for t in new_tags)
        else:
            formatted = " ".join(f"#{t}" for t in new_tags)

        # Append to the Tags: line (or add one if missing)
        if tags_line_match:
            original_line = tags_line_match.group(0)
            updated_line = original_line.rstrip() + " " + formatted
            updated_text = text[:tags_line_match.start()] + updated_line + text[tags_line_match.end():]
        else:
            updated_text = text.rstrip() + f"\nTags: {formatted}\n"

        note.write_text(updated_text, encoding="utf-8")

        tag_names = ", ".join(new_tags)
        notify("ObsRAG", f"Added tags: {tag_names}")
        click.echo(f"Added tags: {tag_names}")

    except Exception as e:
        notify("ObsRAG", str(e))
        raise


@cli.command()
def watch():
    """Watch folder for new PDFs and process them automatically."""
    from obsrag.config import get_config
    from obsrag.pipeline import setup, process_pdf
    from obsrag.watcher import watch_loop

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
