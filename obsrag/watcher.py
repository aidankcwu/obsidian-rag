"""Poll a folder for new PDFs and run the full pipeline on each."""
import json
import time
from pathlib import Path

PROCESSED_LOG = Path(".obsrag/processed.json")


def load_processed() -> set[str]:
    """Load the set of already-processed filenames."""
    if PROCESSED_LOG.exists():
        return set(json.loads(PROCESSED_LOG.read_text()))
    return set()


def save_processed(processed: set[str]):
    """Save the set of processed filenames to disk."""
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_LOG.write_text(json.dumps(sorted(processed), indent=2))


def get_new_pdfs(watch_folder: Path, processed: set[str]) -> list[Path]:
    """Return any PDFs in watch_folder that haven't been processed yet."""
    if not watch_folder.exists():
        print(f"Warning: Watch folder not found at {watch_folder}")
        return []
    return [
        f for f in watch_folder.glob("*.pdf")
        if f.name not in processed
    ]


def watch_loop(process_fn, watch_folder: Path, poll_interval: int = 30):
    """
    Poll watch_folder for new PDFs and run process_fn on each.

    Args:
        process_fn: Callable that takes a Path to a PDF and processes it.
        watch_folder: Folder to watch for new PDFs.
        poll_interval: Seconds between polls.
    """
    processed = load_processed()
    print(f"Watching {watch_folder} for new PDFs (every {poll_interval}s)...")
    print(f"Already processed: {len(processed)} files")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            new_pdfs = get_new_pdfs(watch_folder, processed)

            for pdf_path in new_pdfs:
                print(f"\n{'='*50}")
                print(f"New PDF detected: {pdf_path.name}")
                print(f"{'='*50}")

                try:
                    process_fn(pdf_path)
                    processed.add(pdf_path.name)
                    save_processed(processed)
                    print(f"Marked {pdf_path.name} as processed.")
                except Exception as e:
                    print(f"Error processing {pdf_path.name}: {e}")

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
