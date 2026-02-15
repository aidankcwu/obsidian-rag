"""Poll a folder for new PDFs and run the full pipeline on each."""
import json
import time
from pathlib import Path
from config import WATCH_FOLDER

PROCESSED_LOG = Path(".obsrag/processed.json")
POLL_INTERVAL = 30  # seconds


def load_processed() -> set[str]:
    """Load the set of already-processed filenames."""
    if PROCESSED_LOG.exists():
        return set(json.loads(PROCESSED_LOG.read_text()))
    return set()


def save_processed(processed: set[str]):
    """Save the set of processed filenames to disk."""
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_LOG.write_text(json.dumps(sorted(processed), indent=2))


def get_new_pdfs(processed: set[str]) -> list[Path]:
    """Return any PDFs in WATCH_FOLDER that haven't been processed yet."""
    if not WATCH_FOLDER.exists():
        print(f"Warning: Watch folder not found at {WATCH_FOLDER}")
        return []
    return [
        f for f in WATCH_FOLDER.glob("*.pdf")
        if f.name not in processed
    ]


def watch_loop(process_fn):
    """
    Poll WATCH_FOLDER for new PDFs and run process_fn on each.

    Args:
        process_fn: Callable that takes a Path to a PDF and processes it.
                    Should be the fully initialized pipeline function.
    """
    processed = load_processed()
    print(f"Watching {WATCH_FOLDER} for new PDFs (every {POLL_INTERVAL}s)...")
    print(f"Already processed: {len(processed)} files")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            new_pdfs = get_new_pdfs(processed)

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

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
