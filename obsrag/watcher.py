"""Poll a folder for new PDFs and run the full pipeline on each."""
import json
import time
from pathlib import Path

def load_processed(log_path: Path) -> dict[str, str]:
    """Load processed fingerprints keyed by absolute source path."""
    if not log_path.exists():
        return {}
    raw = json.loads(log_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {}
    return {str(Path(path)): str(fingerprint) for path, fingerprint in raw.items()}


def save_processed(processed: dict[str, str], log_path: Path):
    """Save processed fingerprints to disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(processed, indent=2, sort_keys=True), encoding="utf-8")


def fingerprint_pdf(pdf_path: Path) -> str:
    """Build a stable fingerprint that changes when the source PDF changes."""
    stat = pdf_path.stat()
    resolved = pdf_path.resolve()
    return f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"


def get_new_pdfs(watch_folder: Path, processed: dict[str, str]) -> list[Path]:
    """Return any PDFs that are new or changed since last successful processing."""
    if not watch_folder.exists():
        print(f"Warning: Watch folder not found at {watch_folder}")
        return []
    pdfs = []
    for pdf_path in sorted(watch_folder.glob("*.pdf")):
        path_key = str(pdf_path.resolve())
        fingerprint = fingerprint_pdf(pdf_path)
        if processed.get(path_key) != fingerprint:
            pdfs.append(pdf_path)
    return pdfs


def watch_loop(process_fn, watch_folder: Path, log_path: Path, poll_interval: int = 30):
    """
    Poll watch_folder for new PDFs and run process_fn on each.

    Args:
        process_fn: Callable that takes a Path to a PDF and processes it.
        watch_folder: Folder to watch for new PDFs.
        poll_interval: Seconds between polls.
    """
    processed = load_processed(log_path)
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
                    processed[str(pdf_path.resolve())] = fingerprint_pdf(pdf_path)
                    save_processed(processed, log_path)
                    print(f"Recorded {pdf_path.name} as processed.")
                except Exception as e:
                    print(f"Error processing {pdf_path.name}: {e}")

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
