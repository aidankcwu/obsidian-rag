"""Configuration for Obsidian RAG project."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Obsidian vault path
VAULT_PATH = Path(
    "/Users/aidanwu/Library/CloudStorage/GoogleDrive-samuraishibe1@gmail.com/My Drive/Aidan's Vault"
)

# Persistence directory for the vector store index
PERSIST_DIR = Path(".obsrag/index")

# Number of top results to retrieve
TOP_K = 5

# Embedding model to use (OpenAI)
EMBEDDING_MODEL = "text-embedding-3-small"

# Watch folder for new PDFs from GoodNotes
WATCH_FOLDER = Path(
    "/Users/aidanwu/Library/CloudStorage/GoogleDrive-samuraishibe1@gmail.com/My Drive/[0] Notes Inbox"
)